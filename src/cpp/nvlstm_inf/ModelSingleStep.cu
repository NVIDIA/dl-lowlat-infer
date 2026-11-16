/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cstdio>
#include <cub/cub.cuh>

#include "ModelSingleStep.cuh"
#include "single_step/lt_algo_cache.h"
#include "single_step/single_step.cuh"
#include "ss_gemv_impl/gemv_lstm.h"
#include "ss_gemv_impl/gemv_lstm_utils.cuh"
#include "ss_gemv_impl/kernel_params.h"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

namespace {

/**
 * Determine if model is streaming or not
 */
[[nodiscard]] bool isStreaming(const ModelConfig& config) noexcept {
  return config.algorithm == ModelConfig::Algo::SingleStepStreaming;
}

/**
 * Number of hidden rows needed depending on configuration
 */
[[nodiscard]] std::uint32_t getNumHiddenRows(const ModelConfig& config, std::uint32_t numSteps) noexcept {
  if (isStreaming(config)) {
    return 1;
  } else {
    return numSteps + 1;
  }
}

/**
 * Offset for output depending on configuration
 */
[[nodiscard]] std::uint32_t getOutputRowOffset(const ModelConfig& config) noexcept {
  if (isStreaming(config)) {
    return 0;
  } else {
    return 1;
  }
}

}  // namespace

template <typename TData>
ModelSingleStep<TData>::ModelSingleStep(CudaEnv& cuda_env, const ModelDef& model_def, const ModelConfig& config)
    : Base{cuda_env, model_def, 1, {.streaming = isStreaming(config), .single_stream = config.precompute_single_stream}},
      num_hidden_rows_{getNumHiddenRows(config, model_def.NumSteps())},
      output_row_offset_{getOutputRowOffset(config)},
      num_layers_{model_def.NumLayers()},
      pre_compute_{config.pre_compute},
      // cublasLt runs only the input GEMM of layers >= 1 (K = units); a single-layer model has none.
      use_cublas_lt_{config.use_cublas_lt && model_def.NumLayers() > 1} {
  StateCheck(BatchSize() == 1);

  cuda_env.CreateStreams(1);
  event_inf_start_ = cex::CreateEvent();
  event_post_process_ = cex::CreateEvent();

  // Populate descriptor objects
  auto const num_steps = NumSteps();
  auto const num_features = NumFeatures();
  auto const num_units = NumUnits();
  auto const num_units_gates = num_units * kNumGates;
  auto stream = cuda_env.Stream();

  dev_gemm_x_ = cex::Malloc2D<data_type>(num_steps, num_units_gates);

  // Set up weights and bias parameters
  dev_input_weights_.resize(num_layers_);
  dev_recurrent_weights_.resize(num_layers_);
  dev_biases_.resize(num_layers_);
  dev_gemms_.resize(num_layers_);
  dev_hxs_.resize(num_layers_);
  dev_cxs_in_.resize(num_layers_);
  dev_cxs_out_.resize(num_layers_);
  hxs_done_.resize(num_layers_);

  // batched gemm
  std::vector<data_type*> dev_recurrent_weights_ptrs(num_layers_);
  std::vector<data_type*> dev_hxs_ptrs(num_layers_);
  std::vector<data_type*> dev_gemms_ptrs(num_layers_);
  dev_recurrent_weights_ptrs_ = cex::Malloc<data_type*>(num_layers_);
  dev_hxs_ptrs_ = cex::Malloc<data_type*>(num_layers_);
  dev_gemms_ptrs_ = cex::Malloc<data_type*>(num_layers_);

  std::vector<TData> tmp_h(kNumGates * std::max((uint32_t)num_units, num_features) * num_units);

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    auto const input_cols = layer == 0 ? num_features : num_units;
    auto& input_weights_layer = dev_input_weights_[layer];
    input_weights_layer = cex::Malloc2D<data_type>(num_units_gates, input_cols);
    auto& recurrent_weights_layer = dev_recurrent_weights_[layer];
    recurrent_weights_layer = cex::Malloc2D<data_type>(num_units_gates, num_units);
    dev_biases_[layer] = cex::Malloc<data_type>(num_units_gates);
    dev_gemms_[layer] = cex::Malloc2D<data_type>(num_steps, num_units_gates);
    StateCheck(dev_gemms_[layer].LeadingDim() == dev_gemm_x_.LeadingDim());
    dev_hxs_[layer] = cex::Malloc2D<data_type>(num_hidden_rows_, num_units);
    dev_cxs_in_[layer] = cex::Malloc2D<data_type>(num_hidden_rows_, num_units);
    dev_cxs_out_[layer] = cex::Malloc2D<data_type>(num_hidden_rows_, num_units);

    auto&& matrix_W_conv = Convert(model_def.LstmW().at(layer).Data());
    auto&& matrix_R_conv = Convert(model_def.LstmR().at(layer).Data());

#if GEMM_H_VECTORIZED == 1
    {
      // preprocess weights to have (num_units, 4, num_features) layout
      std::uint32_t num_inputs = (layer == 0) ? num_features : num_units;
      ssgemv::weights_permute_v2(tmp_h.data(), matrix_W_conv.data(), kNumGates, num_inputs, num_units);
      CudaCheck(cudaMemcpy2DAsync(
          dev_input_weights_[layer].Get(),
          dev_input_weights_[layer].Pitch(),
          tmp_h.data(),
          num_inputs * sizeof(data_type),
          num_inputs * sizeof(data_type),
          num_units_gates,
          cudaMemcpyHostToDevice,
          stream));
      ssgemv::weights_permute_v2(tmp_h.data(), matrix_R_conv.data(), kNumGates, num_units, num_units);
      CudaCheck(cudaMemcpy2DAsync(
          dev_recurrent_weights_[layer].Get(),
          dev_recurrent_weights_[layer].Pitch(),
          tmp_h.data(),
          num_units * sizeof(data_type),
          num_units * sizeof(data_type),
          num_units_gates,
          cudaMemcpyHostToDevice,
          stream));
    }
#endif

#if GEMM_H_VECTORIZED == 1
    std::vector<float> biases_reduced(kNumGates * num_units);
#endif
    for (std::size_t idx_gate = 0; idx_gate < kNumGates; ++idx_gate) {
      auto const gate = kGates.at(idx_gate);
#if GEMM_H_VECTORIZED == 0
      auto input_row = input_weights_layer.AtRow(idx_gate * num_units);
      cex::Copy2D(input_row, matrix_W_conv.data() + model_def.OffsetW(gate, layer == 0), num_units, input_cols, stream);

      auto recurrent_row = recurrent_weights_layer.AtRow(idx_gate * num_units);
      cex::Copy2D(recurrent_row, matrix_R_conv.data() + model_def.OffsetR(gate), num_units, num_units, stream);
#endif

      const float* biases1 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, false);
      const float* biases2 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, true);
#if GEMM_H_VECTORIZED == 0
      std::vector<float> biases_reduced(num_units);
      for (std::size_t k = 0; k < num_units; ++k) {
        biases_reduced[k] = biases1[k] + biases2[k];
      }
      auto&& bias_conv = Convert(biases_reduced);
      cex::Copy(dev_biases_[layer].get() + idx_gate * num_units, bias_conv.data(), num_units, stream);
#else
      for (std::size_t k = 0; k < num_units; ++k) {
        biases_reduced[idx_gate * num_units + k] = biases1[k] + biases2[k];
      }
#endif
    }  // for idx_gate
#if GEMM_H_VECTORIZED == 1
    auto&& bias_conv = Convert(biases_reduced);
    ssgemv::gate_transpose(tmp_h.data(), bias_conv.data(), kNumGates * num_units, 1);
    cex::Copy(dev_biases_[layer].get(), tmp_h.data(), kNumGates * num_units, stream);
#endif

    hxs_done_[layer] = cex::CreateEvent();

    // batched gemm
    dev_recurrent_weights_ptrs[layer] = dev_recurrent_weights_[layer].Get();
    dev_hxs_ptrs[layer] = dev_hxs_[layer].Get();
    dev_gemms_ptrs[layer] = dev_gemms_[layer].Get();
  }  // for layer

  // batched gemm interface requires pointers in device memory
  cex::Copy(dev_recurrent_weights_ptrs_.get(), dev_recurrent_weights_ptrs.data(), num_layers_, stream);
  cex::Copy(dev_hxs_ptrs_.get(), dev_hxs_ptrs.data(), num_layers_, stream);
  cex::Copy(dev_gemms_ptrs_.get(), dev_gemms_ptrs.data(), num_layers_, stream);

  ClearHiddenState();
  CudaCheck(cudaStreamSynchronize(stream));

  if (use_cublas_lt_) {
    // cublasLst setup
    auto cublas_lt = Env().CublasLt();
    auto constexpr cuda_data_type = single_step::getCudaDataType<TData>();
    cublasComputeType_t constexpr cublas_compute_type =
        (cuda_data_type == CUDA_R_32F) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
    cublasLtMatmulPreference_t preference = nullptr;
    CublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    CublasCheck(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_lt_bytes_, sizeof(workspace_lt_bytes_)));
    auto scalar_data_type = cuda_data_type;
    CublasCheck(cublasLtMatmulDescCreate(&operation_lt_desc_, cublas_compute_type, scalar_data_type));

    cublasOperation_t transpose_a = CUBLAS_OP_T;
    cublasOperation_t transpose_b = CUBLAS_OP_N;
    CublasCheck(cublasLtMatmulDescSetAttribute(
        operation_lt_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)));
    CublasCheck(cublasLtMatmulDescSetAttribute(
        operation_lt_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)));
    // reset workspace size, since will recalculate after the heuristic
    workspace_lt_bytes_ = 0;
    a_desc_lt_.resize(2);
    b_desc_lt_.resize(2);
    c_desc_lt_.resize(2);
    d_desc_lt_.resize(2);

    auto const gemm_x = dev_gemm_x_.Get();
    auto const gemm_leading_dim = dev_gemm_x_.LeadingDim();
    for (int layer = 1; layer < 2; layer++) {
      auto const hidden_leading_dim = dev_hxs_[layer].LeadingDim();

      auto const isLayer0 = (layer == 0);
      auto const num_inputs_x = static_cast<int>(isLayer0 ? num_features : num_units);
      auto const leading_dim_x = static_cast<int>(isLayer0 ? num_features : hidden_leading_dim);
      int rows_a = static_cast<int>(kNumGates * num_units);  // rows op(A) and C, m
      int lda = dev_input_weights_[layer].LeadingDim();
      int cols_b = static_cast<int>(num_steps);  // cols op(B) and C, n
      int rows_b = num_inputs_x;                 // cols op(A), rows op(B). k
      int cols_a = rows_b;
      int ldb = leading_dim_x;
      int ldc = gemm_leading_dim;

      CublasCheck(cublasLtMatrixLayoutCreate(&a_desc_lt_[layer], cuda_data_type, cols_a, rows_a, lda));
      CublasCheck(cublasLtMatrixLayoutCreate(&b_desc_lt_[layer], cuda_data_type, rows_b, cols_b, ldb));
      CublasCheck(cublasLtMatrixLayoutCreate(&c_desc_lt_[layer], cuda_data_type, rows_a, cols_b, ldc));
      CublasCheck(cublasLtMatrixLayoutCreate(&d_desc_lt_[layer], cuda_data_type, rows_a, cols_b, ldc));

      int num_results = 0;
      CublasCheck(cublasLtMatmulAlgoGetHeuristic(
          cublas_lt,
          operation_lt_desc_,
          a_desc_lt_[layer],
          b_desc_lt_[layer],
          c_desc_lt_[layer],
          d_desc_lt_[layer],
          preference,
          MAX_HEUR_RES_,
          &heuristic_results_lt_[layer][0],
          &num_results));
      StateCheck(num_results > 0);

      CublasCheck(cublasLtMatmulAlgoCheck(
          cublas_lt,
          operation_lt_desc_,
          a_desc_lt_[layer],
          b_desc_lt_[layer],
          c_desc_lt_[layer],
          d_desc_lt_[layer],
          &(heuristic_results_lt_[layer][ALGO_NUM_].algo),
          &heuristic_results_lt_[layer][ALGO_NUM_]));

      if (!config.lt_algo_cache.empty()) UseTunedAlgorithm(config.lt_algo_cache, layer, rows_a, cols_b, rows_b);

      workspace_lt_bytes_ = std::max(workspace_lt_bytes_, heuristic_results_lt_[layer][ALGO_NUM_].workspaceSize);
    }
    workspace_lt_ = cex::Malloc<char*>(workspace_lt_bytes_);
  } else if (!config.lt_algo_cache.empty()) {
    std::fprintf(stderr, "[lt-algo-cache] not used: only half precision with more than one layer runs cublasLt\n");
  }
}

template <typename TData>
void ModelSingleStep<TData>::UseTunedAlgorithm(std::string const& path, int layer, int m, int n, int k) {
  auto constexpr cuda_data_type = single_step::getCudaDataType<TData>();
  cublasComputeType_t constexpr compute_type = (cuda_data_type == CUDA_R_32F) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
  int device_id = 0;
  CudaCheck(cudaGetDevice(&device_id));
  // The SMs this model's GEMMs run on: its precompute's green context, if any, otherwise the device.
  auto const sm_count = GetSmCountFromGreenStream(Env().Stream());
  auto const key = lt_cache::MakeKey(device_id, sm_count, cuda_data_type, compute_type, m, n, k);
  auto const entries = lt_cache::Load(path);
  auto const* hit = lt_cache::Find(entries, key);
  cublasLtMatmulHeuristicResult_t checked{};
  bool const usable = hit != nullptr && cublasLtMatmulAlgoCheck(
                                            Env().CublasLt(),
                                            operation_lt_desc_,
                                            a_desc_lt_[layer],
                                            b_desc_lt_[layer],
                                            c_desc_lt_[layer],
                                            d_desc_lt_[layer],
                                            &hit->algo,
                                            &checked) == CUBLAS_STATUS_SUCCESS;
  if (usable) {
    checked.algo = hit->algo;
    heuristic_results_lt_[layer][ALGO_NUM_] = checked;
  }
  std::fprintf(
      stderr,
      "[lt-algo-cache] m=%d n=%d k=%d on %d SMs: %s\n",
      m,
      n,
      k,
      key.sm_count,
      usable ? "tuned algorithm" : (hit == nullptr ? "no entry, heuristic" : "entry rejected, heuristic"));
}

template <typename TData>
ModelSingleStep<TData>::~ModelSingleStep() {
  if (use_cublas_lt_) {
    cublasLtMatmulDescDestroy(operation_lt_desc_);
    for (size_t ii = 0; ii < a_desc_lt_.size(); ii++) {
      cublasLtMatrixLayoutDestroy(a_desc_lt_[ii]);
      cublasLtMatrixLayoutDestroy(b_desc_lt_[ii]);
      cublasLtMatrixLayoutDestroy(c_desc_lt_[ii]);
      cublasLtMatrixLayoutDestroy(d_desc_lt_[ii]);
    }
  }
}

template <typename TData>
void ModelSingleStep<TData>::LstmForward() {
  auto const num_units = NumUnits();
  auto const num_steps = NumSteps();
  auto const num_features = NumFeatures();

  auto constexpr cuda_data_type = single_step::getCudaDataType<TData>();
  auto const compute_type = single_step::getComputeType<TData>(AllowTf32());

  auto const gemm_rows = static_cast<int>(kNumGates * num_units);
  auto const gemm_cols = static_cast<int>(num_steps);
  auto const gemm_leading_dim = dev_gemm_x_.LeadingDim();

  dim3 const dim_block{static_cast<unsigned int>(num_units)};
  dim3 const dim_grid{num_steps};

  // Without pre_compute_ the hidden-state GEMM for the next call runs here, on a forked stream unless everything stays
  // on the main stream. The persistent models always run without pre_compute_ (main.cu, ModelAsSingleStepGemv), so for
  // them SingleStream() alone decides; a standalone SingleStep model (--pre-compute=0) keeps the fork.
  bool const fork = !pre_compute_ && !Base::SingleStream();
  if (fork) {
    Env().ForkStreams(event_inf_start_.get());
  }
  auto stream = Env().Stream();
  auto const cublas = Env().Cublas();
  auto const cublas_lt = Env().CublasLt();
  CublasCheck(cublasSetStream(cublas, stream));
  auto const gemm_x = dev_gemm_x_.Get();

  for (std::uint32_t layer = 0; layer < num_layers_; ++layer) {
    auto const hidden_leading_dim = dev_hxs_[layer].LeadingDim();
    auto const gemm_h = dev_gemms_[layer].Get();

    auto const isLayer0 = layer == 0;
    auto const num_inputs_x = static_cast<int>(isLayer0 ? num_features : num_units);
    auto const leading_dim_x = static_cast<int>(isLayer0 ? num_features : hidden_leading_dim);
    if (isLayer0 && Base::SkipReplicate()) {
      auto const hx_out = dev_hxs_[layer].Row(output_row_offset_);
      auto const cx_in = dev_cxs_in_[layer].Get();
      auto const cx_out = dev_cxs_out_[layer].Row(output_row_offset_);
      auto const bias = dev_biases_[layer].get();
      single_step::launch_single_step_layer_0(
          dev_input_weights_[layer].Get(),
          dev_input_weights_[layer].LeadingDim(),
          Base::GetDevXLast(),
          num_features,
          Base::Layer0InputProduct(),
          DevY().get(),
          gemm_h,
          gemm_leading_dim,
          bias,
          cx_in,
          cx_out,
          hx_out,
          hidden_leading_dim,
          num_units,
          num_steps,
          layer == num_layers_ - 1,
          stream);
    } else {
      TData* const features{isLayer0 ? DevXExpanded().get() : dev_hxs_[layer - 1].Row(output_row_offset_)};
      if (!use_cublas_lt_) {
        CublasCheck(cublasGemmEx(
            cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            gemm_rows,     // rows op(A) and C
            gemm_cols,     // cols op(B) and C
            num_inputs_x,  // cols op(A), rows op(B)
            cex::ConstOne<data_type>(),
            dev_input_weights_[layer].Get(),  // A: 4 * num_units x num_inputs
            cuda_data_type,
            dev_input_weights_[layer].LeadingDim(),  // leading dimension A
            features,                                // B: num_steps x num_inputs
            cuda_data_type,
            leading_dim_x,  // leading dimension B
            cex::ConstZero<data_type>(),
            gemm_x,  // C: num_steps x 4 * num_units
            cuda_data_type,
            gemm_leading_dim,  // leading dimension C
            compute_type,
            CUBLAS_GEMM_DEFAULT));
      } else {
        uint16_t a = 0x3c00, b = 0x0;
        CublasCheck(cublasLtMatmul(
            cublas_lt,
            operation_lt_desc_,
            &a,                               // cex::ConstOne<data_type>(),
            dev_input_weights_[layer].Get(),  // A: 4 * num_units x num_inputs
            a_desc_lt_[1],
            features,  // B: num_steps x num_inputs
            b_desc_lt_[1],
            &b,      // cex::ConstZero<data_type>(),
            gemm_x,  // C: num_steps x 4 * num_units
            c_desc_lt_[1],
            gemm_x,  // C: num_steps x 4 * num_units
            d_desc_lt_[1],
            //&algos_lt_[1],
            &heuristic_results_lt_[1][ALGO_NUM_].algo,
            workspace_lt_.get(),
            workspace_lt_bytes_,
            stream));
      }

      auto const hx_out = dev_hxs_[layer].Row(output_row_offset_);
      auto const cx_in = dev_cxs_in_[layer].Get();
      auto const cx_out = dev_cxs_out_[layer].Row(output_row_offset_);
      auto const bias = dev_biases_[layer].get();
      CudaCheck(cudaGetLastError());  // reset errors
#if GEMM_H_VECTORIZED == 1
      single_step::launch_eltwise(
          DevY().get(),
          gemm_h,
          gemm_x,
          gemm_leading_dim,
          bias,
          cx_in,
          cx_out,
          hx_out,
          hidden_leading_dim,
          num_units,
          num_steps,
          layer == num_layers_ - 1,
          stream);
#else
      single_step::launch_rnn_lstm(
          num_units,
          num_steps,
          DevY().get(),
          Base::OutputTarget(),
          Base::DevMul(),
          Base::Bias(),
          gemm_h,
          gemm_x,
          gemm_leading_dim,
          bias,
          cx_in,
          cx_out,
          hx_out,
          hidden_leading_dim,
          num_units,
          num_steps,
          layer == num_layers_ - 1,
          stream);
#endif
    }
    CudaCheck(cudaGetLastError());  // detect synchronous errors
    CudaCheck(cudaEventRecord(hxs_done_[layer].get(), stream));
  }

  if (!pre_compute_) {
    // Preparation for the next call
    PrepareHiddenState(fork ? Env().Streams(0) : stream);
    Env().JoinStreams();  // join any remaining forked streams
  }
}

template <typename TData>
void ModelSingleStep<TData>::PrepareHiddenState(cudaStream_t stream) {
  auto const num_units = NumUnits();
  auto constexpr cuda_data_type = single_step::getCudaDataType<TData>();
  auto const compute_type = single_step::getComputeType<TData>(AllowTf32());
  auto const gemm_rows = static_cast<int>(kNumGates * num_units);
  auto const gemm_cols = static_cast<int>(NumSteps());
  auto const gemm_leading_dim = dev_gemm_x_.LeadingDim();

  auto const cublas = Env().Cublas();
  CublasCheck(cublasSetStream(cublas, stream));

  std::vector<data_type*> dev_recurrent_weights_ptrs(num_layers_);
  std::vector<data_type*> dev_hxs_ptrs(num_layers_);
  std::vector<data_type*> dev_gemms_ptrs(num_layers_);
  size_t dev_recurrent_weights_leading_dim = dev_recurrent_weights_[0].LeadingDim();
  size_t dev_hxs_leading_dim = dev_hxs_[0].LeadingDim();
  if (!pre_compute_) {
    for (std::uint32_t layer = 0; layer < num_layers_; ++layer) {
      CudaCheck(cudaStreamWaitEvent(stream, hxs_done_[layer].get()));
    }
  }
  for (std::uint32_t layer = 0; layer < num_layers_; ++layer) {
    dev_recurrent_weights_ptrs[layer] = dev_recurrent_weights_[layer].Get();
    dev_hxs_ptrs[layer] = dev_hxs_[layer].Get();
    dev_gemms_ptrs[layer] = dev_gemms_[layer].Get();
    StateCheck(dev_recurrent_weights_[layer].LeadingDim() == dev_recurrent_weights_leading_dim);
    StateCheck(dev_hxs_[layer].LeadingDim() == dev_hxs_leading_dim);
  }
  auto const num_inputs_h = static_cast<int>(num_units);
  CublasCheck(cublasGemmBatchedEx(
      cublas,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      gemm_rows,     // rows op(A) and C
      gemm_cols,     // cols op(B) and C
      num_inputs_h,  // cols op(A), rows op(B)
      cex::ConstOne<data_type>(),
      (const void**)dev_recurrent_weights_ptrs_.get(),  // A: 4 * num_units x num_units
      cuda_data_type,
      dev_recurrent_weights_leading_dim,  // leading dimension A
      (const void**)dev_hxs_ptrs_.get(),  // B: num_steps x num_units
      cuda_data_type,
      dev_hxs_leading_dim,  // leading dimension B
      cex::ConstZero<data_type>(),
      (void**)dev_gemms_ptrs_.get(),  // C: num_steps x 4 * num_units
      cuda_data_type,
      gemm_leading_dim,  // leading dimension C
      num_layers_,
      compute_type,
      CUBLAS_GEMM_DEFAULT));
}

template <typename TData>
void ModelSingleStep<TData>::InitX(util::Tensor32 const& t) {
  ClearHiddenState();
  Base::InitX(t);
}

template <typename TData>
void ModelSingleStep<TData>::Aggregate() {
  // rnn_lstm aggregates non-vectorized multi-layer models; other paths require a separate kernel.
  if (num_layers_ == 1 || GEMM_H_VECTORIZED == 1) {
    Base::Aggregate();
  }
  for (std::uint32_t layer = 0; layer < num_layers_; ++layer) {
    dev_cxs_in_[layer].Array().swap(dev_cxs_out_[layer].Array());
  }
}

template <typename TData>
void ModelSingleStep<TData>::ClearHiddenState() {
  auto stream = Env().Stream();
  auto const num_units_bytes = NumUnits() * sizeof(data_type);
  for (std::uint32_t layer = 0; layer < num_layers_; ++layer) {
    auto& hxs = dev_hxs_[layer];
    auto& cxs_in = dev_cxs_in_[layer];
    auto& cxs_out = dev_cxs_out_[layer];
    CudaCheck(cudaMemset2DAsync(hxs.Get(), hxs.Pitch(), 0, num_units_bytes, num_hidden_rows_, stream));
    CudaCheck(cudaEventRecord(hxs_done_[layer].get(), stream));
    CudaCheck(cudaMemset2DAsync(cxs_in.Get(), cxs_in.Pitch(), 0, num_units_bytes, num_hidden_rows_, stream));
    CudaCheck(cudaMemset2DAsync(cxs_out.Get(), cxs_out.Pitch(), 0, num_units_bytes, num_hidden_rows_, stream));
  }
  PrepareHiddenState(stream);
}

template <typename TData>
void ModelSingleStep<TData>::PostProcess() {
  if (pre_compute_) {
    auto stream = Env().Stream();
    PrepareHiddenState(stream);
    CudaCheck(cudaEventRecord(event_post_process_.get(), stream));
  }
}

template <typename TData>
bool ModelSingleStep<TData>::PostProcessReady() {
  return !pre_compute_ || cex::EventReady(event_post_process_.get());
}

template <typename TData>
void ModelSingleStep<TData>::PostProcessWait() {
  if (pre_compute_) {
    CudaCheck(cudaEventSynchronize(event_post_process_.get()));
  }
}

}  // namespace nvidia::quant::lowlat_infer

template class nvidia::quant::lowlat_infer::ModelSingleStep<float>;
template class nvidia::quant::lowlat_infer::ModelSingleStep<half>;
