/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <iostream>

#include "ModelSingleStepGemv.cuh"
#include "fp16_reg_util.cuh"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

template <typename TData>
ModelSingleStepGemv<TData>::ModelSingleStepGemv(CudaEnv &cuda_env, const ModelDef &model_def, const ModelConfig &config)
    : Base{cuda_env, model_def, 1, false, true}  // full_windows=false, single_step=true
      ,
      cuda_env_{cuda_env},
      bias_{model_def.HasBias() ? model_def.Bias() : 0.0f},
      num_layers_(model_def.NumLayers()),
      num_steps_(model_def.NumSteps()),
      num_units_(model_def.NumUnits()),
      num_features_(model_def.NumFeatures()),
      in_graph_{config.lstm_l_in_graph},
      steps_per_graph_{config.steps_per_graph} {
  static_assert(NUM_GATES == kNumGates);

  switch (num_features_) {
    case LSTM_S_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_s_params_type>();
      break;
    case LSTM_M_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_m_params_type>();
      break;
    case LSTM_L_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_l_params_type>();
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_features: " + std::to_string(num_features_) +
          ". Expected: " + std::to_string(LSTM_S_NUM_INPUTS) + ", " + std::to_string(LSTM_M_NUM_INPUTS) + ", or " +
          std::to_string(LSTM_L_NUM_INPUTS));
  }

  model_singe_step_ = std::make_unique<ModelSingleStep<TData>>(cuda_env, model_def, config);
  // with --gdrcopy every kernel of this model reads a staged input from device memory (params.gmem_input, __ldcv)
  Base::UseDeviceInput();

  auto custom_stream = cuda_env.CustomStream();

  // Populate descriptor objects
  auto const num_units_gates = num_units_ * kNumGates;

#if AARCH64_GH == 1
  pinned_ref_out_ = cex::MallocPaged<float>(1);
#else
  pinned_ref_out_ = cex::MallocHost<float>(1);
#endif

  dev_input_weights_.resize(num_layers_);
  dev_biases_.resize(num_layers_);

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    dev_input_buf_ = cex::Malloc<data_type>(2 * std::max(num_features_, num_units_));
  }

  std::vector<TData> tmp_h(kNumGates * std::max(num_units_, num_features_) * num_units_);

  // FP16 register-resident kernel for lstm_l, if all its CTAs fit the device at once: the words its CTAs exchange, the
  // weights in its tile layout (fp16_reg/gemv_lstm_fp16_reg.h).
  const ssgemv::Fp16_reg_model_dims fp16_reg_dims{int(num_features_), int(num_units_), int(num_layers_)};
  if constexpr (kFp16RegL) {
    if (fp16_reg_dims == ssgemv::FP16_REG_LSTM_L_DIMS) {
      fp16_reg_l_ = ssgemv::verify_lstm_l_fp16_reg_occupancy_reqs();
      if (fp16_reg_l_) {
        dev_fp16_reg_xchg_ = fp16_reg_device_exchange(fp16_reg_dims);
      } else {
        std::cerr << "FP16 register-resident kernel: lstm_l's CTAs do not fit this device at once; the device-wide "
                     "kernel runs it"
                  << std::endl;
      }
    }
  }
  if (in_graph_) {
    // --lstm-l-in-graph: the FP16 register lstm_l kernel in its step's precompute graph (main.cu checks the rest)
    if (!USE_FP16_REG_L_GRAPH) throw std::invalid_argument("--lstm-l-in-graph needs a build with USE_FP16_REG_L_GRAPH");
    if (!fp16_reg_l_) throw std::invalid_argument("--lstm-l-in-graph needs the FP16 register lstm_l kernel");
    if (steps_per_graph_ == 0 || (steps_per_graph_ != 1 && steps_per_graph_ % 2 != 0))
      throw std::invalid_argument("--steps-per-graph must be 1 or even");
    dev_fp16_reg_x_dot_ = cex::Malloc<data_type>(kNumGates * num_units_);
  }

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    std::uint32_t num_inputs = (layer == 0) ? num_features_ : num_units_;

    // weights
    dev_biases_[layer] = cex::Malloc<data_type>(num_units_gates);

    auto &&matrix_W_conv = Convert(model_def.LstmW().at(layer).Data());
    if (fp16_reg_l_) {
      dev_input_weights_[layer] =
          fp16_reg_device_weights(fp16_reg_dims, int(layer), matrix_W_conv.data(), custom_stream);
    } else {
      dev_input_weights_[layer] = cex::Malloc<data_type>(num_units_gates * num_inputs);
      ssgemv::weights_permute(tmp_h.data(), matrix_W_conv.data(), kNumGates, num_inputs, num_units_);
      cex::Copy(dev_input_weights_[layer].get(), tmp_h.data(), num_units_gates * num_inputs, custom_stream);
    }

    // biases
    std::vector<float> biases_reduced(kNumGates * num_units_);
    for (std::size_t idx_gate = 0; idx_gate < kNumGates; ++idx_gate) {
      auto const gate = kGates.at(idx_gate);
      const float *biases1 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, false);
      const float *biases2 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, true);
      for (std::size_t k = 0; k < num_units_; ++k) {
        biases_reduced[idx_gate * num_units_ + k] = biases1[k] + biases2[k];
      }
    }
    auto &&bias_conv = Convert(biases_reduced);
    ssgemv::gate_transpose(tmp_h.data(), bias_conv.data(), kNumGates * num_units_, 1);
    cex::Copy(dev_biases_[layer].get(), tmp_h.data(), kNumGates * num_units_, custom_stream);

    // dot product
    dev_mul_ = cex::Malloc<float>(num_units_);
    cex::Copy(dev_mul_, model_def.MatMul().Data().data(), num_units_, custom_stream);
  }  // end layer loop

  GetParamsPtr()->max_smem_size = config.max_shared_mem_per_block;
  GetParamsPtr()->sm_count = config.sm_count;
  GetParamsPtr()->sm_arch = config.sm_arch;
};

template <typename TData>
template <typename Params>
void ModelSingleStepGemv<TData>::InitParams(Params &params) {
  Base::InitParams();

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    params.gmem_weights[layer] = dev_input_weights_[layer].get();
    params.gmem_biases[layer] = dev_biases_[layer].get();

    const auto &dev_gemms = model_singe_step_->Gemms();
    std::size_t leading_dim = dev_gemms[layer].LeadingDim();

    params.gmem_gemm_h[layer] = (void *)(dev_gemms[layer].Get() + (num_steps_ - 1) * leading_dim);

    const auto &dev_cxs_in =
        (odd_step_ && Base::IsCapturing()) ? model_singe_step_->CxOut() : model_singe_step_->CxIn();
    leading_dim = dev_cxs_in[layer].LeadingDim();
    params.gmem_c_in[layer] = (void *)(dev_cxs_in[layer].Get() + (num_steps_ - 1) * leading_dim);
  }
  params.dot_prod_bias = bias_;
  params.gmem_dot_prod_weights = (void *)dev_mul_.get();
  params.host_ld_complete = Base::HostSignal();
  params.gmem_xchg = dev_fp16_reg_xchg_.get();

  params.gmem_ld_complete = Base::DeviceSignal();
  params.use_gdrcopy = Env().UseGdrcopy();

  // Todo move to base
  params.host_addr = (Base::IsStaging()) ? nullptr : Base::PinedInputAddr();

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    params.gmem_h_out = dev_input_buf_.get();
  }

  Base::SetKernelInput(params);
}

template <typename TData>
void ModelSingleStepGemv<TData>::InitParams() {
  switch (num_features_) {
    case LSTM_S_NUM_INPUTS:
      InitParams(*static_cast<lstm_s_params_type *>(params_ptr_.get()));
      break;
    case LSTM_M_NUM_INPUTS:
      InitParams(*static_cast<lstm_m_params_type *>(params_ptr_.get()));
      break;
    case LSTM_L_NUM_INPUTS:
      InitParams(*static_cast<lstm_l_params_type *>(params_ptr_.get()));
      break;
    default:
      break;
  }
}

template <typename TData>
void ModelSingleStepGemv<TData>::PreCompute() {
  if (Base::GraphPtr() == nullptr) {
    model_singe_step_->SetPinnedOut(pinned_ref_out_.get());
    model_singe_step_->PreProcess();
    if (Base::IsStaging()) {
      // Unliqely to happen without capturing the graph
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputPtr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const *>(Base::HostInputPtr()));
      }
    } else {
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputAddr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const **>(Base::HostInputAddr()));
      }
    }
    model_singe_step_->Output(pinned_ref_out_.get());
    model_singe_step_->PostProcess();
  } else {
    StateCheck(graph_with_kernel_[0] == nullptr);  // that graph has the kernel at its root: PreProcess launches it
    model_singe_step_->PreProcess();
    model_singe_step_->PreProcessWait();
    model_singe_step_->Launch(*Base::GraphPtr());
    model_singe_step_->InputWait();
    model_singe_step_->OutputWait();
    model_singe_step_->PostProcess();
  }
};

template <typename TData>
void ModelSingleStepGemv<TData>::PreProcess() {
#if USE_FP16_REG_L_GRAPH
  if (graph_with_kernel_[0] != nullptr) {
    // --lstm-l-in-graph: the step's kernel and its precompute in one graph (every steps_per_graph_ steps), behind the
    // last one in the stream (and any InitX), from the first step on (the graphs are captured before it, GraphPtr() is
    // set by the first Launch); with several steps per graph each kernel waits for the host's reset.
    ResetSignalForLaunch();
    if (steps_in_graph_ == 0) {
      model_singe_step_->PreProcess();
      model_singe_step_->PreProcessWait();
      model_singe_step_->LaunchExec(graph_with_kernel_, steps_per_graph_);
      model_singe_step_->PostProcess();
    }
    steps_in_graph_ = (steps_in_graph_ + 1) % steps_per_graph_;
    graph_launched_ = true;
    ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
        (uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
    return;
  }
#endif
  if (Base::GraphPtr() == nullptr) {
    Base::PreProcess();
    return;
  }
  // With the captured precompute graph the host does not wait between the steps: the step's kernel and its graph go on
  // the precompute's stream (this model runs without green contexts, so it is the whole device), whose order makes the
  // kernel wait for the precompute so far (the last step's graph, an InitX) and the graph for the kernel, which exits
  // only after the host has staged the step's input and the kernel has used the states. Both launches overlap the last
  // step's graph; the host waits for the kernel's ready signal only.
  InitParams();
  KernelLaunch(cuda_env_.Stream());
  model_singe_step_->PreProcess();
  model_singe_step_->PreProcessWait();
  model_singe_step_->Launch(*Base::GraphPtr());
  model_singe_step_->PostProcess();
  graph_launched_ = true;

  // Wait till data is loaded by the kernel
  ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
      (uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
}

template <typename TData>
void ModelSingleStepGemv<TData>::PostProcess() {
  // the step's graph went with its kernel (PreProcess), except before the graph was captured
  if (!graph_launched_) PreCompute();
  graph_launched_ = false;
};

template <typename TData>
bool ModelSingleStepGemv<TData>::PostProcessReady() {
  return model_singe_step_->PostProcessReady();
}

template <typename TData>
void ModelSingleStepGemv<TData>::PostProcessWait() {
  model_singe_step_->PostProcessWait();
}

}  // namespace nvidia::quant::lowlat_infer

template class nvidia::quant::lowlat_infer::ModelSingleStepGemv<float>;
template class nvidia::quant::lowlat_infer::ModelSingleStepGemv<half>;
