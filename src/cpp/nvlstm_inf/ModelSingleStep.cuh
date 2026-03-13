/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "CudaEnv.cuh"
#include "LstmConstants.hpp"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "ModelSingleStepBase.cuh"
#include "cuda_helpers.cuh"

namespace nvidia::quant::lowlat_infer {

// Supported data types are: float, half
template <typename TData = float>
class ModelSingleStep : public ModelSingleStepBase<TData> {
 public:
  using Base = ModelSingleStepBase<TData>;
  using Base::BatchSize;
  using Base::NumFeatures;
  using Base::NumSteps;
  using Base::NumUnits;
  using typename Base::data_type;

  ModelSingleStep(
      CudaEnv& cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
      ModelDef const& model_def,
      ModelConfig const& config = {});

  ModelSingleStep(
      CudaEnv& cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
      ModelDef const& model_def,
      std::uint32_t batch_size,
      ModelConfig const& config = {})
      : ModelSingleStep(cuda_env, model_def, config) {
    StateCheck(batch_size == 1);
    StateCheck(
        (!config.algorithm.has_value()) || (config.algorithm == ModelConfig::Algo::SingleStep) ||
        (config.algorithm == ModelConfig::Algo::SingleStepStreaming));
  }

  ~ModelSingleStep();

  // Run the Lstm inference and store results in dev_y_.
  void LstmForward() override;

  void InitX(util::Tensor32 const& t) override;

  void Aggregate() override;

  void PostProcess() override;
  bool PostProcessReady() override;
  void PostProcessWait() override;

  // Getters for the precomputed data structures and input weights
  // required by the SingLstepGemv implementation
  [[nodiscard]] const std::vector<cex::DeviceArray2D<TData>>& Gemms() const { return dev_gemms_; };
  [[nodiscard]] const std::vector<cex::DeviceArray2D<TData>>& CxIn() const { return dev_cxs_in_; };
  [[nodiscard]] const std::vector<cex::DeviceArray2D<TData>>& CxOut() const { return dev_cxs_out_; };

 protected:
  using Base::AllowTf32;
  using Base::Convert;
  using Base::DevXExpanded;
  using Base::DevY;
  using Base::Env;

 private:
  void ClearHiddenState();

  void PrepareHiddenState(cudaStream_t stream);

  std::uint32_t num_hidden_rows_;
  std::uint32_t output_row_offset_;
  std::uint32_t num_layers_;
  bool pre_compute_;

  cex::EventPtr event_inf_start_, event_post_process_;

  // Pointers to the device memory with weight matrices applied to x
  std::vector<cex::DeviceArray2D<data_type>> dev_input_weights_;
  // Pointers to the device memory with weight matrices applied to h
  std::vector<cex::DeviceArray2D<data_type>> dev_recurrent_weights_;
  // Pointers to the device memory with biases
  std::vector<cex::DevicePtr<data_type>> dev_biases_;

  // device memory pointers
  cex::DeviceArray2D<data_type> dev_gemm_x_;
  std::vector<cex::DeviceArray2D<data_type>> dev_gemms_;
  std::vector<cex::DeviceArray2D<data_type>> dev_hxs_;
  std::vector<cex::DeviceArray2D<data_type>> dev_cxs_in_, dev_cxs_out_;

  std::vector<cex::EventPtr> hxs_done_;

  // batched gemm interface
  cex::DevicePtr<data_type*> dev_recurrent_weights_ptrs_;
  cex::DevicePtr<data_type*> dev_hxs_ptrs_;
  cex::DevicePtr<data_type*> dev_gemms_ptrs_;

  // cublasLt interface
  bool use_cublas_lt_ = false;
  cex::DevicePtr<char*> workspace_lt_;
  size_t workspace_lt_bytes_ = 1024 * 1024 * 1024;
  static constexpr int MAX_HEUR_RES_ = 4;
  static constexpr int ALGO_NUM_ = 2;
  cublasLtMatmulHeuristicResult_t heuristic_results_lt_[2][MAX_HEUR_RES_];
  cublasLtMatmulAlgo_t algos_lt_[2];
  // descriptors
  cublasLtMatmulDesc_t operation_lt_desc_;
  std::vector<cublasLtMatrixLayout_t> a_desc_lt_;
  std::vector<cublasLtMatrixLayout_t> b_desc_lt_;
  std::vector<cublasLtMatrixLayout_t> c_desc_lt_;
  std::vector<cublasLtMatrixLayout_t> d_desc_lt_;
};

}  // namespace nvidia::quant::lowlat_infer

extern template class nvidia::quant::lowlat_infer::ModelSingleStep<float>;
extern template class nvidia::quant::lowlat_infer::ModelSingleStep<half>;
