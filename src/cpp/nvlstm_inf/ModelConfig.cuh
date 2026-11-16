/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ss_gemv_impl/kernel_traits.h"

namespace nvidia::quant::lowlat_infer {

struct ModelConfig final {
  enum class Algo {
    SingleStep,
    SingleStepStreaming,
    SingleStepGemv,
    PingPongPersistent,
  };

  std::optional<Algo> algorithm{};
  ssgemv::Data_kind data_type = ssgemv::DATA_TYPE_FP16;  // FP16, FP32, or BF16
  bool pre_compute = false;  // enable pre-computation compliant with the "new" rules of the benchmark

  bool use_cublas_lt = false;  // allows to fine-tune gemms
  /// Cache file of tuned cublasLt algorithms (nvLstmSingleStepGemmTune, single_step/lt_algo_cache.h); empty: none.
  std::string lt_algo_cache;
  bool paged_input = false;   // use paged memory for input on GH
  bool as_supported = false;  // if spinning kernels are used
  int as_suggested_cluster_size = 16;
  bool use_gdrcopy = false;  // use gdrcopy for signaling in persistent kernels
  /// FP16 lstm_m runs with the FP16 register-resident kernel (USE_FP16_REG_KERNELS); main.cu decides, as it depends on
  /// the number of instances that fit.
  bool fp16_reg_m = false;
  // the precompute graph runs on a single stream (no fork/join nodes); main.cu enables it for persistent kernels
  bool precompute_single_stream = false;
  /// The persistent FP16 lstm_s / lstm_m models' precompute is the resident SingleStep kernel
  /// (USE_SINGLE_STEP_RESIDENT, single_step/single_step_resident.cuh), which the inference kernel waits on, instead of
  /// the cuBLAS graph per step.
  bool precompute_resident = false;
  /// Its CTAs over the window steps per tile of units (--precompute-step-splits): more CTAs per pass, a shorter pass.
  int precompute_step_splits = 1;
  /// The persistent FP16 lstm_l model's FP16 register kernel is captured at the root of each step's precompute graph,
  /// which takes layer 0's W * x from it (USE_FP16_REG_L_GRAPH, --lstm-l-in-graph); steps_per_graph steps (1 or even)
  /// per graph launch (--steps-per-graph).
  bool lstm_l_in_graph = false;
  unsigned steps_per_graph = 1;
  bool ping_pong_model_persistent = false;
  int max_shared_mem_per_block = 0;
  int sm_count = 0;
  int sm_arch = 0;

  [[nodiscard]] constexpr ssgemv::Data_kind Type() const { return data_type; }

  // Backward compatibility: return 16 or 32
  [[nodiscard]] constexpr std::uint8_t Precision() const {
    return (static_cast<int>(data_type) == 2) ? 32 : 16;  // DATA_TYPE_FP32 = 2
  }
};

}  // namespace nvidia::quant::lowlat_infer
