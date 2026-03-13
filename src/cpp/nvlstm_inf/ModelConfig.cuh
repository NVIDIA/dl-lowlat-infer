/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <optional>

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
  bool pre_compute = false;                               // enable pre-computation compliant with the "new" rules of the benchmark

  bool use_cublas_lt = false;                      // allows to fine-tune gemms
  bool paged_input = false;                        // use paged memory for input on GH
  bool skip_scalar_product_in_precompute = false;  // the computation is redundant in precompute
  bool as_supported = false;                       // if spinning kernels are used
  int as_suggested_cluster_size = 16;
  bool use_gdrcopy = false;                        // use gdrcopy for signaling in persistent kernels
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
