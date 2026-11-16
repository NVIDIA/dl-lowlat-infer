/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_FP16_REG_UTIL_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_FP16_REG_UTIL_CUH_

// The models' side of the FP16 register-resident kernels (ss_gemv_impl/fp16_reg/).

#include <cuda_runtime.h>
#include <vector>

#include "cuda_helpers.cuh"
#include "ss_gemv_impl/fp16_reg/gemv_lstm_fp16_reg.h"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

/// Layer `layer`'s input weights (as the model stores them, converted to TData) in the tile layout of the FP16
/// register-resident kernel (gemv_lstm_fp16_reg.h), on the device.
template <typename TData>
cex::DevicePtr<TData> fp16_reg_device_weights(
    ssgemv::Fp16_reg_model_dims dims, int layer, TData const *weights, cudaStream_t stream) {
  auto const elems = ssgemv::fp16_reg_weight_elems(dims, layer);
  StateCheck(elems > 0);
  std::vector<TData> tile_w(elems);
  StateCheck(ssgemv::fp16_reg_permute_weights(tile_w.data(), weights, dims, layer));
  auto dev = cex::Malloc<TData>(elems);
  cex::Copy(dev.get(), tile_w.data(), elems, stream);
  return dev;
}

/// The words the CTAs of the FP16 register-resident kernel exchange for a model (params.gmem_xchg), on the device; none
/// (an empty pointer) for a model on one CTA.
inline cex::DevicePtr<unsigned long long> fp16_reg_device_exchange(ssgemv::Fp16_reg_model_dims dims) {
  const auto words = ssgemv::fp16_reg_exchange_words(dims);
  return words > 0 ? cex::Malloc<unsigned long long>(words) : cex::DevicePtr<unsigned long long>{};
}

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_FP16_REG_UTIL_CUH_
