/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// FP16 gate activations for the FP16 register-resident kernels (Fp16_reg_tile_config::fast_eltwise): tanh.approx.f16
// for tanh, and sigma(x) = (1 + tanh(x / 2)) / 2 for the sigmoid, in FP16 throughout.

#include <cuda_fp16.h>

#include "../gemv_lstm_utils.cuh"

namespace nvidia::quant::lowlat_infer::ssgemv {

/// The sigmoid in FP16 through tanh.approx: 0.5 * tanh(0.5 * in) + 0.5.
static __forceinline__ __device__ __half sigma_g_16bit_fast(__half in) {
  const __half half_v = __float2half(0.5f);
  return __hfma(half_v, htanh_approx(half_v * in), half_v);
}
/// tanh in FP16 (tanh.approx.f16).
static __forceinline__ __device__ __half sigma_c_16bit_fast(__half in) { return htanh_approx(in); }

/// lstm_eltwise (gemv_lstm_utils.cuh) with the activations above: 4 gates and the cell state in, hidden and cell out.
template <Data_kind DATA_KIND>
__forceinline__ __device__ void lstm_eltwise_fast(
    float& hx,
    float& cx,
    typename Data_type<DATA_KIND>::Elem4_type acc,
    typename Data_type<DATA_KIND>::Elem_type c_in) {
  static_assert(DATA_KIND == DATA_TYPE_FP16, "the gates are computed in FP16: FP16 data only");
  union {
    uint2 packed;
    __half h[4];
  } gate;
  gate.packed = acc;

  __half h_i = sigma_g_16bit_fast(gate.h[0]);
  __half h_f = sigma_g_16bit_fast(gate.h[1]);
  __half h_g = sigma_c_16bit_fast(gate.h[2]);
  __half h_o = sigma_g_16bit_fast(gate.h[3]);

  __half cx_h = h_f * __ushort_as_half(c_in) + h_i * h_g;
  __half hx_h = h_o * sigma_c_16bit_fast(cx_h);

  cx = __half2float(cx_h);
  hx = __half2float(hx_h);
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv
