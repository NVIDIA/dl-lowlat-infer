/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Host side of the FP16 register-resident kernel (gemv_lstm_fp16_reg_impl.cu; launched with run_lstm_s_fp16_reg_t and
// run_lstm_l_fp16_reg_t, gemv_lstm.h): which configuration a model runs with on a device, the routine that puts a
// layer's weights into that configuration's layout in global memory (Fp16_reg_layer_geometry, "tile layout"), and the
// size of the buffer its CTAs exchange words through (params.gmem_xchg). The kernels load their weights at every
// launch; in the tile layout a warp's load reads 32 consecutive 4-gate words whatever the tile, where the [input][unit]
// layout of weights_permute (gemv_lstm.h) is read in 8-byte pieces of separate sectors by the tiles with many lanes per
// unit.
//
// The caller passes the problem size; the configuration is chosen here (fp16_reg_kernel_config, kernel_params.h), as
// for the launch, so that callers never handle tile sizes.

#include <cstddef>
#include <cstring>
#include <vector_types.h>  // kernel_traits.h uses the CUDA vector types

#include "../kernel_params.h"
#include "../kernel_traits.h"
#include "gemv_lstm_fp16_reg_geometry.h"

namespace nvidia::quant::lowlat_infer::ssgemv {

// The model sizes of this build that run with the kernel (kernel_params.h): lstm_s, lstm_m and lstm_l.
/// lstm_s's sizes in this build.
inline constexpr Fp16_reg_model_dims FP16_REG_LSTM_S_DIMS{LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS, LSTM_S_NUM_LAYERS};
/// lstm_m's sizes in this build.
inline constexpr Fp16_reg_model_dims FP16_REG_LSTM_M_DIMS{LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS, LSTM_M_NUM_LAYERS};
/// lstm_l's sizes in this build.
inline constexpr Fp16_reg_model_dims FP16_REG_LSTM_L_DIMS{LSTM_L_NUM_INPUTS, LSTM_L_NUM_UNITS, LSTM_L_NUM_LAYERS};

/// Gate q of the kernels' 4-gate words is gate FP16_REG_GATE_ORDER[q] of the model's storage (as weights_permute).
inline constexpr int FP16_REG_GATE_ORDER[4] = {0, 2, 3, 1};

/// The configuration the FP16 register-resident kernel runs a model with (kernel_params.h; the same on every device so
/// far), on the host and in the launch alike; tile.units_per_cta 0 if the model is not one of this build's sizes that
/// run with it.
constexpr Fp16_reg_kernel_config fp16_reg_kernel_config(Fp16_reg_model_dims dims) {
  if (dims == FP16_REG_LSTM_S_DIMS) return FP16_REG_KERNEL_LSTM_S;
  if (dims == FP16_REG_LSTM_M_DIMS) return FP16_REG_KERNEL_LSTM_M;
  if (dims == FP16_REG_LSTM_L_DIMS) return FP16_REG_KERNEL_LSTM_L;
  return {};
}

/// Weights of layer l in the tile layout of `tile`, 4 per 4-gate word (a count of weights, whatever their type).
inline size_t fp16_reg_tile_weight_elems(Fp16_reg_model_dims dims, Fp16_reg_tile_config tile, int l) {
  return size_t(Fp16_reg_geometry{dims, tile}.weight_words(l)) * 4;
}

/// Layer l's weights into the tile layout of `tile`. in: the layer's weights as the model stores them, [gate][unit]
/// [input] with the gates in the model's order (the input of weights_permute); out: fp16_reg_tile_weight_elems(dims,
/// tile, l) weights, zeros where a lane has no weight. TData: the model's data type (FP16 for the kernel).
template <typename TData>
void fp16_reg_tile_permute_weights(
    TData* out, const TData* in, Fp16_reg_model_dims dims, Fp16_reg_tile_config tile, int l) {
  const Fp16_reg_geometry g{dims, tile};
  const Fp16_reg_layer_geometry lg = g.layer(l);
  const int nt = g.block_threads(), cta_words = g.cta_weight_words(l);
  std::memset(out, 0, fp16_reg_tile_weight_elems(dims, tile, l) * sizeof(TData));
  auto put = [&](int cta, int word, int unit, int k) {
    for (int q = 0; q < 4; q++)
      out[(size_t(cta) * cta_words + word) * 4 + q] =
          in[(size_t(FP16_REG_GATE_ORDER[q]) * dims.units + unit) * lg.inputs + k];
  };
  for (int cta = 0; cta < g.ctas(); cta++) {
    for (int t = 0; t < nt; t++) {
      const int unit = cta * tile.units_per_cta + g.unit_of(t), lane = g.lane_of(t);
      if (g.unit_of(t) >= tile.units_per_cta || unit >= dims.units) continue;
      for (int j = 0; j < lg.reg_elems(); j++) {
        const int k = lg.input_of_reg(lane, j);
        if (lg.in_regs(k)) put(cta, lg.reg_weight_word(j, t, nt), unit, k);
      }
      for (int j = 0; j < lg.smem_elems(); j++) {
        const int k = lg.input_of_smem(lane, j);
        if (lg.in_smem(k)) put(cta, lg.smem_weight_word(j, t, nt), unit, k);
      }
    }
  }
}

// The routines a model harness calls, for the configuration this build's kernel runs the model with.
/// Elements of layer l's weights in the tile layout of the model's configuration; 0 if the model is not one of this
/// build's sizes.
inline size_t fp16_reg_weight_elems(Fp16_reg_model_dims dims, int l) {
  const Fp16_reg_tile_config tile = fp16_reg_kernel_config(dims).tile;
  return tile.units_per_cta > 0 ? fp16_reg_tile_weight_elems(dims, tile, l) : 0;
}
/// Layer l's weights into the tile layout of the model's configuration (fp16_reg_tile_permute_weights); false, and
/// nothing written, if the model is not one of this build's sizes.
template <typename TData>
bool fp16_reg_permute_weights(TData* out, const TData* in, Fp16_reg_model_dims dims, int l) {
  const Fp16_reg_tile_config tile = fp16_reg_kernel_config(dims).tile;
  if (tile.units_per_cta == 0) return false;
  fp16_reg_tile_permute_weights(out, in, dims, tile, l);
  return true;
}

/// CTAs the kernel runs the model with, all resident at once (a cooperative launch beyond one); 0 if the model is not
/// one of this build's sizes.
inline int fp16_reg_ctas(Fp16_reg_model_dims dims) {
  const Fp16_reg_kernel_config config = fp16_reg_kernel_config(dims);
  return config.tile.units_per_cta > 0 ? Fp16_reg_exchange_layout{dims, config}.ctas() : 0;
}

/// Words of params.gmem_xchg (unsigned long long): what the CTAs exchange (Fp16_reg_exchange_layout); 0 for a model on
/// one CTA, or not one of this build's sizes. The kernel zeroes them at every launch.
inline size_t fp16_reg_exchange_words(Fp16_reg_model_dims dims) {
  const Fp16_reg_kernel_config config = fp16_reg_kernel_config(dims);
  return config.tile.units_per_cta > 0 ? size_t(Fp16_reg_exchange_layout{dims, config}.words()) : 0;
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv
