/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Compute tile of the FP16 register-resident LSTM kernels (README.md).
//
// A tile is what one CTA computes of every layer: units_per_cta units. Each unit's four gates are computed by
// lanes_per_unit lanes, each owning one slice of the layer's inputs (a K-split inside the CTA); the slices' partial
// sums are combined with XOR shuffles, and through shared memory when a unit spans several warps.
//
// Weights stay on chip: in registers, except for the last inputs_in_smem inputs of a layer, whose weights are in
// shared memory. Both parts are split evenly over all lanes of a unit, because register allocation is the same for
// every thread: only a shorter register slice in every lane saves registers.
//
// The knobs come in two structs (gemv_lstm_fp16_reg_geometry.h): Fp16_reg_model_dims is the problem size, which
// becomes the real kernels' template parameters; Fp16_reg_tile_config is how one CTA computes its part. The traits
// below are the geometry's functions evaluated at compile time.
//
// The weights come from global memory in the tile layout (Fp16_reg_layer_geometry), which the host prepares with
// gemv_lstm_fp16_reg.h.

#include <cooperative_groups.h>
#include <cuda/cmath>
#include <stdint.h>

#include "../gemv_lstm.h"
#include "../gemv_lstm_utils.cuh"
#include "../kernel_params.h"
#include "gemv_lstm_fp16_reg_eltwise.cuh"
#include "gemv_lstm_fp16_reg_geometry.h"

namespace nvidia::quant::lowlat_infer::ssgemv {

/// Slice geometry of one layer (Fp16_reg_layer_geometry) at compile time.
template <int INPUTS_, int INPUTS_IN_SMEM_, int LANES_PER_UNIT_>
struct Fp16_reg_layer_traits {
  static_assert(0 <= INPUTS_IN_SMEM_ && INPUTS_IN_SMEM_ <= INPUTS_, "inputs_in_smem must be in [0, inputs]");
  /// The layer's slice geometry.
  static __host__ __device__ constexpr Fp16_reg_layer_geometry geometry() {
    return {INPUTS_, INPUTS_IN_SMEM_, LANES_PER_UNIT_};
  }
  /// The layer's inputs, those with their weights in shared memory, and those in registers.
  enum { INPUTS = INPUTS_, INPUTS_IN_SMEM = INPUTS_IN_SMEM_, INPUTS_IN_REGS = geometry().inputs_in_regs() };
  /// Lanes per unit.
  enum { LANES_PER_UNIT = LANES_PER_UNIT_ };
  enum { REG_ELEMS = geometry().reg_elems() };            ///< register weights per lane
  enum { SMEM_ELEMS = geometry().smem_elems() };          ///< shared-memory weights per lane
  enum { REG_ARRAY = REG_ELEMS > 0 ? REG_ELEMS : 1 };     ///< C++ arrays cannot be empty
  enum { SMEM_ARRAY = SMEM_ELEMS > 0 ? SMEM_ELEMS : 1 };  ///< the same, as a divisor
  /// Slice strides of the input vector, and its FP16 elements (Fp16_reg_layer_geometry).
  enum { REG_PAD = geometry().reg_pad(), SMEM_PAD = geometry().smem_pad(), X_ELEMS = geometry().x_elems() };

  /// Input of register weight j of a lane (Fp16_reg_layer_geometry::input_of_reg).
  static __host__ __device__ constexpr int input_of_reg(int lane, int j) { return geometry().input_of_reg(lane, j); }
  /// Input of shared-memory weight j of a lane.
  static __host__ __device__ constexpr int input_of_smem(int lane, int j) { return geometry().input_of_smem(lane, j); }
  /// Whether input k has its weights in registers.
  static __host__ __device__ constexpr bool in_regs(int k) { return geometry().in_regs(k); }
  /// Whether input k has its weights in shared memory.
  static __host__ __device__ constexpr bool in_smem(int k) { return geometry().in_smem(k); }
  /// Position of input k in the slice-padded input vector.
  static __host__ __device__ constexpr int x_index(int k) { return geometry().x_index(k); }
};

/// Everything derived from the problem size and the tile configuration.
template <Fp16_reg_model_dims MODEL, Fp16_reg_tile_config TILE>
struct Fp16_reg_tile_traits {
  static constexpr Fp16_reg_model_dims model = MODEL;  ///< the problem size
  static constexpr Fp16_reg_tile_config tile = TILE;   ///< the tile configuration
  /// Units per CTA, layers, and lanes per unit.
  enum { UNITS_PER_CTA = TILE.units_per_cta, LAYERS = MODEL.layers, LANES_PER_UNIT = TILE.lanes_per_unit };
  /// FP16 gate activations (lstm_eltwise_fast).
  static constexpr bool FAST_ELTWISE = TILE.fast_eltwise;
  /// Data type of the weights, the inputs and the accumulation. FP16 only. BF16 would need the bfloat16 HFMA2 in
  /// fp16_reg_fma4, fast_eltwise off, and likely FP32 accumulation over long inputs; FP32 would need 16-byte weights
  /// (twice the registers), FFMA, 32-bit inputs in the input layout, float4 shuffles, and one hidden value per tagged
  /// word in the exchange.
  static constexpr Data_kind DATA_KIND = DATA_TYPE_FP16;
  static_assert(DATA_KIND == DATA_TYPE_FP16, "FP16 only: see above for what BF16 and FP32 need");
  static_assert(!FAST_ELTWISE || DATA_KIND == DATA_TYPE_FP16, "fast_eltwise computes the gates in FP16");
  static_assert(LAYERS >= 2, "at least two layers");
  static_assert(UNITS_PER_CTA >= 1 && UNITS_PER_CTA <= MODEL.units, "units_per_cta must be in [1, units]");
  static_assert(
      LANES_PER_UNIT >= 1 && (LANES_PER_UNIT & (LANES_PER_UNIT - 1)) == 0 && LANES_PER_UNIT <= 1024,
      "lanes_per_unit must be a power of two, at most 1024");

  /// The tile's geometry.
  static __host__ __device__ constexpr Fp16_reg_geometry geometry() { return {MODEL, TILE}; }
  /// Threads per warp.
  enum { WARP_SIZE = Fp16_reg_geometry::WARP_SIZE };
  /// lanes_per_unit <= 32: several units per warp. lanes_per_unit > 32: one unit spans WARPS_PER_UNIT warps.
  enum { UNITS_PER_WARP = geometry().units_per_warp(), WARPS_PER_UNIT = geometry().warps_per_unit() };
  /// Warps and threads of the CTA.
  enum { NUM_WARPS = geometry().num_warps(), BLOCK_THREADS = geometry().block_threads() };
  static_assert(BLOCK_THREADS <= 1024, "too many threads per CTA");
  /// Whether every thread computes a unit; otherwise the last warp has lanes past units_per_cta.
  enum { ALL_UNITS_VALID = LANES_PER_UNIT > WARP_SIZE || UNITS_PER_CTA % UNITS_PER_WARP == 0 };

  /// Layer 0's slice geometry (the model's inputs).
  using LAYER0_TRAITS = Fp16_reg_layer_traits<MODEL.inputs, TILE.layer0_inputs_in_smem, TILE.lanes_per_unit>;
  /// The slice geometry of each layer above (the layer below's units).
  using LAYERS_ABOVE_TRAITS = Fp16_reg_layer_traits<MODEL.units, TILE.upper_inputs_in_smem, TILE.lanes_per_unit>;

  /// Shared-memory weights, 4 gates (uint2) per element, per layer laid out [element j][unit * lanes_per_unit + lane]
  /// so that a warp reads consecutive 8-byte words.
  enum { SMEM_W0 = LAYER0_TRAITS::SMEM_ELEMS * UNITS_PER_CTA * LANES_PER_UNIT };
  enum { SMEM_WU = LAYERS_ABOVE_TRAITS::SMEM_ELEMS * UNITS_PER_CTA * LANES_PER_UNIT };
  /// Rounded up to 16 bytes: the input vectors that follow the weights are read as uint4.
  enum { SMEM_W_BYTES = cuda::round_up((SMEM_W0 + (LAYERS - 1) * SMEM_WU) * 8, 16) };
  /// Offset (uint2 elements) of layer l's shared-memory weights.
  static __host__ __device__ constexpr int smem_w_offset(int l) { return l == 0 ? 0 : SMEM_W0 + (l - 1) * SMEM_WU; }
  /// Weight registers per thread, over all layers.
  enum { WEIGHT_REGS = 2 * (LAYER0_TRAITS::REG_ELEMS + (LAYERS - 1) * LAYERS_ABOVE_TRAITS::REG_ELEMS) };
};

/// Where a thread sits in the tile.
template <typename TILE_TRAITS>
struct Fp16_reg_lane {
  int unit;     ///< unit of the tile
  int lane;     ///< lane of the unit, 0 .. lanes_per_unit - 1: its slice of every layer's inputs
  bool valid;   ///< unit < units_per_cta, and a unit of the model (the kernel clears it past the last one)
  bool leader;  ///< lane 0: holds the unit's complete sum after the reduction

  /// This thread's place in the tile (from threadIdx.x).
  __device__ __forceinline__ Fp16_reg_lane() {
    unit = TILE_TRAITS::geometry().unit_of(threadIdx.x);
    lane = TILE_TRAITS::geometry().lane_of(threadIdx.x);
    valid = unit < TILE_TRAITS::UNITS_PER_CTA;
    leader = lane == 0;
  }

  /// Index of this lane's shared-memory weight j within its layer's part.
  __device__ __forceinline__ int smem_w_index(int j) const {
    return (j * TILE_TRAITS::UNITS_PER_CTA + unit) * TILE_TRAITS::LANES_PER_UNIT + lane;
  }
};

/// Loads this thread's weights of one layer: the register part into w_reg, the shared-memory part into s_w (the
/// layer's part of the shared-memory weights). g_w is this CTA's part of the layer in the tile layout
/// (Fp16_reg_layer_geometry), which holds zeros where a lane has no weight.
template <typename TILE_TRAITS, typename LAYER_TRAITS>
static __device__ __forceinline__ void fp16_reg_load_weights(
    const uint2* g_w, const Fp16_reg_lane<TILE_TRAITS>& ln, uint2 (&w_reg)[LAYER_TRAITS::REG_ARRAY], uint2* s_w) {
#pragma unroll
  for (int j = 0; j < LAYER_TRAITS::REG_ELEMS; j++)
    w_reg[j] = g_w[LAYER_TRAITS::geometry().reg_weight_word(j, threadIdx.x, TILE_TRAITS::BLOCK_THREADS)];
  if (ln.valid) {
#pragma unroll
    for (int j = 0; j < LAYER_TRAITS::SMEM_ELEMS; j++)
      s_w[ln.smem_w_index(j)] =
          g_w[LAYER_TRAITS::geometry().smem_weight_word(j, threadIdx.x, TILE_TRAITS::BLOCK_THREADS)];
  }
}

/// The 32-bit word holding FP16 element k of a run held as uint4 vectors; k is a compile-time constant after
/// unrolling, so this folds to a register select.
template <int N>
static __device__ __forceinline__ uint32_t fp16_reg_word(const uint4 (&v)[N], int k) {
  const uint4 q = v[k / 8];
  const int r = (k % 8) / 2;
  return (r == 0) ? q.x : (r == 1) ? q.y : (r == 2) ? q.z : q.w;
}

/// acc += w * x for 4 gates, x being the low (k even) or high (k odd) FP16 of x_word. The intrinsics let the compiler
/// use HFMA2's operand selector (.H0_H0 / .H1_H1) for the broadcast, where unpacking x to 16 bits and repacking it
/// costs two integer instructions per element; fma.rn.f16x2 either way, so results are unchanged.
static __device__ __forceinline__ uint2 fp16_reg_fma4(uint2 w, uint32_t x_word, int k, uint2 acc) {
  const __half2 x2 = *reinterpret_cast<const __half2*>(&x_word);
  const __half2 xb = (k % 2) ? __high2half2(x2) : __low2half2(x2);
  const __half2 lo = __hfma2(*reinterpret_cast<const __half2*>(&w.x), xb, *reinterpret_cast<const __half2*>(&acc.x));
  const __half2 hi = __hfma2(*reinterpret_cast<const __half2*>(&w.y), xb, *reinterpret_cast<const __half2*>(&acc.y));
  return make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
}

/// The 4-gate word of the lane `lane_mask` away (__shfl_xor_sync over the whole warp).
static __device__ __forceinline__ uint2 fp16_reg_shfl_xor(uint2 v, int lane_mask) {
  auto r = __shfl_xor_sync(0xffffffffu, *reinterpret_cast<unsigned long long*>(&v), lane_mask);
  return *reinterpret_cast<uint2*>(&r);
}

/// One layer's pre-activation of this thread's unit (4 gates): this lane's slice of W * x, reduced over the warp's
/// lanes of the unit. With lanes_per_unit <= 32 every lane of the unit returns the complete sum; with more, lane 0 of
/// each warp returns that warp's partial, which fp16_reg_combine_warps adds up.
template <typename TILE_TRAITS, typename LAYER_TRAITS>
static __device__ __forceinline__ uint2 fp16_reg_layer_dot(
    const uint2 (&w_reg)[LAYER_TRAITS::REG_ARRAY],
    const uint2* s_w,
    const uint16_t* s_x,
    const Fp16_reg_lane<TILE_TRAITS>& ln) {
  constexpr Data_kind DK = TILE_TRAITS::DATA_KIND;
  const uint2 zero4 = make_uint2(0u, 0u);
  uint2 a0 = zero4, a1 = zero4;  // two accumulators halve the dependent HFMA2 chain
  if constexpr (LAYER_TRAITS::REG_ELEMS > 0) {
    uint4 xr[LAYER_TRAITS::REG_PAD / 8];
    const uint4* xv = reinterpret_cast<const uint4*>(s_x + ln.lane * LAYER_TRAITS::REG_PAD);
#pragma unroll
    for (int j = 0; j < LAYER_TRAITS::REG_PAD / 8; j++) xr[j] = xv[j];
#pragma unroll
    for (int j = 0; j < LAYER_TRAITS::REG_ELEMS; j++) {
      if (j % 2 == 0)
        a0 = fp16_reg_fma4(w_reg[j], fp16_reg_word(xr, j), j, a0);
      else
        a1 = fp16_reg_fma4(w_reg[j], fp16_reg_word(xr, j), j, a1);
    }
  }
  if constexpr (LAYER_TRAITS::SMEM_ELEMS > 0) {
    if (TILE_TRAITS::ALL_UNITS_VALID || ln.valid) {  // lanes past units_per_cta have no shared-memory weights
      uint4 xm[LAYER_TRAITS::SMEM_PAD / 8];
      const uint4* xv = reinterpret_cast<const uint4*>(
          s_x + LAYER_TRAITS::LANES_PER_UNIT * LAYER_TRAITS::REG_PAD + ln.lane * LAYER_TRAITS::SMEM_PAD);
#pragma unroll
      for (int j = 0; j < LAYER_TRAITS::SMEM_PAD / 8; j++) xm[j] = xv[j];
#pragma unroll
      for (int j = 0; j < LAYER_TRAITS::SMEM_ELEMS; j++) {
        const uint2 w = s_w[ln.smem_w_index(j)];
        if ((LAYER_TRAITS::REG_ELEMS + j) % 2 == 0)
          a0 = fp16_reg_fma4(w, fp16_reg_word(xm, j), j, a0);
        else
          a1 = fp16_reg_fma4(w, fp16_reg_word(xm, j), j, a1);
      }
    }
  }
  uint2 acc = add4<DK>(a0, a1);
#pragma unroll
  for (int m = 1; m < (TILE_TRAITS::LANES_PER_UNIT < TILE_TRAITS::WARP_SIZE ? TILE_TRAITS::LANES_PER_UNIT
                                                                            : TILE_TRAITS::WARP_SIZE);
       m <<= 1) {
    acc = add4<DK>(acc, fp16_reg_shfl_xor(acc, m));
  }
  return acc;
}

/// lanes_per_unit > 32 only: adds the warps' partials of each unit through shared memory (s_part: units_per_cta *
/// WARPS_PER_UNIT uint2). Returns the complete sum in the unit's leader; costs one block barrier.
template <typename TILE_TRAITS>
static __device__ __forceinline__ uint2
fp16_reg_combine_warps(uint2 acc, uint2* s_part, const Fp16_reg_lane<TILE_TRAITS>& ln) {
  static_assert(TILE_TRAITS::LANES_PER_UNIT > TILE_TRAITS::WARP_SIZE);
  const int lane_id = threadIdx.x % TILE_TRAITS::WARP_SIZE;
  if (lane_id == 0 && ln.valid) s_part[ln.unit * TILE_TRAITS::WARPS_PER_UNIT + ln.lane / TILE_TRAITS::WARP_SIZE] = acc;
  __syncthreads();
  if (ln.leader && ln.valid) {
#pragma unroll
    for (int w = 1; w < TILE_TRAITS::WARPS_PER_UNIT; w++)
      acc = add4<TILE_TRAITS::DATA_KIND>(acc, s_part[ln.unit * TILE_TRAITS::WARPS_PER_UNIT + w]);
  }
  return acc;
}

/// Gate activations: acc (4 gates, bias and recurrent part included) and the cell state in -> hidden and cell out.
template <typename TILE_TRAITS>
static __device__ __forceinline__ void fp16_reg_eltwise(float& hx, float& cx, uint2 acc, uint16_t c_in) {
  if constexpr (TILE_TRAITS::FAST_ELTWISE)
    lstm_eltwise_fast<TILE_TRAITS::DATA_KIND>(hx, cx, acc, c_in);
  else
    lstm_eltwise<TILE_TRAITS::DATA_KIND>(hx, cx, acc, c_in);
}

/// One thread's part of a tile: its weights in registers (the CTA's shared-memory weights next to them) and its unit's
/// biases.
template <typename TILE_TRAITS>
struct Fp16_reg_tile {
  using LAYER0_TRAITS = typename TILE_TRAITS::LAYER0_TRAITS;              ///< layer 0's slice geometry
  using LAYERS_ABOVE_TRAITS = typename TILE_TRAITS::LAYERS_ABOVE_TRAITS;  ///< the layers above's
  Fp16_reg_lane<TILE_TRAITS> ln;                                          ///< this thread's place in the tile
  uint2 w0[LAYER0_TRAITS::REG_ARRAY];                                     ///< layer 0's register weights
  uint2 wu[TILE_TRAITS::LAYERS - 1][LAYERS_ABOVE_TRAITS::REG_ARRAY];      ///< the register weights of the layers above
  uint2 bias[TILE_TRAITS::LAYERS];                                        ///< every layer's 4 gate biases of the unit

  /// Loads the weights and biases, once. g_w[l]: the CTA's part of layer l's weights in the tile layout; g_bias[l]: the
  /// biases of the CTA's units; s_w: the CTA's shared-memory weights.
  __device__ __forceinline__ void load(const uint2* const* g_w, const uint2* const* g_bias, uint2* s_w) {
    fp16_reg_load_weights<TILE_TRAITS, LAYER0_TRAITS>(g_w[0], ln, w0, s_w);
#pragma unroll
    for (int l = 1; l < TILE_TRAITS::LAYERS; l++)
      fp16_reg_load_weights<TILE_TRAITS, LAYERS_ABOVE_TRAITS>(
          g_w[l], ln, wu[l - 1], s_w + TILE_TRAITS::smem_w_offset(l));
#pragma unroll
    for (int l = 0; l < TILE_TRAITS::LAYERS; l++) bias[l] = ln.valid ? g_bias[l][ln.unit] : make_uint2(0u, 0u);
  }

  /// Layer l (a compile-time constant after unrolling): the unit's pre-activation from the layer's slice-padded input
  /// x, plus bias and recurrent part, then the gate activations. hx and cx are valid in the unit's leader. s_part: the
  /// cross-warp step's shared memory (lanes_per_unit > 32); the caller puts a block barrier between two layers, which
  /// also keeps a layer's s_part writes from overtaking the previous layer's reads. Returns the unit's W * x (valid in
  /// the leader).
  __device__ __forceinline__ uint2
  layer(int l, const uint16_t* x, const uint2* s_w, uint2* s_part, uint2 gemm_h, uint16_t c_in, float& hx, float& cx)
      const {
    uint2 acc = l == 0 ? fp16_reg_layer_dot<TILE_TRAITS, LAYER0_TRAITS>(w0, s_w, x, ln)
                       : fp16_reg_layer_dot<TILE_TRAITS, LAYERS_ABOVE_TRAITS>(
                             wu[l - 1], s_w + TILE_TRAITS::smem_w_offset(l), x, ln);
    if constexpr (TILE_TRAITS::LANES_PER_UNIT > TILE_TRAITS::WARP_SIZE)
      acc = fp16_reg_combine_warps<TILE_TRAITS>(acc, s_part, ln);
    const uint2 dot = acc;
    acc = add4<TILE_TRAITS::DATA_KIND>(acc, bias[l]);
    acc = add4<TILE_TRAITS::DATA_KIND>(acc, gemm_h);
    fp16_reg_eltwise<TILE_TRAITS>(hx, cx, acc, c_in);
    return dot;
  }

  /// The unit's term w * hx of an output dot product (from its leader), added up over the warp's units: the warp's sum
  /// in lane 0.
  __device__ __forceinline__ float warp_dot(float w, float hx) const {
    float dot = (ln.leader && ln.valid) ? w * hx : 0.0f;
    if constexpr (TILE_TRAITS::LANES_PER_UNIT < TILE_TRAITS::WARP_SIZE) {
#pragma unroll
      for (int off = TILE_TRAITS::WARP_SIZE / 2; off >= TILE_TRAITS::LANES_PER_UNIT; off /= 2)
        dot += __shfl_xor_sync(0xffffffffu, dot, off);
    }
    return dot;
  }
};

}  // namespace nvidia::quant::lowlat_infer::ssgemv
