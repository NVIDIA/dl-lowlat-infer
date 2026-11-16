/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Geometry of the FP16 register-resident LSTM kernels' compute tile (gemv_lstm_fp16_reg_tile.cuh): where each thread
// sits, which inputs its weights belong to, and where every weight is in global memory. Plain constexpr functions of
// the problem size and the tile configuration, so that the kernels evaluate them at compile time and the host's weight
// routines (gemv_lstm_fp16_reg.h) at run time: both place every weight the same way. Host-only code can
// include this header.

#include <cuda/cmath>
#include <cuda_runtime_api.h>  // __host__ / __device__ for host compilers too

namespace nvidia::quant::lowlat_infer::ssgemv {

/// Problem size.
struct Fp16_reg_model_dims {
  int inputs;  ///< features per step, the inputs of layer 0
  int units;   ///< units per layer, the inputs of every layer above
  int layers;  ///< LSTM layers
  /// Same sizes.
  friend constexpr bool operator==(Fp16_reg_model_dims const&, Fp16_reg_model_dims const&) = default;
};

/// How one CTA computes its part of every layer.
struct Fp16_reg_tile_config {
  int units_per_cta;          ///< units of every layer computed by the CTA
  int lanes_per_unit;         ///< lanes sharing one unit's dot product (a power of two, at most 1024)
  int layer0_inputs_in_smem;  ///< of layer 0's inputs, how many (the last ones) have their weights in shared memory
  int upper_inputs_in_smem;   ///< the same for each layer above
  bool fast_eltwise;          ///< FP16 gate activations with tanh.approx (lstm_eltwise_fast) instead of FP32
};

/// The kernel's configuration of one model size: its tile, and how its CTAs work together.
struct Fp16_reg_kernel_config {
  /// How one CTA computes its part of every layer.
  Fp16_reg_tile_config tile{};
  /// CTAs per copy of the exchanged hidden state and input; 0: one copy for all CTAs.
  int cluster_ctas = 0;
  /// The hidden state through distributed shared memory instead of tagged words in global memory: all CTAs one
  /// hardware cluster (at most 8, the portable size), each CTA's words stored into every CTA's shared memory with
  /// st.async.
  bool dsmem_exchange = false;
};

/// Slice geometry of one layer with `inputs` inputs, the last `inputs_in_smem` of them with weights in shared memory.
/// Each part is split evenly over the lanes of a unit: lane l owns `reg_elems()` consecutive inputs of the register
/// part and `smem_elems()` of the shared-memory part.
struct Fp16_reg_layer_geometry {
  /// The layer's inputs, how many of them (the last ones) have their weights in shared memory, and lanes per unit.
  int inputs, inputs_in_smem, lanes_per_unit;

  /// Inputs with their weights in registers.
  __host__ __device__ constexpr int inputs_in_regs() const { return inputs - inputs_in_smem; }
  /// Register weights of each lane (its slice of the register part).
  __host__ __device__ constexpr int reg_elems() const { return cuda::ceil_div(inputs_in_regs(), lanes_per_unit); }
  /// Shared-memory weights of each lane (its slice of the shared-memory part).
  __host__ __device__ constexpr int smem_elems() const { return cuda::ceil_div(inputs_in_smem, lanes_per_unit); }
  /// Slice strides of the input vector in shared memory: multiples of 8 FP16, so every slice starts on a 16-byte
  /// boundary and loads as uint4 vectors.
  __host__ __device__ constexpr int reg_pad() const { return cuda::round_up(reg_elems(), 8); }
  /// The same for the shared-memory part.
  __host__ __device__ constexpr int smem_pad() const { return cuda::round_up(smem_elems(), 8); }
  /// FP16 elements of the slice-padded input vector.
  __host__ __device__ constexpr int x_elems() const { return lanes_per_unit * (reg_pad() + smem_pad()); }

  /// Input of register weight j / shared-memory weight j of a lane; past the part's end for padding.
  __host__ __device__ constexpr int input_of_reg(int lane, int j) const { return lane * reg_elems() + j; }
  /// The same for shared-memory weight j.
  __host__ __device__ constexpr int input_of_smem(int lane, int j) const {
    return inputs_in_regs() + lane * smem_elems() + j;
  }
  /// Whether input k has its weights in registers.
  __host__ __device__ constexpr bool in_regs(int k) const { return k < inputs_in_regs(); }
  /// Whether input k has its weights in shared memory.
  __host__ __device__ constexpr bool in_smem(int k) const { return k >= inputs_in_regs() && k < inputs; }

  /// Position of input k in the slice-padded input vector: the register part's slices first, then the shared-memory
  /// part's.
  __host__ __device__ constexpr int x_index(int k) const {
    const int r = reg_elems() > 0 ? reg_elems() : 1, s = smem_elems() > 0 ? smem_elems() : 1;
    return in_regs(k)
               ? (k / r) * reg_pad() + k % r
               : lanes_per_unit * reg_pad() + ((k - inputs_in_regs()) / s) * smem_pad() + (k - inputs_in_regs()) % s;
  }

  /// The weight layout in global memory ("tile layout"), per CTA and layer: element j of the register part of every
  /// thread of the CTA, j = 0 .. reg_elems() - 1, then the same for the shared-memory part; one 4-gate word each, zero
  /// for lanes past the units and for inputs past the layer's. A warp's load of one element reads 32 consecutive words.
  __host__ __device__ constexpr int cta_weight_words(int block_threads) const {
    return (reg_elems() + smem_elems()) * block_threads;
  }
  /// Word of register weight j of thread `thread` in the CTA's part.
  __host__ __device__ constexpr int reg_weight_word(int j, int thread, int block_threads) const {
    return j * block_threads + thread;
  }
  /// Word of shared-memory weight j of thread `thread` in the CTA's part.
  __host__ __device__ constexpr int smem_weight_word(int j, int thread, int block_threads) const {
    return (reg_elems() + j) * block_threads + thread;
  }
};

/// The tile of one problem size and configuration: the CTA's threads, and where each one sits.
struct Fp16_reg_geometry {
  Fp16_reg_model_dims dims;   ///< the problem size
  Fp16_reg_tile_config tile;  ///< the tile

  /// Threads per warp.
  static constexpr int WARP_SIZE = 32;

  /// Slice geometry of layer l (layer 0: the model's inputs; above: its units).
  __host__ __device__ constexpr Fp16_reg_layer_geometry layer(int l) const {
    return l == 0 ? Fp16_reg_layer_geometry{dims.inputs, tile.layer0_inputs_in_smem, tile.lanes_per_unit}
                  : Fp16_reg_layer_geometry{dims.units, tile.upper_inputs_in_smem, tile.lanes_per_unit};
  }
  /// lanes_per_unit <= 32: several units per warp. lanes_per_unit > 32: one unit spans warps_per_unit() warps.
  __host__ __device__ constexpr int units_per_warp() const {
    return tile.lanes_per_unit <= WARP_SIZE ? WARP_SIZE / tile.lanes_per_unit : 1;
  }
  /// Warps per unit (1 for lanes_per_unit <= 32).
  __host__ __device__ constexpr int warps_per_unit() const {
    return tile.lanes_per_unit <= WARP_SIZE ? 1 : tile.lanes_per_unit / WARP_SIZE;
  }
  /// Warps of the CTA.
  __host__ __device__ constexpr int num_warps() const {
    return tile.lanes_per_unit <= WARP_SIZE ? cuda::ceil_div(tile.units_per_cta, units_per_warp())
                                            : tile.units_per_cta * warps_per_unit();
  }
  /// Threads of the CTA.
  __host__ __device__ constexpr int block_threads() const { return num_warps() * WARP_SIZE; }
  /// CTAs over the model's units.
  __host__ __device__ constexpr int ctas() const { return cuda::ceil_div(dims.units, tile.units_per_cta); }

  /// Unit of the tile, and lane of that unit (its slice of every layer's inputs), of thread t of the CTA.
  __host__ __device__ constexpr int unit_of(int t) const {
    return tile.lanes_per_unit <= WARP_SIZE ? t / WARP_SIZE * units_per_warp() + t % WARP_SIZE / tile.lanes_per_unit
                                            : t / WARP_SIZE / warps_per_unit();
  }
  /// Lane of thread t's unit.
  __host__ __device__ constexpr int lane_of(int t) const {
    return tile.lanes_per_unit <= WARP_SIZE ? t % WARP_SIZE % tile.lanes_per_unit
                                            : t / WARP_SIZE % warps_per_unit() * WARP_SIZE + t % WARP_SIZE;
  }

  /// 4-gate words of one CTA's part of layer l in the tile layout, and of the whole layer.
  __host__ __device__ constexpr int cta_weight_words(int l) const { return layer(l).cta_weight_words(block_threads()); }
  /// 4-gate words of the whole layer l in the tile layout.
  __host__ __device__ constexpr long long weight_words(int l) const {
    return static_cast<long long>(ctas()) * cta_weight_words(l);
  }
};

/// The tagged 8-byte words (gemv_lstm_tagged_word.cuh) the CTAs of a multi-CTA kernel exchange in global memory, in
/// one buffer: the hidden state of a layer, two values per word, [cluster][parity of the exchange][cta][word of the
/// CTA] (a CTA's units rounded up to whole words: any units per CTA, the last CTA possibly partial; none with
/// dsmem_exchange); each CTA's part of the output, [parity][cta]; the go word, CTA 0's signal to the others; each CTA's
/// acknowledgement that it has loaded its states (resident with two host signals per step), [cta]; the input, two
/// values per word, when CTA 0 reads it from host memory for all, [cluster][word]. None for one CTA.
struct Fp16_reg_exchange_layout {
  Fp16_reg_model_dims dims;       ///< the problem size
  Fp16_reg_kernel_config config;  ///< the kernel's configuration

  /// CTAs of the kernel.
  __host__ __device__ constexpr int ctas() const { return Fp16_reg_geometry{dims, config.tile}.ctas(); }
  /// CTAs per copy of the hidden state and input (all CTAs with cluster_ctas 0).
  __host__ __device__ constexpr int cluster_ctas() const {
    return config.cluster_ctas > 0 ? config.cluster_ctas : ctas();
  }
  /// Copies of the hidden state and input.
  __host__ __device__ constexpr int clusters() const { return cuda::ceil_div(ctas(), cluster_ctas()); }
  /// Words of one CTA's hidden values (two per word).
  __host__ __device__ constexpr int h_words_per_cta() const { return cuda::ceil_div(config.tile.units_per_cta, 2); }
  /// Words of a whole layer's hidden values.
  __host__ __device__ constexpr int h_words() const { return ctas() * h_words_per_cta(); }
  /// Words of the input (two values per word).
  __host__ __device__ constexpr int input_words() const { return cuda::ceil_div(dims.inputs, 2); }

  /// Offsets in the buffer, in words, of its parts (above), and its size; no words for one CTA.
  __host__ __device__ constexpr int h_offset() const { return 0; }
  /// The output parts: after the hidden state, which takes none with dsmem_exchange.
  __host__ __device__ constexpr int dot_offset() const {
    return h_offset() + (config.dsmem_exchange ? 0 : clusters() * 2 * h_words());
  }
  /// The go word.
  __host__ __device__ constexpr int go_offset() const { return dot_offset() + 2 * ctas(); }
  /// Each CTA's acknowledgement that it has loaded its states.
  __host__ __device__ constexpr int ack_offset() const { return go_offset() + 1; }
  /// The input, when CTA 0 passes it on.
  __host__ __device__ constexpr int input_offset() const { return ack_offset() + ctas(); }
  /// Words of the buffer (params.gmem_xchg).
  __host__ __device__ constexpr int words() const {
    return ctas() == 1 ? 0 : input_offset() + clusters() * input_words();
  }
};

}  // namespace nvidia::quant::lowlat_infer::ssgemv
