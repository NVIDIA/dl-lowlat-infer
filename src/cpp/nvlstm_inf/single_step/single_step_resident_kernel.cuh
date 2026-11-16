/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The resident SingleStep precompute's kernel (README.md, single_step_resident.cuh): the SingleStep precompute fused
// into one kernel (TMA loads with an mbarrier pipeline, the layer-0 input term as one GEMV, the recurrent term fused)
// and made resident:
//  * per pass, every CTA waits for the host's go words and copies the input into its shared memory;
//  * the CTAs order their phases with per-CTA epoch flags, not grid barriers (each waits only for the data it reads);
//  * the window states are double buffered by pass (the previous pass's h and c are read, this pass's written), with
//    one row more than the windows: window s of the previous pass becomes row s + 1, row 0 stays zero as the window
//    that starts with this input (as in ModelSingleStep: output row offset 1);
//  * a pass that publishes computes, per layer, R * h and takes c of row num_steps - 1 into state slot (arg - 1) % 2,
//    then (CTA 0, once every CTA's part is written) writes arg to the ready word; every pass ends with the pass
//    number in the host's done word.

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../LstmConstants.hpp"
#include "../ModelDef.h"
#include "../cuda_error_util.h"
#include "../cuda_helpers.cuh"
#include "single_step_resident.cuh"
#include "single_step_resident_tile.cuh"

namespace nvidia::quant::lowlat_infer::single_step {

namespace {

namespace cg = cooperative_groups;

using namespace resident_tile;

constexpr int MAX_LAYERS = 8;
constexpr int MAX_KPAD_INPUTS = 1024;  // the input, copied into every CTA's shared memory
constexpr unsigned GO_EXIT = 0xffffffffu;
// In the go's argument: the state buffers were cleared after the critical warp prefetched from them (Init while the
// kernel runs), so it prefetches again after the go. The ready values stay far below it.
constexpr unsigned GO_RELOAD = 1u << 31;

// The tile (the prototype's selector's choice at an lstm_s size): 8 units per CTA, 256 threads, K in chunks of 64, a
// 2-stage pipeline (3 gained nothing and took the shared memory that resident weight slabs need), no K split across
// warps.
constexpr int UNITS_PER_CTA = 8, BLOCK_THREADS = 256, K_CHUNK = 64, NUM_STAGES = 2, K_GROUPS = 1;
// Warp roles: the BLOCK_THREADS bulk threads (warps 0..) advance every window; one more warp per CTA, the critical
// warp, computes only the oldest window's step and the consumer's slot, right after the go.
constexpr int CTA_THREADS = BLOCK_THREADS + WARP_SIZE;
constexpr int MAX_KPAD_UNITS = 1024;  // the critical warp's vectors in shared memory

// The CTAs order their phases with epoch flags instead of grid barriers: per CTA and sync point a word holding the last
// pass that CTA finished the point in (pass numbers only grow, across launches too, so the words are never cleared).
// A producer sets its flag after the CTA's stores (__syncthreads, then thread 0: fence, release store); a consumer
// waits only for the flags of the data it reads. Sync points: the layers' h (one per layer), the slot's part.
enum { RP_FLAG_SLOT = MAX_LAYERS, RP_FLAG_POINTS };
#ifndef RP_SPIN_NS
#define RP_SPIN_NS 32  // ns between polls of the epoch flags
#endif

// One warp: waits until every CTA's flag in flags[0, ctas) shows at least `pass`. The spin uses relaxed loads with a
// short back-off (spinning acquire loads of every waiting CTA on the flags' lines slowed the producers down), one
// acquire fence once all are seen.
__device__ __forceinline__ void rp_wait_flags(unsigned* flags, int ctas, unsigned pass) {
  for (int i = threadIdx.x % WARP_SIZE; i < ctas; i += WARP_SIZE) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> flag(flags[i]);
    while (static_cast<int>(flag.load(cuda::memory_order_relaxed) - pass) < 0) __nanosleep(RP_SPIN_NS);
  }
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
  __syncwarp();
}
// One thread, after a barrier that follows its role's stores: this CTA is through the point in `pass`.
__device__ __forceinline__ void rp_set_flag(unsigned* flag, unsigned pass) {
  __threadfence();
  cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*flag).store(pass, cuda::memory_order_release);
}

struct Rp_params {
  int num_layers, num_units, num_inputs, num_steps;
  int num_n_tiles, step_splits, unit_ctas;
  int kpad_inputs, kpad_units;
  int act_rows;         // rows per layer of the state buffers: num_steps + 1
  unsigned first_pass;  // the pass number of the first go after this launch (the passes continue across launches)
  // Weights in the per-CTA slabs of the tile ([cta][4 * UNITS_PER_CTA][kpad], (i,f) / (g,o) row pairs).
  half const* gmem_weights_x0;             // layer 0's input weights (the GEMV)
  half const* gmem_weights_h[MAX_LAYERS];  // recurrent weights (the consumer's R * h)
  half const* gmem_weights_xr;             // layers 1..: input weights, [layer - 1][cta][4 * UNITS_PER_CTA][kpad_units]
  // Weights resident in shared memory, loaded once per launch (bit l: layer l's R, layer l's input weights from layer
  // 1 on), after the stage buffers at byte resident_offset of the aligned dynamic shared memory: R of the resident
  // layers in order, then the input weights.
  unsigned resident_wh, resident_wxr;
  bool resident_x0;  // layer 0's input slab too (for the critical warp), first, in the same layout (kpad_inputs)
  int resident_offset;
  int crit_prev_offset;  // the critical warp's previous h of every layer (num_layers * kpad_units halves)
  half const* gmem_biases[MAX_LAYERS];  // [4 * num_units], gate-major
  half* gmem_h[2];                      // [layer][act_rows][kpad_units], by pass parity
  half* gmem_c[2];
  half const* host_input;       // [num_inputs], pinned host memory
  unsigned const* host_go_seq;  // pinned host memory: the pass number, or GO_EXIT
  unsigned const* host_go_arg;  // pinned host memory: 0, or the ready value this pass publishes
  unsigned* host_done;          // pinned host memory: the last pass done
  unsigned* host_published;     // pinned host memory: the last pass whose slot is published (after the ready word)
  // [layer][kpad_units / 2]: the critical warps' h of the oldest window as tagged words, two units' FP16 h and the pass
  // number in one 8-byte store, so a reader that sees its pass holds the data (no flag, no fence).
  unsigned long long* gmem_crit_tagged;
  unsigned* sync_flags;       // [RP_FLAG_POINTS][CTAs]: the epoch flags
  bool go_relay;              // the go words and the input in pinned host memory: CTA 0 passes them on
  unsigned* gmem_go;          // go_relay: [2], CTA 0's go (pass number) and its arg, to the other CTAs
  half* gmem_x;               // go_relay: [kpad_inputs], the input, to the other CTAs
  half* gmem_slot_gemm_h[2];  // [layer][4 * num_units]
  half* gmem_slot_c[2];       // [layer][num_units]
  unsigned* ready;
  CUtensorMap map_wh;    // rows [layer][cta][4 * UNITS_PER_CTA]
  CUtensorMap map_wxr;   // layers 1..: rows [layer - 1][cta][4 * UNITS_PER_CTA]
  CUtensorMap map_h[2];  // rows [layer][act_rows]
};

// The bulk threads' barrier (named barrier 1), so that the critical warp is not held by it.
__device__ __forceinline__ void bulk_sync() { asm volatile("bar.sync 1, %0;\n" ::"n"(BLOCK_THREADS) : "memory"); }

__launch_bounds__(CTA_THREADS, 1) __global__
    void single_step_resident_kernel(const __grid_constant__ Rp_params params) {
  static_assert(K_GROUPS == 1, "no K-group reduction in this kernel");
  using Tiling = Warp_tiling<UNITS_PER_CTA, BLOCK_THREADS, K_GROUPS>;
  enum { NUM_WARPS = Tiling::NUM_WARPS };
  enum { N_GROUPS = Tiling::N_GROUPS };
  enum { TILE_ROWS = UNITS_PER_CTA * 4 };
  enum { NUM_M_TILES = Tiling::NUM_M_TILES };
  enum { NUM_TILE_PAIRS = UNITS_PER_CTA / 8 };
  enum { N_TILES_PER_WARP = Tiling::N_TILES_PER_WARP };
  static_assert(TILE_ROWS == WARP_SIZE, "the critical warp takes one slab row per lane");

  extern __shared__ __align__(1024) char smem_storage[];
  int const lane_id = threadIdx.x % WARP_SIZE;
  int const warp_id = threadIdx.x / WARP_SIZE;
  int const n_group = warp_id / K_GROUPS;
  int const cta_id = blockIdx.x;
  bool const cta0 = blockIdx.x == 0 && blockIdx.y == 0;
  uint32_t const smem_base_offset = static_cast<uint32_t>(__cvta_generic_to_shared(smem_storage));
  char* smem_base = smem_storage + ((1024 - (smem_base_offset & 1023)) & 1023);
  char* smem_a = smem_base;
  int const n_tiles_box_rows = cuda::ceil_div(params.num_n_tiles, params.step_splits);
  char* smem_b = smem_base + NUM_STAGES * TILE_ROWS * K_CHUNK * 2;
  __shared__ __align__(8) uint64_t full_bar[NUM_STAGES];
  __shared__ __align__(8) uint64_t empty_bar[NUM_STAGES];
  uint32_t chunk_counter = 0;
  uint32_t const full_bar_s = opaque<uint32_t>(static_cast<uint32_t>(__cvta_generic_to_shared(full_bar)));
  uint32_t const empty_bar_s = opaque<uint32_t>(static_cast<uint32_t>(__cvta_generic_to_shared(empty_bar)));
  if (threadIdx.x == 0) {
    for (int st = 0; st < NUM_STAGES; ++st) {
      mbarrier_init(&full_bar[st], 1);
      mbarrier_init(&empty_bar[st], NUM_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  int const n_tiles_per_split = cuda::ceil_div(params.num_n_tiles, params.step_splits);
  int const n_tile_begin = blockIdx.y * n_tiles_per_split;
  int const num_local_n_tiles = min(n_tiles_per_split, params.num_n_tiles - n_tile_begin);
  int const step_begin = n_tile_begin * 8;

  int const ctas = gridDim.x * gridDim.y;
  int const cta_rank = blockIdx.y * gridDim.x + blockIdx.x;
  auto flags = [&](int point) { return params.sync_flags + point * ctas; };
  __shared__ unsigned s_go, s_arg;
  __shared__ float x_gemv[TILE_ROWS];
  __shared__ __align__(16) half s_x[MAX_KPAD_INPUTS];
  // Critical warp: the oldest window's h before this input (every layer; dynamic shared memory at crit_prev_offset)
  // and its units' c, prefetched before the go.
  half* const s_crit_prev = reinterpret_cast<half*>(smem_base + params.crit_prev_offset);
  __shared__ float s_crit_c[MAX_LAYERS][UNITS_PER_CTA];
  __shared__ __align__(16) half s_crit_in[MAX_KPAD_UNITS];  // critical warp: the layer's h of the window, all units

  // Resident weight slabs: this CTA's rows of each, chunk after chunk in the layout of a TMA stage (128-byte swizzle);
  // layer 0's input slab (kpad_inputs) first if resident, then R, then the input weights of layers 1...
  int const slab_bytes = params.kpad_units / K_CHUNK * TILE_ROWS * K_CHUNK * 2;
  int const x0_slab_bytes = params.resident_x0 ? params.kpad_inputs / K_CHUNK * TILE_ROWS * K_CHUNK * 2 : 0;
  uint32_t const resident_x0 =
      params.resident_x0 ? static_cast<uint32_t>(__cvta_generic_to_shared(smem_base)) + params.resident_offset : 0u;
  uint32_t const resident_base =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_base)) + params.resident_offset + x0_slab_bytes;
  auto copy_slab = [&](uint32_t dst, half const* src, int kpad) {
    int const pieces_per_row = kpad / 8;
    for (int i = threadIdx.x; i < TILE_ROWS * pieces_per_row; i += CTA_THREADS) {
      int const row = i / pieces_per_row, piece = i % pieces_per_row;
      uint4 const v = *reinterpret_cast<uint4 const*>(src + static_cast<std::size_t>(row) * kpad + piece * 8);
      uint32_t const addr = dst + (piece / 8) * TILE_ROWS * 128 + swizzled_offset<128>(row, piece % 8);
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                   : "memory");
    }
  };
  auto resident_wh = [&](int layer) -> uint32_t {
    if (!((params.resident_wh >> layer) & 1)) return 0;
    return resident_base + __popc(params.resident_wh & ((1u << layer) - 1)) * slab_bytes;
  };
  auto resident_wxr = [&](int layer) -> uint32_t {
    if (!((params.resident_wxr >> layer) & 1)) return 0;
    return resident_base +
           (__popc(params.resident_wh) + __popc(params.resident_wxr & ((1u << layer) - 1))) * slab_bytes;
  };
  for (int k = threadIdx.x; k < MAX_KPAD_UNITS; k += CTA_THREADS) s_crit_in[k] = __float2half(0.f);
  if (resident_x0)
    copy_slab(
        resident_x0,
        params.gmem_weights_x0 + static_cast<std::size_t>(cta_id) * TILE_ROWS * params.kpad_inputs,
        params.kpad_inputs);
  for (int layer = 0; layer < params.num_layers; ++layer) {
    if (uint32_t const dst = resident_wh(layer))
      copy_slab(
          dst,
          params.gmem_weights_h[layer] + static_cast<std::size_t>(cta_id) * TILE_ROWS * params.kpad_units,
          params.kpad_units);
    if (uint32_t const dst = layer > 0 ? resident_wxr(layer) : 0u)
      copy_slab(
          dst,
          params.gmem_weights_xr +
              (static_cast<std::size_t>(layer - 1) * params.unit_ctas + cta_id) * TILE_ROWS * params.kpad_units,
          params.kpad_units);
  }
  __syncthreads();

  // Layer 0's input weights of this warp's rows of the input GEMV (rows warp_id + r * NUM_WARPS) stay in registers for
  // the whole launch when K takes one 16-byte load per lane (kpad_inputs <= 256): they do not depend on the input, and
  // loading them after the go put a round trip to L2 per row on every pass. Bulk warps only: the critical warp
  // (warp_id == NUM_WARPS) does not run the GEMV, and its rows would lie past the CTA's slab.
  constexpr int X0_ROWS = TILE_ROWS / NUM_WARPS;
  static_assert(TILE_ROWS % NUM_WARPS == 0, "the input GEMV gives every warp the same number of rows");
  half const* const w_x0_slab =
      params.gmem_weights_x0 + static_cast<std::size_t>(cta_id) * TILE_ROWS * params.kpad_inputs;
  bool const x0_resident = params.kpad_inputs <= WARP_SIZE * 8;
  uint4 w_x0[X0_ROWS];
#pragma unroll
  for (int r = 0; r < X0_ROWS; ++r)
    w_x0[r] =
        x0_resident && warp_id < NUM_WARPS && lane_id * 8 < params.kpad_inputs
            ? *reinterpret_cast<uint4 const*>(
                  w_x0_slab + static_cast<std::size_t>(warp_id + r * NUM_WARPS) * params.kpad_inputs + lane_id * 8)
            : make_uint4(0, 0, 0, 0);

  // The critical warp's step of the oldest window: lane = slab row (gate row / 8, unit cta_id * 8 + row % 8). Per
  // layer: the gates from the layer's input (x, or the previous layer's new h of the window) and the window's h before
  // this input; c and h of the CTA's units; the units' h to every CTA (exchange buffer + epoch flag); then, with the
  // layer's new h of all units, this CTA's rows of the slot's R * h. Weights from the resident slabs or global memory.
  // Before the go: the oldest window's states before this input, every layer (row num_steps - 2 of the previous
  // pass's buffer, which every CTA's bulk has finished once the previous pass's flags are in).
  auto critical_prefetch = [&](int in, unsigned pass) {
    rp_wait_flags(flags(params.num_layers - 1), ctas, pass - 1);
    for (int layer = 0; layer < params.num_layers; ++layer) {
      std::size_t const prev =
          (static_cast<std::size_t>(layer) * params.act_rows + params.num_steps - 2) * params.kpad_units;
      for (int k = lane_id * 8; k < params.kpad_units; k += WARP_SIZE * 8)
        *reinterpret_cast<uint4*>(s_crit_prev + layer * params.kpad_units + k) =
            __ldcg(reinterpret_cast<uint4 const*>(params.gmem_h[in] + prev + k));
      int const unit = cta_id * UNITS_PER_CTA + lane_id;
      if (lane_id < UNITS_PER_CTA)
        s_crit_c[layer][lane_id] =
            unit < params.num_units ? __half2float(__ldcg(params.gmem_c[in] + prev + unit)) : 0.f;
    }
    __syncwarp();
  };

  auto critical_step = [&](int in, int out, int slot, unsigned pass) {
    int const row = lane_id;
    int const gate = row / 8;
    int const unit = cta_id * UNITS_PER_CTA + row % 8;
    bool const valid = unit < params.num_units;
    std::size_t const units = params.num_units;
    // The CTA's 32 slab rows times a vector in shared memory on the tensor cores, the result for row `lane` in that
    // lane: m16n8k16 with the vector as the only non-zero column of B (lanes 0-3 hold it). The A fragments come by
    // ldmatrix from the resident slab (swizzled, chunk after chunk, as in the bulk's GEMMs), or, for a slab in global
    // memory, as each lane's own fragment words by plain loads, four k-steps at a time so that they are in flight
    // together. Column 0 of the result is in lanes 4 r' (rows r' and r' + 8 of each m16 tile), gathered to lane = row
    // by four shuffles.
    auto slab_dot = [&](uint32_t res, half const* gslab, int kpad, half const* vec) {
      float c[2][4] = {};
      auto b_frag = [&](int kk, uint32_t (&b)[2]) {
        b[0] = b[1] = 0u;
        if (lane_id < 4) {
          b[0] = *reinterpret_cast<uint32_t const*>(vec + kk * 16 + lane_id * 2);
          b[1] = *reinterpret_cast<uint32_t const*>(vec + kk * 16 + 8 + lane_id * 2);
        }
      };
      if (res != 0) {
        for (int kk = 0; kk < kpad / 16; ++kk) {
          uint32_t b[2];
          b_frag(kk, b);
          uint32_t const chunk_base = res + (kk / 4) * TILE_ROWS * 128;
          int const piece = (kk % 4) * 2 + (lane_id >> 4);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            uint32_t a[4];
            ldmatrix_x4_addr(a, chunk_base + swizzled_offset<128>(mt * 16 + (lane_id & 15), piece));
            mma_m16n8k16(c[mt], a, b);
          }
        }
      } else {
        // Fragment words of row-major A: a0 (row g, k 2t), a1 (row g + 8, k 2t), a2 (row g, k 2t + 8), a3 (row g + 8,
        // k 2t + 8), two halves each; kpad is a multiple of 64, so the k-steps come in fours.
        int const g = lane_id >> 2, t = lane_id & 3;
        for (int kk0 = 0; kk0 < kpad / 16; kk0 += 4) {
          uint32_t a[4][2][4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              half const* const p = gslab + static_cast<std::size_t>(mt * 16 + g) * kpad + (kk0 + j) * 16 + 2 * t;
              a[j][mt][0] = __ldg(reinterpret_cast<uint32_t const*>(p));
              a[j][mt][1] = __ldg(reinterpret_cast<uint32_t const*>(p + 8 * kpad));
              a[j][mt][2] = __ldg(reinterpret_cast<uint32_t const*>(p + 8));
              a[j][mt][3] = __ldg(reinterpret_cast<uint32_t const*>(p + 8 * kpad + 8));
            }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t b[2];
            b_frag(kk0 + j, b);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) mma_m16n8k16(c[mt], a[j][mt], b);
          }
        }
      }
      int const src = (lane_id % 8) * 4;
      float const v00 = __shfl_sync(0xffffffff, c[0][0], src), v02 = __shfl_sync(0xffffffff, c[0][2], src);
      float const v10 = __shfl_sync(0xffffffff, c[1][0], src), v12 = __shfl_sync(0xffffffff, c[1][2], src);
      return lane_id < 8 ? v00 : lane_id < 16 ? v02 : lane_id < 24 ? v10 : v12;
    };
    for (int layer = 0; layer < params.num_layers; ++layer) {
      half const* const h_prev = s_crit_prev + layer * params.kpad_units;
      float const c_prev = row < 8 ? s_crit_c[layer][row] : 0.f;
      half const* const r_slab =
          params.gmem_weights_h[layer] + static_cast<std::size_t>(cta_id) * TILE_ROWS * params.kpad_units;
      float const pre =
          (valid ? __half2float(params.gmem_biases[layer][gate * units + unit]) : 0.f) +
          (layer == 0 ? slab_dot(
                            resident_x0,
                            params.gmem_weights_x0 + static_cast<std::size_t>(cta_id) * TILE_ROWS * params.kpad_inputs,
                            params.kpad_inputs,
                            s_x)
                      : slab_dot(
                            resident_wxr(layer),
                            params.gmem_weights_xr + (static_cast<std::size_t>(layer - 1) * params.unit_ctas + cta_id) *
                                                         TILE_ROWS * params.kpad_units,
                            params.kpad_units,
                            s_crit_in)) +
          slab_dot(resident_wh(layer), r_slab, params.kpad_units, h_prev);
      // Gate rows i, f, g, o of a unit are lanes u, u + 8, u + 16, u + 24.
      float const pre_f = __shfl_down_sync(0xffffffff, pre, 8);
      float const pre_g = __shfl_down_sync(0xffffffff, pre, 16);
      float const pre_o = __shfl_down_sync(0xffffffff, pre, 24);
      float hx = 0.f;
      if (row < 8 && valid) {
        float const cx = sigma_g(pre_f) * c_prev + sigma_g(pre) * sigma_c(pre_g);
        hx = sigma_g(pre_o) * sigma_c(cx);
        params.gmem_slot_c[slot][layer * units + unit] = __float2half(cx);
      }
      // The units' h as tagged words: units 2 j and 2 j + 1 with the pass number, from the even lanes.
      unsigned long long* const tagged =
          params.gmem_crit_tagged + static_cast<std::size_t>(layer) * params.kpad_units / 2;
      float const hx_next = __shfl_down_sync(0xffffffff, hx, 1);
      if (row < 8 && row % 2 == 0 && valid) {
        __half2 const pair = __floats2half2_rn(hx, hx_next);
        unsigned long long const word =
            (static_cast<unsigned long long>(pass) << 32) | *reinterpret_cast<uint32_t const*>(&pair);
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(tagged[unit / 2])
            .store(word, cuda::memory_order_relaxed);
      }
      // The window's new h of this layer, all units: the next layer's input and the slot's R * h.
      for (int j = lane_id; j < cuda::ceil_div(params.num_units, 2); j += WARP_SIZE) {
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> w(tagged[j]);
        unsigned long long v;
        while (static_cast<int>(static_cast<unsigned>((v = w.load(cuda::memory_order_relaxed)) >> 32) - pass) < 0)
          __nanosleep(RP_SPIN_NS);
        *reinterpret_cast<uint32_t*>(s_crit_in + 2 * j) = static_cast<uint32_t>(v);
      }
      __syncwarp();
      float const rh = slab_dot(resident_wh(layer), r_slab, params.kpad_units, s_crit_in);
      if (valid) params.gmem_slot_gemm_h[slot][(layer * 4 + gate) * units + unit] = __float2half(rh);
    }
    __syncwarp();
    if (lane_id == 0) rp_set_flag(&flags(RP_FLAG_SLOT)[cta_rank], pass);
    if (cta0) rp_wait_flags(flags(RP_FLAG_SLOT), ctas, pass);
  };

  for (unsigned pass = params.first_pass;; ++pass) {
    // The previous pass done in every CTA: this pass reads its states and overwrites the buffers of the pass before it
    // (and the slot's reads of them). Usually long true when the go comes, so off the critical path.
    if (warp_id == 0) rp_wait_flags(flags(params.num_layers - 1), ctas, pass - 1);
    if (warp_id == NUM_WARPS && blockIdx.y == 0) critical_prefetch((pass + 1) & 1, pass);
    // The go. With the go words and the input in device memory (GDRCopy) every CTA waits for the host's go and reads
    // the input itself; in pinned host memory (go_relay) only CTA 0 does, over PCIe, and passes both on through device
    // memory (many CTAs polling and reading over PCIe slowed every pass several times over).
    bool const from_host = cta0 || !params.go_relay;
    if (threadIdx.x == 0) {
      unsigned seq;
      if (from_host) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_system> go(*const_cast<unsigned*>(params.host_go_seq));
        while ((seq = go.load(cuda::memory_order_acquire)) != pass && seq != GO_EXIT) {
        }
        s_arg = *const_cast<unsigned volatile*>(params.host_go_arg);
      } else {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> go(params.gmem_go[0]);
        while ((seq = go.load(cuda::memory_order_acquire)) != pass && seq != GO_EXIT) {
        }
        s_arg = *const_cast<unsigned volatile*>(&params.gmem_go[1]);
      }
      s_go = seq;
    }
    __syncthreads();
    if (s_go != GO_EXIT) {
      // __ldcv: the host rewrites the input slot every pass (pinned host memory, or device memory through GDRCopy)
      half const* const src = from_host ? params.host_input : params.gmem_x;
      for (int k = threadIdx.x; k < params.kpad_inputs; k += CTA_THREADS) {
        half const v = k < params.num_inputs ? __ldcv(src + k) : __float2half(0.f);
        s_x[k] = v;
        if (cta0 && params.go_relay) params.gmem_x[k] = v;
      }
    }
    if (cta0 && params.go_relay) {
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        params.gmem_go[1] = s_arg;
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(params.gmem_go[0])
            .store(s_go, cuda::memory_order_release);
      }
    }
    __syncthreads();
    if (s_go == GO_EXIT) break;
    unsigned const arg = s_arg & ~GO_RELOAD;
    int const in = (pass + 1) & 1, out = pass & 1;  // the previous pass's states, and this pass's
    int const slot = (arg - 1) & 1;

    if (warp_id == NUM_WARPS) {
      // The critical warp: the oldest window's step with this input (the window that has seen num_steps - 2 inputs,
      // in row num_steps - 2 of the previous pass's buffer, which every CTA's bulk finished before the go), one slab
      // row per lane, and the consumer's slot from it. The bulk computes the same window again with the others.
      // With step splits, the CTA with the unit tile's first steps (blockIdx.y 0) does it for the tile; the others only
      // report their (empty) part of the slot.
      if (arg != 0 && blockIdx.y == 0) {
        if (s_arg & GO_RELOAD) critical_prefetch(in, pass);
        critical_step(in, out, slot, pass);
      }
      if (arg != 0 && blockIdx.y != 0 && lane_id == 0) rp_set_flag(&flags(RP_FLAG_SLOT)[cta_rank], pass);
      if (cta0 && lane_id == 0) {
        // The ready word is read on the device (the inference kernel); the host only takes the published word as its
        // cue to signal the next step, whose kernel waits for the ready word itself, so no system-scope fence.
        if (arg != 0)
          cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*params.ready).store(arg, cuda::memory_order_release);
        cuda::atomic_ref<unsigned, cuda::thread_scope_system>(*params.host_published)
            .store(pass, cuda::memory_order_relaxed);
      }
      continue;
    }

    for (int layer = 0; layer < params.num_layers; ++layer) {
      float acc[NUM_M_TILES][N_TILES_PER_WARP][4];
#pragma unroll
      for (int mt = 0; mt < NUM_M_TILES; ++mt)
#pragma unroll
        for (int t = 0; t < N_TILES_PER_WARP; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][t][e] = 0.f;

      // Layer 0's input term W_x * x is the same for every window: one GEMV, overlapped with the first TMA copies of
      // the recurrent term; every thread then picks the rows of its accumulator fragments.
      float x_term[NUM_M_TILES][4];
#pragma unroll
      for (int mt = 0; mt < NUM_M_TILES; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) x_term[mt][e] = 0.f;
      auto layer0_gemv = [&]() {
        if (layer != 0) return;
        auto dot8 = [](float& sum, uint4 const& wv, uint4 const& xv) {
          auto const* wh = reinterpret_cast<__half2 const*>(&wv);
          auto const* xh = reinterpret_cast<__half2 const*>(&xv);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float2 const wf = __half22float2(wh[q]);
            float2 const xf = __half22float2(xh[q]);
            sum += wf.x * xf.x + wf.y * xf.y;
          }
        };
        float sum[X0_ROWS];
#pragma unroll
        for (int r = 0; r < X0_ROWS; ++r) sum[r] = 0.f;
        if (x0_resident) {
          if (lane_id * 8 < params.kpad_inputs) {
            uint4 const xv = *reinterpret_cast<uint4 const*>(s_x + lane_id * 8);
#pragma unroll
            for (int r = 0; r < X0_ROWS; ++r) dot8(sum[r], w_x0[r], xv);
          }
        } else {
          for (int k0 = lane_id * 8; k0 < params.kpad_inputs; k0 += WARP_SIZE * 8) {
            uint4 wv[X0_ROWS];
#pragma unroll
            for (int r = 0; r < X0_ROWS; ++r)  // all of the warp's rows in flight together
              wv[r] = *reinterpret_cast<uint4 const*>(
                  w_x0_slab + static_cast<std::size_t>(warp_id + r * NUM_WARPS) * params.kpad_inputs + k0);
            uint4 const xv = *reinterpret_cast<uint4 const*>(s_x + k0);
#pragma unroll
            for (int r = 0; r < X0_ROWS; ++r) dot8(sum[r], wv[r], xv);
          }
        }
#pragma unroll
        for (int r = 0; r < X0_ROWS; ++r) {
#pragma unroll
          for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], offset);
          if (lane_id == 0) x_gemv[warp_id + r * NUM_WARPS] = sum[r];
        }
        bulk_sync();
#pragma unroll
        for (int mt = 0; mt < NUM_M_TILES; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) x_term[mt][e] = x_gemv[mt * 16 + (lane_id >> 2) + (e >= 2 ? 8 : 0)];
      };

      // Layer 0: the previous pass's h, written with ordinary stores by every CTA (their flags awaited at the pass's
      // start, before the go's barrier) or cleared by Init: make it visible to the async proxy before the TMA reads.
      if (layer == 0) asm volatile("fence.proxy.async;\n" ::: "memory");
      // Recurrent term R * h of the previous pass's windows: rows [layer][0, num_steps) of the previous buffer.
      fused_gemm_segment_tma<UNITS_PER_CTA, BLOCK_THREADS, K_CHUNK, NUM_STAGES, K_GROUPS>(
          acc,
          &params.map_wh,
          (layer * params.unit_ctas + cta_id) * TILE_ROWS,
          &params.map_h[in],
          layer * params.act_rows + step_begin,
          params.kpad_units,
          num_local_n_tiles,
          n_tiles_box_rows,
          smem_a,
          smem_b,
          full_bar_s,
          empty_bar_s,
          chunk_counter,
          resident_wh(layer),
          layer0_gemv);

      if (layer > 0) {
        // The previous layer's h of this pass, from every CTA.
        if (warp_id == 0) rp_wait_flags(flags(layer - 1), ctas, pass);
        bulk_sync();
        // Written with ordinary stores by other CTAs: make it visible to the async proxy.
        asm volatile("fence.proxy.async;\n" ::: "memory");
        // Input term: the previous layer's h of this pass, window s in row s + 1.
        fused_gemm_segment_tma<UNITS_PER_CTA, BLOCK_THREADS, K_CHUNK, NUM_STAGES, K_GROUPS>(
            acc,
            &params.map_wxr,
            ((layer - 1) * params.unit_ctas + cta_id) * TILE_ROWS,
            &params.map_h[out],
            (layer - 1) * params.act_rows + 1 + step_begin,
            params.kpad_units,
            num_local_n_tiles,
            n_tiles_box_rows,
            smem_a,
            smem_b,
            full_bar_s,
            empty_bar_s,
            chunk_counter,
            resident_wxr(layer),
            [] {});
      }

      // Epilogue: gates, cell update, h, for the n-tile slots this warp owns; window s -> row s + 1.
      half const* c_in = params.gmem_c[in] + static_cast<std::size_t>(layer) * params.act_rows * params.kpad_units;
      half* c_out = params.gmem_c[out] + static_cast<std::size_t>(layer) * params.act_rows * params.kpad_units;
      half* h_out = params.gmem_h[out] + static_cast<std::size_t>(layer) * params.act_rows * params.kpad_units;
#pragma unroll
      for (int pr = 0; pr < NUM_TILE_PAIRS; ++pr) {
        int const unit = cta_id * UNITS_PER_CTA + pr * 8 + (lane_id >> 2);
        if (unit >= params.num_units) continue;
        auto const bias = params.gmem_biases[layer];
        float const bias_i = __half2float(bias[unit]);
        float const bias_f = __half2float(bias[params.num_units + unit]);
        float const bias_c = __half2float(bias[2 * params.num_units + unit]);
        float const bias_o = __half2float(bias[3 * params.num_units + unit]);
#pragma unroll
        for (int t = 0; t < N_TILES_PER_WARP; ++t) {
          int const j = n_group + t * N_GROUPS;
          if (j >= num_local_n_tiles) continue;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            int const step = step_begin + j * 8 + (lane_id & 3) * 2 + e;
            if (step >= params.num_steps) continue;
            std::size_t const offset = static_cast<std::size_t>(step) * params.kpad_units + unit;
            float const gate_in = sigma_g(acc[2 * pr][t][e] + x_term[2 * pr][e] + bias_i);
            float const gate_forget = sigma_g(acc[2 * pr][t][2 + e] + x_term[2 * pr][2 + e] + bias_f);
            float const gate_c_tilde = sigma_c(acc[2 * pr + 1][t][e] + x_term[2 * pr + 1][e] + bias_c);
            float const gate_out = sigma_g(acc[2 * pr + 1][t][2 + e] + x_term[2 * pr + 1][2 + e] + bias_o);
            float const cx = gate_forget * __half2float(c_in[offset]) + gate_in * gate_c_tilde;
            c_out[offset + params.kpad_units] = __float2half(cx);
            h_out[offset + params.kpad_units] = __float2half(gate_out * sigma_c(cx));
          }
        }
      }
      // The stage buffers are reused by the next layer's first loads; this CTA's h of the layer is written.
      bulk_sync();
      if (threadIdx.x == 0) rp_set_flag(&flags(layer)[cta_rank], pass);
    }

    // The done word: every CTA's bulk through the pass, and every CTA's part of the slot.
    if (cta0 && warp_id == 0) {
      rp_wait_flags(flags(params.num_layers - 1), ctas, pass);
      if (arg != 0) rp_wait_flags(flags(RP_FLAG_SLOT), ctas, pass);
    }
    if (cta0 && threadIdx.x == 0)
      cuda::atomic_ref<unsigned, cuda::thread_scope_system>(*params.host_done).store(pass, cuda::memory_order_release);
  }
}

}  // namespace
}  // namespace nvidia::quant::lowlat_infer::single_step
