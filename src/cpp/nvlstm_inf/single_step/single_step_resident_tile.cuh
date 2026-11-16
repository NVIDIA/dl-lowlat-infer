/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The tensor-core tile of the resident SingleStep precompute (single_step_resident_kernel.cuh), kept apart for other
// kernels with the same
// tile: FP16 mma.sync m16n8k16 with FP32 accumulation, A (weights, a slab of 4 * UNITS_PER_CTA rows per CTA with the
// (i,f) / (g,o) row pairing) and B (window states) streamed by TMA through an mbarrier pipeline into 128-byte-swizzled
// stages, fragments by ldmatrix; and the host helpers that pack the weights into those slabs and describe the buffers
// to TMA.

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda/cmath>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cuda_error_util.h"

namespace nvidia::quant::lowlat_infer::single_step::resident_tile {

constexpr int WARP_SIZE = 32;
constexpr int MAX_N_TILES = 16;  // n8 tiles: up to 128 window steps
constexpr bool USE_TMA = true;
constexpr bool SKIP_A_LOADS = false;
constexpr bool SKIP_B_LOADS = false;

__forceinline__ __device__ float sigma_g(float in) { return __fdividef(1.f, (1.f + __expf(-in))); }
__forceinline__ __device__ float sigma_c(float in) { return tanhf(in); }

// Hides a value from the optimiser.  The compiler otherwise rematerialises cheap per-thread values (warp and lane
// indices, precomputed fragment offsets) from threadIdx inside the hot loops instead of keeping them in registers; ncu
// showed those recomputations as more than half of all executed instructions.
template <typename T>
__forceinline__ __device__ T opaque(T v) {
  asm volatile("" : "+r"(v));
  return v;
}

// Variants taking the 32-bit shared-space address directly, for loops that precompute it.
__forceinline__ __device__ void ldmatrix_x4_addr(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__forceinline__ __device__ void ldmatrix_x2_addr(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__forceinline__ __device__ void mma_m16n8k16(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__forceinline__ __device__ void mbarrier_init(uint64_t* bar, int count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))),
      "r"(count));
}
// Variants taking precomputed 32-bit shared-space addresses (avoid a cvta per call in the hot loop).
__forceinline__ __device__ void mbarrier_expect_tx_addr(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__forceinline__ __device__ void mbarrier_arrive_addr(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__forceinline__ __device__ void mbarrier_wait_addr(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra WAIT_DONE;\n"
      "bra WAIT_LOOP;\n"
      "WAIT_DONE:\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__forceinline__ __device__ void tma_load_2d_addr(
    uint32_t smem_dst, CUtensorMap const* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
      "[%4];\n" ::"r"(smem_dst),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(c0),
      "r"(c1),
      "r"(bar)
      : "memory");
}
__forceinline__ __device__ void ldmatrix_x4_b_addr(uint32_t (&r0)[2], uint32_t (&r1)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0[0]), "=r"(r0[1]), "=r"(r1[0]), "=r"(r1[1])
               : "r"(addr));
}

// Byte offset of 16-byte piece `piece` of row `row` of a tile with rows of ROW_BYTES, XOR swizzled against ldmatrix
// bank conflicts.
template <int ROW_BYTES>
__forceinline__ __device__ int swizzled_offset(int row, int piece) {
  return row * ROW_BYTES + ((piece ^ (row & 7)) * 16);
}

// Accumulator geometry of one warp: NUM_M_TILES m16 tiles x N_TILES_PER_WARP n8 tiles.  The warps of a CTA form a
// K_GROUPS x N_GROUPS grid: the K_GROUPS warps of one n-group split the k16 steps of every K chunk between them, so a
// warp loads its A fragments once per k step and reuses them for all of its n-tiles; the partial accumulators are
// reduced across the K groups once per layer.
template <int UNITS_PER_CTA, int BLOCK_THREADS, int K_GROUPS>
struct Warp_tiling {
  enum { NUM_WARPS = BLOCK_THREADS / WARP_SIZE };
  enum { N_GROUPS = NUM_WARPS / K_GROUPS };
  enum { NUM_M_TILES = UNITS_PER_CTA / 4 };
  enum { N_TILES_PER_WARP = cuda::ceil_div(MAX_N_TILES, N_GROUPS) };
  // Floats of partial accumulator exchanged through shared memory: every warp writes the n-tiles it does not own.
  enum { REDUCE_BYTES = NUM_WARPS * N_TILES_PER_WARP * NUM_M_TILES * 4 * WARP_SIZE * 4 };
};

// Shared memory of one CTA: the stages, or the reduction buffer if that is larger (they are not live
// together).
inline std::size_t fused_smem_bytes(
    int units_per_cta, int block_threads, int k_chunk, int num_stages, int k_groups, int local_n_tiles) {
  std::size_t const stages =
      static_cast<std::size_t>(num_stages) * (units_per_cta * 4 + local_n_tiles * 8) * (k_chunk * 2);
  int const num_warps = block_threads / WARP_SIZE;
  int const n_groups = num_warps / k_groups;
  int const n_tiles_per_warp = cuda::ceil_div(MAX_N_TILES, n_groups);
  std::size_t const reduce =
      static_cast<std::size_t>(num_warps) * n_tiles_per_warp * (units_per_cta / 4) * 4 * WARP_SIZE * 4;
  // TMA: the stage buffers are aligned up to 1024 bytes by hand, so keep that much slack.
  return std::max(stages, reduce) + (USE_TMA ? 1024 : 0);
}

// One K segment through TMA: acc += W_slab[TILE_ROWS x kpad] * B[rows x kpad]^T, over this warp's share of every K
// chunk.  Pipeline: full_bar[s] completes when the TMA copies of a chunk have landed
// (the producer arms it with the byte count); empty_bar[s] completes when every warp has finished reading the chunk.
// Thread 0 is the producer: after its own warp has consumed chunk c it waits for the other warps to release the same
// stage and refills it with chunk c + NUM_STAGES, so the other warps never block on a block-wide barrier.  The chunk
// counter runs across segments and layers so the barrier phases stay consistent.
template <int UNITS_PER_CTA, int BLOCK_THREADS, int K_CHUNK, int NUM_STAGES, int K_GROUPS, typename AfterPrologue>
__forceinline__ __device__ void fused_gemm_segment_tma(
    float (&acc)[Warp_tiling<UNITS_PER_CTA, BLOCK_THREADS, K_GROUPS>::NUM_M_TILES]
                [Warp_tiling<UNITS_PER_CTA, BLOCK_THREADS, K_GROUPS>::N_TILES_PER_WARP][4],
    CUtensorMap const* map_a,
    int a_row0,
    CUtensorMap const* map_b,
    int b_row0,
    int kpad,
    int num_local_n_tiles,
    int n_tiles_box,
    char* smem_a,
    char* smem_b,
    uint32_t full_bar_s,
    uint32_t empty_bar_s,
    uint32_t& chunk_counter,
    uint32_t a_resident,  // shared address of the A slab resident in the stage layout (chunk after chunk), 0: streamed
    AfterPrologue&& after_prologue) {
  using Tiling = Warp_tiling<UNITS_PER_CTA, BLOCK_THREADS, K_GROUPS>;
  enum { N_GROUPS = Tiling::N_GROUPS };
  enum { TILE_ROWS = UNITS_PER_CTA * 4 };
  enum { NUM_M_TILES = Tiling::NUM_M_TILES };
  enum { N_TILES_PER_WARP = Tiling::N_TILES_PER_WARP };
  enum { NUM_SUB_TILES = K_CHUNK / 64 };
  enum { KS_PER_GROUP = K_CHUNK / 16 / K_GROUPS };

  int const lane_id = opaque<int>(threadIdx.x % WARP_SIZE);
  int const warp_id = opaque<int>(threadIdx.x / WARP_SIZE);
  int const k_group = opaque<int>(warp_id % K_GROUPS);
  int const n_group = opaque<int>(warp_id / K_GROUPS);
  int const a_sub_bytes = TILE_ROWS * 128;
  int const b_sub_bytes = n_tiles_box * 8 * 128;
  int const a_stage_bytes = NUM_SUB_TILES * a_sub_bytes;
  int const b_stage_bytes = NUM_SUB_TILES * b_sub_bytes;
  int const num_chunks = kpad / K_CHUNK;
  bool const is_producer = threadIdx.x == 0;

  uint32_t const smem_a_base = opaque<uint32_t>(static_cast<uint32_t>(__cvta_generic_to_shared(smem_a)));
  uint32_t const smem_b_base = opaque<uint32_t>(static_cast<uint32_t>(__cvta_generic_to_shared(smem_b)));
  auto issue = [&](int chunk, uint32_t global_chunk) {
    uint32_t const stage = global_chunk % NUM_STAGES;
    uint32_t const full = full_bar_s + 8 * stage;
    bool const load_a = !SKIP_A_LOADS && a_resident == 0;
    mbarrier_expect_tx_addr(full, (load_a ? a_stage_bytes : 0) + (SKIP_B_LOADS ? 0 : b_stage_bytes));
#pragma unroll
    for (int j = 0; j < NUM_SUB_TILES; ++j) {
      if (load_a) {
        tma_load_2d_addr(
            smem_a_base + stage * a_stage_bytes + j * a_sub_bytes, map_a, chunk * K_CHUNK + 64 * j, a_row0, full);
      }
      if (!SKIP_B_LOADS) {
        tma_load_2d_addr(
            smem_b_base + stage * b_stage_bytes + j * b_sub_bytes, map_b, chunk * K_CHUNK + 64 * j, b_row0, full);
      }
    }
  };

  // The fragment addresses do not change from chunk to chunk except for the stage base, so compute the per-lane offsets
  // (sub-tile, row, swizzled piece) once per segment instead of for every ldmatrix of every chunk.
  uint32_t a_offset[KS_PER_GROUP][NUM_M_TILES];
  uint32_t b_offset[KS_PER_GROUP][N_TILES_PER_WARP];
#pragma unroll
  for (int i = 0; i < KS_PER_GROUP; ++i) {
    int const ks = k_group * KS_PER_GROUP + i;
#pragma unroll
    for (int mt = 0; mt < NUM_M_TILES; ++mt) {
      int const row = mt * 16 + (lane_id & 15);
      int const piece = ks * 2 + (lane_id >> 4);
      a_offset[i][mt] = opaque<uint32_t>((piece >> 3) * a_sub_bytes + swizzled_offset<128>(row, piece & 7));
    }
#pragma unroll
    for (int t = 0; t < N_TILES_PER_WARP; ++t) {
      int const n = (n_group + t * N_GROUPS) * 8 + (lane_id & 7);
      // x4 (even KS_PER_GROUP): the four matrices are four consecutive pieces; x2 (odd): two pieces.
      int const piece = ks * 2 + ((KS_PER_GROUP % 2 == 0) ? (lane_id >> 3) : ((lane_id >> 3) & 1));
      b_offset[i][t] = opaque<uint32_t>((piece >> 3) * b_sub_bytes + swizzled_offset<128>(n, piece & 7));
    }
  }
  if (is_producer) {
    for (int i = 0; i < NUM_STAGES && i < num_chunks; ++i) {
      uint32_t const g = chunk_counter + i;
      if (g >= NUM_STAGES) {
        mbarrier_wait_addr(
            empty_bar_s + 8 * (g % NUM_STAGES), ((g / NUM_STAGES) - 1) & 1);  // previous use of the stage
      }
      issue(i, g);
    }
  }
  // All threads run this while the first TMA copies are in flight.
  after_prologue();

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    uint32_t const g = chunk_counter + chunk;
    int const stage = g % NUM_STAGES;
    mbarrier_wait_addr(full_bar_s + 8 * stage, (g / NUM_STAGES) & 1);

    uint32_t const a_stage_addr =
        a_resident != 0 ? a_resident + chunk * a_stage_bytes : smem_a_base + stage * a_stage_bytes;
    uint32_t const b_stage_addr = smem_b_base + stage * b_stage_bytes;
    uint32_t b_next[N_TILES_PER_WARP][2];
#pragma unroll
    for (int i = 0; i < KS_PER_GROUP; ++i) {
      uint32_t a_frag[NUM_M_TILES][4];
#pragma unroll
      for (int mt = 0; mt < NUM_M_TILES; ++mt) {
        ldmatrix_x4_addr(a_frag[mt], a_stage_addr + a_offset[i][mt]);
      }
#pragma unroll
      for (int t = 0; t < N_TILES_PER_WARP; ++t) {
        int const j = n_group + t * N_GROUPS;
        if (j < num_local_n_tiles) {
          uint32_t b_frag[2];
          if constexpr (KS_PER_GROUP % 2 == 0) {
            // One x4 load covers this k step and the next (consecutive pieces of the same row); keep the second half
            // for the next iteration.
            if (i % 2 == 0) {
              ldmatrix_x4_b_addr(b_frag, b_next[t], b_stage_addr + b_offset[i][t]);
            } else {
              b_frag[0] = b_next[t][0];
              b_frag[1] = b_next[t][1];
            }
          } else {
            ldmatrix_x2_addr(b_frag, b_stage_addr + b_offset[i][t]);
          }
#pragma unroll
          for (int mt = 0; mt < NUM_M_TILES; ++mt) {
            mma_m16n8k16(acc[mt][t], a_frag[mt], b_frag);
          }
        }
      }
    }

    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive_addr(empty_bar_s + 8 * stage);
    }
    if (is_producer && chunk + NUM_STAGES < num_chunks) {
      mbarrier_wait_addr(empty_bar_s + 8 * stage, (g / NUM_STAGES) & 1);  // every warp has released this stage
      issue(chunk + NUM_STAGES, g + NUM_STAGES);
    }
  }
  chunk_counter += num_chunks;
}

// Gate-major [4U][K] -> per-CTA slabs [cta][4 * units_per_cta][kpad] with the (i,f) / (g,o) row pairing of the tile.
inline std::vector<half> pack_weights(std::vector<half> const& w, int num_units, int k, int kpad, int units_per_cta) {
  int const num_ctas = cuda::ceil_div(num_units, units_per_cta);
  int const tile_rows = units_per_cta * 4;
  std::vector<half> out(static_cast<std::size_t>(num_ctas) * tile_rows * kpad, __float2half(0.f));
  for (int cta = 0; cta < num_ctas; ++cta) {
    for (int pr = 0; pr < units_per_cta / 8; ++pr) {
      for (int rt = 0; rt < 2; ++rt) {
        for (int hi = 0; hi < 2; ++hi) {
          for (int r = 0; r < 8; ++r) {
            int const unit = cta * units_per_cta + pr * 8 + r;
            if (unit >= num_units) continue;
            int const gate = (rt == 0) ? (hi ? 1 : 0) : (hi ? 3 : 2);
            int const local_row = (2 * pr + rt) * 16 + hi * 8 + r;
            std::copy_n(
                w.data() + static_cast<std::size_t>(gate * num_units + unit) * k,
                k,
                out.data() + (static_cast<std::size_t>(cta) * tile_rows + local_row) * kpad);
          }
        }
      }
    }
  }
  return out;
}

// A 2D FP16 tensor map with 64-element boxes of box_rows rows, 128-byte swizzle; elements past `inner` read as zero.
// row_pitch_bytes 0: rows packed (inner * 2 bytes).
inline void make_tensor_map(
    CUtensorMap* map,
    void const* gptr,
    std::uint64_t inner,
    std::uint64_t rows,
    std::uint32_t box_rows,
    std::uint64_t row_pitch_bytes = 0) {
  cuuint64_t const dims[2] = {inner, rows};
  cuuint64_t const strides[1] = {row_pitch_bytes != 0 ? row_pitch_bytes : inner * sizeof(half)};
  cuuint32_t const box[2] = {64, box_rows};
  cuuint32_t const element_strides[2] = {1, 1};
  CUresult const r = cuTensorMapEncodeTiled(
      map,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      2,
      const_cast<void*>(gptr),
      dims,
      strides,
      box,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CuCheck(r);
}

}  // namespace nvidia::quant::lowlat_infer::single_step::resident_tile
