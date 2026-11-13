/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cooperative_groups.h>
#include <cuda/cmath>
#include <pthread.h>
#include <stdexcept>
#include <time.h>

#include "dispatch_utils.cuh"
#include "gemv_lstm.h"
#include "gemv_lstm_utils.cuh"
#include "kernel_params.h"

namespace nvidia::quant::lowlat_infer::ssgemv {

#define DEV_TIME_OVERHEAD_EXP 0

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the warp group size from the SM architecture
// Tunable parameter, depennds on the number of SMs in the device architecture
// as well as the shared memory size available per SM
int get_dev_warpgroup_size_from_sm_arch(int sm_arch) {
  // H200 and B200 support 16 SMs in a cluster
  // Other devices support 8 SMs in a cluster
  return (sm_arch == 900 || sm_arch == 1000) ? 8 : 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEV_TIME_OVERHEAD_EXP == 1
static inline __device__ void dev_early_exit(volatile int *signal, cooperative_groups::grid_group &grid) {
  if (grid.thread_rank() == 0) {
    asm volatile("st.global.wt.s32 [%0], %1;" ::"l"(signal), "r"(0));
  }
  return;
}
#endif

template <
    int NUM_LAYERS,
    int NUM_INPUTS,
    int NUM_UNITS,
    int BLOCK_THREADS,
    int WARP_GROUP_SIZE,
    Data_kind DATA_KIND,
    bool STAGE = false,
    bool DEVICE_ALWAYS_SPIN = 0,
    bool GDRCOPY_ENABLED = 0>
__launch_bounds__(BLOCK_THREADS, 1) __global__
    void gemv_lstm_dev_subwarp_regbuf_t_kernel(Lstm_bpt_params<NUM_LAYERS, NUM_INPUTS, NUM_UNITS> params) {
  using data_type = Data_type<DATA_KIND>;
  using elem4_type = typename data_type::Elem4_type;
  using elem2_type = typename data_type::Elem2_type;
  using elem_type = typename data_type::Elem_type;

  enum { WARP_SIZE = 32 };
  enum { NUM_WARPS = BLOCK_THREADS / WARP_SIZE };
  enum { NUM_SUBWARP_GROUPS = WARP_SIZE / WARP_GROUP_SIZE };
  enum { NUM_GROUPS = NUM_SUBWARP_GROUPS * NUM_WARPS };

  enum { WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS };

  enum { NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE };
  // to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS
  enum { NUM_FEATURES_UP = cuda::round_up(NUM_FEATURES, NUM_GROUPS) };

  enum {
    SMEM_SIZE = WARP_GROUP_SIZE * WEIGHT_INPUTS * sizeof(elem4_type) + 2 * NUM_FEATURES_UP * sizeof(elem_type) +
                BLOCK_THREADS * sizeof(elem4_type)
  };

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  static_assert(NUM_UNITS % NUM_WARPS == 0 && NUM_UNITS);
  static_assert(BLOCK_THREADS % WARP_GROUP_SIZE == 0);

  extern __shared__ char smem_storage[];

  auto smem_weights = reinterpret_cast<elem4_type *>(smem_storage);

  auto shared_sum = reinterpret_cast<elem4_type *>(smem_weights + WEIGHT_INPUTS * WARP_GROUP_SIZE);

  auto smem_input = reinterpret_cast<elem_type *>(shared_sum + BLOCK_THREADS);

  constexpr auto zero4 = data_type::zero4;
#if GEMM_H_VECTORIZED == 0
  constexpr elem_type zero = data_type::zero;
#endif

  auto gmem_input = reinterpret_cast<elem_type *>(params.gmem_input);
  auto gmem_h_out = reinterpret_cast<elem_type *>(params.gmem_h_out);
  auto gmem_dot_product_weights = reinterpret_cast<float *>(params.gmem_dot_prod_weights);

  elem4_type weights[NUM_FEATURES_UP / NUM_GROUPS];

  elem4_type bias[NUM_LAYERS];
  elem_type c_in[NUM_LAYERS];
  elem4_type gemm_h[NUM_LAYERS];

  static_assert(WARP_GROUP_SIZE <= WARP_SIZE);

  const int group_idx = lane_id % WARP_GROUP_SIZE;
  const int group_id = lane_id / WARP_GROUP_SIZE + NUM_SUBWARP_GROUPS * warp_id;

  namespace cg = cooperative_groups;

  cg::grid_group grid = cg::this_grid();
  int grid_block_rank = grid.block_rank();

  const int tid = grid_block_rank * WARP_GROUP_SIZE + group_idx;

  grid.sync();

  const bool valid_unit = tid < NUM_UNITS;

  float dot_product_weight = valid_unit ? gmem_dot_product_weights[tid] : 0.0f;

  auto gmem_weights = reinterpret_cast<elem4_type *>(params.gmem_weights[0]);

  enum { NUM_INPUTS_PER_GROUP = cuda::ceil_div(NUM_INPUTS, NUM_GROUPS) };
  enum { NUM_UNITS_PER_GROUP = cuda::ceil_div(NUM_UNITS, NUM_GROUPS) };

// load first layer to registers
#pragma unroll
  for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    if (NUM_INPUTS % NUM_GROUPS != 0)
      if (i >= NUM_INPUTS) continue;
    weights[ii] = (valid_unit &&
                   (i < NUM_INPUTS)
                   // handle the case when WARP_SIZE % WARP_GROUP_SIZE != 0
                   && (lane_id < WARP_GROUP_SIZE * NUM_SUBWARP_GROUPS))
                      ? gmem_weights[i * NUM_UNITS + tid]
                      : zero4;
  }

  if (valid_unit && threadIdx.x < WARP_GROUP_SIZE) {
    auto s_weights = smem_weights;

    // load therest to smem
    for (int ll = 1; ll < NUM_LAYERS; ll++) {
      gmem_weights = reinterpret_cast<elem4_type *>(params.gmem_weights[ll]);
#pragma unroll
      for (int ii = 0; ii < NUM_UNITS; ii++) {
        s_weights[ii * WARP_GROUP_SIZE + group_idx] = gmem_weights[ii * NUM_UNITS + tid];
      }
      s_weights += NUM_UNITS * WARP_GROUP_SIZE;
    }
  }

  if (valid_unit) {
#pragma unroll
    for (int ll = 0; ll < NUM_LAYERS; ll++) {
      auto gmem_biases = reinterpret_cast<elem4_type *>(params.gmem_biases[ll]);
      if (!DEVICE_ALWAYS_SPIN) {
        auto gmem_c_in = reinterpret_cast<elem_type *>(params.gmem_c_in[ll]);
        bias[ll] = gmem_biases[tid];
        c_in[ll] = gmem_c_in[tid];
#if GEMM_H_VECTORIZED == 1
        auto gmem_gemm_h = reinterpret_cast<elem4_type *>(params.gmem_gemm_h[ll]);
        gemm_h[ll] = gmem_gemm_h[tid];
#else
        union Vec4_t {
          elem_type x[4];
          elem4_type x4;
        };
        Vec4_t val;
        auto gmem_gemm_h = reinterpret_cast<elem_type *>(params.gmem_gemm_h[ll]);
        val.x[0] = gmem_gemm_h[tid];
        val.x[1] = gmem_gemm_h[tid + 1 * NUM_UNITS];
        val.x[2] = gmem_gemm_h[tid + 2 * NUM_UNITS];
        val.x[3] = gmem_gemm_h[tid + 3 * NUM_UNITS];
        gemm_h[ll] = val.x4;
#endif
      }
    }
  }

  gpu_cpu_sync_part_0(params.host_ld_complete, grid);

  // auto s_exit = reinterpret_cast<uint32_t *>(smem_input[1]);
  auto s_exit = reinterpret_cast<uint32_t *>(smem_input + NUM_FEATURES_UP);

  bool odd_step = false;
  int ibuf = 0;
#if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
#endif

  do {
    *s_exit = 0;

#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::grid_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(
        params.gmem_ld_complete, grid, &device_spin_switch, s_exit);
#else
    gpu_cpu_sync_part_1<cooperative_groups::grid_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(
        params.gmem_ld_complete, grid, s_exit);
#endif

    // shutdown the kernel
    if (DEVICE_ALWAYS_SPIN == 1) {
      if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE) {
        if (grid.thread_rank() == 0) {
          gpu_pingpong_handles::signal_atomic_relaxed<uint>(
              (uint *)params.host_ld_complete, HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
        }
        return;
      } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE) {
        odd_step = false;
        ibuf = 0;
#if AS_STATES_LD_BAR == 1
        // workaround - we must continue with preprocess to signal states load after the initX/reset
#if DISABLE_ATOMIC_EXCHANGE == 1
        device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
        gpu_cpu_sync_part_1_no_aex<cooperative_groups::grid_group, false, GDRCOPY_ENABLED>(
            params.gmem_ld_complete, grid, &device_spin_switch);
#else
        gpu_cpu_sync_part_1<cooperative_groups::grid_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, grid);
#endif
#else
        continue;
#endif
      }

// load states
#pragma unroll
      for (int ii = 0; ii < NUM_LAYERS; ii++) {
        auto gmem_c_in = (odd_step) ? reinterpret_cast<elem_type *>(params.gmem_c_out[ii])
                                    : reinterpret_cast<elem_type *>(params.gmem_c_in[ii]);
#if GEMM_H_VECTORIZED == 1
        auto gmem_gemm_h = reinterpret_cast<elem4_type *>(params.gmem_gemm_h[ii]);
        gemm_h[ii] = valid_unit ? gmem_gemm_h[tid] : zero4;
#else
        union Vec4_t {
          elem_type x[4];
          elem4_type x4;
        };
        Vec4_t val;
        auto gmem_gemm_h = reinterpret_cast<elem_type *>(params.gmem_gemm_h[ii]);
        val.x[0] = valid_unit ? gmem_gemm_h[tid] : zero;
        val.x[1] = valid_unit ? gmem_gemm_h[tid + 1 * NUM_UNITS] : zero;
        val.x[2] = valid_unit ? gmem_gemm_h[tid + 2 * NUM_UNITS] : zero;
        val.x[3] = valid_unit ? gmem_gemm_h[tid + 3 * NUM_UNITS] : zero;
        gemm_h[ii] = val.x4;
#endif
        c_in[ii] = valid_unit ? gmem_c_in[tid] : 0;
      }

#if AS_STATES_LD_BAR == 1
      // notify host that we are ready for compute
      gpu_cpu_sync_part_0(params.host_ld_complete, grid);

      // // wait for host to signal starting the computation or exit (LSTM Forward on host)
#if DISABLE_ATOMIC_EXCHANGE == 1
      gpu_cpu_sync_part_1_no_aex<cooperative_groups::grid_group, false, GDRCOPY_ENABLED>(
          params.gmem_ld_complete, grid, &device_spin_switch);
#else
      gpu_cpu_sync_part_1<cooperative_groups::grid_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, grid);
#endif
#endif
    }

#if DEV_TIME_OVERHEAD_EXP == 1
    dev_early_exit(params.host_ld_complete, grid);
#endif

    ibuf = 0;
    auto gmem_h_out_cur = gmem_h_out + ibuf * NUM_FEATURES;

    if (blockIdx.x == 0) {
      if (STAGE == false) {
        auto host_addr = reinterpret_cast<elem_type **>(params.host_addr);
        static_assert(BLOCK_THREADS * sizeof(elem4_type) >= sizeof(void *));
        auto smem_host_ptr = reinterpret_cast<elem_type **>(shared_sum);
        if (threadIdx.x == 0) {
          *smem_host_ptr = host_addr[0];
        }
        __syncthreads();
#pragma unroll
        for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS) {
          smem_input[ii] = (*smem_host_ptr)[ii];
          gmem_h_out_cur[ii] = smem_input[ii];
        }
      } else {
#pragma unroll
        for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS) {
          smem_input[ii] = __ldcv(&gmem_input[ii]);
          gmem_h_out_cur[ii] = smem_input[ii];
        }
      }
      //__syncthreads();
    }

    grid.sync();

    if (blockIdx.x != 0) {
      for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS) {
        smem_input[ii] = gmem_h_out_cur[ii];
      }
    }
    __syncthreads();

    auto acc = zero4;

// first layer
#pragma unroll
    for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
      int i = group_id * NUM_INPUTS_PER_GROUP + ii;
      acc = fma4<DATA_KIND>(weights[ii], smem_input[i], acc);
    }

    shared_sum[threadIdx.x] = acc;

    float hx;

    // need that before we can write to smem_input below
    __syncthreads();

    enum { MAX_WARP_SIZE_GROUP_SIZE = Max<WARP_SIZE, WARP_GROUP_SIZE>::VALUE };

    if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
      auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE) ? group_idx : threadIdx.x % WARP_SIZE;
#pragma unroll
      for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
        acc = add4<DATA_KIND>(acc, shared_sum[ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
      }

      if (WARP_GROUP_SIZE < WARP_SIZE) {
        // final reduction using schfl_down_sync
        for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
          acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
        }
      }
    }

    if (threadIdx.x < WARP_GROUP_SIZE) {
      acc = add4<DATA_KIND>(acc, bias[0]);
      acc = add4<DATA_KIND>(acc, gemm_h[0]);

      float4 gate = to_float4<DATA_KIND>(acc);
      gate.x = sigma_g(gate.x);
      gate.y = sigma_g(gate.y);
      gate.z = sigma_c(gate.z);
      gate.w = sigma_g(gate.w);

      auto cx = gate.y * to_float<DATA_KIND>(c_in[0]) + gate.x * gate.z;
      hx = gate.w * sigma_c(cx);

#if DEBUG_OUTPUTS == 1
      auto gmem_c_out = reinterpret_cast<elem_type *>(params.gmem_c_out[0]);
      if (valid_unit) gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif
    }
    __syncthreads();

    if (NUM_LAYERS > 1) {
      if (threadIdx.x < WARP_GROUP_SIZE) {
        if (valid_unit) {
          __stcg(&gmem_h_out_cur[tid], from_float<DATA_KIND>(hx));
        }
      }
    } else {  // NUM_LAYERS > 1
      grid.sync();
    }

    // LAYERS loop starts from 1 because we already processed layer 0
    auto s_weights = smem_weights;
    for (int ll = 1; ll < NUM_LAYERS; ll++) {
      acc = zero4;

      grid.sync();

// load the input vector
#pragma unroll
      for (int ii = threadIdx.x; ii < NUM_UNITS; ii += BLOCK_THREADS) {
        smem_input[ii] = __ldcg(&gmem_h_out_cur[ii]);
      }
      __syncthreads();

#pragma unroll
      for (int ii = 0; ii < NUM_UNITS_PER_GROUP; ii++) {
        int i = group_id * NUM_UNITS_PER_GROUP + ii;
        if (NUM_UNITS % NUM_GROUPS != 0)
          if (i >= NUM_UNITS) continue;
        acc = fma4<DATA_KIND>(s_weights[i * WARP_GROUP_SIZE + group_idx], smem_input[i], acc);
      }

      shared_sum[threadIdx.x] = acc;

      // need that before we can write to smem_input below
      __syncthreads();

      if (ll < NUM_LAYERS - 1) {
        s_weights += NUM_UNITS * WARP_GROUP_SIZE;
      }

      if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
        auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE) ? group_idx : threadIdx.x % WARP_SIZE;
#pragma unroll
        for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
          acc = add4<DATA_KIND>(acc, shared_sum[ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
        }

        if (WARP_GROUP_SIZE < WARP_SIZE) {
          // final reduction using schfl_down_sync
          for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
            acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
          }
        }
      }

      if (threadIdx.x < WARP_GROUP_SIZE) {
        acc = add4<DATA_KIND>(acc, bias[ll]);
        acc = add4<DATA_KIND>(acc, gemm_h[ll]);

        float4 gate = to_float4<DATA_KIND>(acc);
        gate.x = sigma_g(gate.x);
        gate.y = sigma_g(gate.y);
        gate.z = sigma_c(gate.z);
        gate.w = sigma_g(gate.w);

        auto cx = gate.y * to_float<DATA_KIND>(c_in[ll]) + gate.x * gate.z;
        hx = gate.w * sigma_c(cx);
      }
      __syncthreads();

      ibuf ^= 1;
      gmem_h_out_cur = gmem_h_out + ibuf * NUM_FEATURES;
      if (ll < NUM_LAYERS - 1) {
        // broadcast hidden state to all the SMs in the cluster
        if (threadIdx.x < WARP_GROUP_SIZE) {
          if (valid_unit) __stcg(&gmem_h_out_cur[tid], from_float<DATA_KIND>(hx));
        }
      }
#if DEBUG_OUTPUTS == 1
      else {
        if (threadIdx.x < WARP_GROUP_SIZE) {
          if (valid_unit) __stcg(&gmem_h_out_cur[tid], from_float<DATA_KIND>(hx));
        }
      }
      auto gmem_c_out = reinterpret_cast<elem_type *>(params.gmem_c_out[ll]);
      gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif
    }  // layers loop

    // Below we need to find the scallar product between dot_prod weights and hx
    enum { MIN_WARP_SIZE_GROUP_SIZE = Min<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };
    enum { NUM_WARPS_IN_GROUP = cuda::ceil_div(WARP_GROUP_SIZE, WARP_SIZE) };
    static_assert(
        NUM_FEATURES * sizeof(elem_type) >= cuda::ceil_div(NUM_UNITS, WARP_GROUP_SIZE) * sizeof(float),
        "gmem_h_out sub-buffer too small for dot-product scratch");
    auto gmem_h_out_float_cur = reinterpret_cast<float *>(gmem_h_out_cur);
    float dot_product = 0.f;

    // __syncthreads();

    auto smem_tmp = reinterpret_cast<float *>(smem_input);

    // 1) Each block finds its partial sum

    if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
      dot_product = (valid_unit && threadIdx.x < WARP_GROUP_SIZE) ? dot_product_weight * hx : 0.0f;

#pragma unroll
      for (int offset = MIN_WARP_SIZE_GROUP_SIZE / 2; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
      }

      if (lane_id == 0) {
        smem_tmp[warp_id] = dot_product;
      }
    }
    if (MAX_WARP_SIZE_GROUP_SIZE > WARP_SIZE)
      __syncthreads();  // should change
    else
      __syncwarp();

    // 2)  Each block writes to block 0 its partial sum
    if (threadIdx.x == 0) {
#pragma unroll
      for (int ii = 1; ii < NUM_WARPS_IN_GROUP; ii++) {
        dot_product += smem_tmp[ii];
      }

#if DEBUG_OUTPUTS == 0
      gmem_h_out_float_cur[grid_block_rank] = dot_product;
#endif
    }

    grid.sync();

    // if (grid.block_rank() == 0) {
    //   printf("Thread %d:\n", grid.thread_rank());
    // }

    // 3) Block 0 performs the final reduction
    static_assert(BLOCK_THREADS >= 112);
    if (grid_block_rank == 0) {
      dot_product = (threadIdx.x < gridDim.x) ? gmem_h_out_float_cur[threadIdx.x] : 0.0f;

#pragma unroll
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
      }

      if (lane_id == 0) {
        smem_tmp[warp_id] = dot_product;
      }
      __syncthreads();

      if (threadIdx.x == 0) {
#pragma unroll
        for (int ii = 1; ii < NUM_WARPS; ii++) {
          dot_product += smem_tmp[ii];
        }
        dot_product += params.dot_prod_bias;
        // if (isnan(dot_product)) {
        //   printf("dot_product is nan: %f\n", dot_product);
        // }
      }
    }

    gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, grid);

  } while (DEVICE_ALWAYS_SPIN);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEV_KERNEL_RUN(KERNEL, DATA_TYPE)              \
  {                                                    \
    if (SMEM_SIZE > 0)                                 \
      CHECK_CUDA(cudaFuncSetAttribute(                 \
          KERNEL<                                      \
              NUM_LAYERS,                              \
              NUM_INPUTS,                              \
              NUM_UNITS,                               \
              BLOCK_THREADS,                           \
              WARP_GROUP_SIZE,                         \
              DATA_TYPE,                               \
              STAGE,                                   \
              DEVICE_ALWAYS_SPIN,                      \
              GDRCOPY_ENABLED>,                        \
          cudaFuncAttributeMaxDynamicSharedMemorySize, \
          SMEM_SIZE));                                 \
    void *kernel_args[] = {(void *)&params};           \
    dim3 block(BLOCK_THREADS, 1, 1);                   \
    dim3 grid(GRID_SIZE, 1, 1);                        \
    cudaLaunchCooperativeKernel(                       \
        (const void *)KERNEL<                          \
            NUM_LAYERS,                                \
            NUM_INPUTS,                                \
            NUM_UNITS,                                 \
            BLOCK_THREADS,                             \
            WARP_GROUP_SIZE,                           \
            DATA_TYPE,                                 \
            STAGE,                                     \
            DEVICE_ALWAYS_SPIN,                        \
            GDRCOPY_ENABLED>,                          \
        grid,                                          \
        block,                                         \
        kernel_args,                                   \
        SMEM_SIZE,                                     \
        stream);                                       \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Arch Arch_traits, Data_kind DATA_KIND, bool STAGE, bool DEVICE_ALWAYS_SPIN, bool GDRCOPY_ENABLED>
inline void launch_lstm_m_dev_t(Lstm_m_dev_t_params &params, cudaStream_t stream) {
  constexpr int NUM_UNITS = Lstm_m_dev_t_params::NUM_UNITS;
  constexpr int NUM_INPUTS = Lstm_m_dev_t_params::NUM_INPUTS;
  constexpr int NUM_LAYERS = Lstm_m_dev_t_params::NUM_LAYERS;

  constexpr int WARP_SIZE = 32;

  constexpr int NUM_WARPS = Arch_traits.DEV_LSTM_M_NUM_WARPS_PER_BLOCK;
  constexpr int WARP_GROUP_SIZE = Arch_traits.DEV_LSTM_M_WARP_GROUP_SIZE;

  constexpr int NUM_SUBWARP_GROUPS = WARP_SIZE / WARP_GROUP_SIZE;
  constexpr int NUM_GROUPS = NUM_SUBWARP_GROUPS * NUM_WARPS;

  constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;

  constexpr int GRID_SIZE = cuda::ceil_div(NUM_UNITS, WARP_GROUP_SIZE);

  constexpr int WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS;

  constexpr int NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE;
  // to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS
  constexpr int NUM_FEATURES_UP = cuda::round_up(NUM_FEATURES, NUM_GROUPS);

  constexpr int ELEM_SIZE = (DATA_KIND == DATA_TYPE_FP32) ? sizeof(float) : sizeof(uint16_t);

  // size per block
  constexpr int SMEM_SIZE =
      WARP_GROUP_SIZE * WEIGHT_INPUTS * ELEM_SIZE * 4 + 2 * NUM_FEATURES_UP * ELEM_SIZE + BLOCK_THREADS * ELEM_SIZE * 4;

  assert(GRID_SIZE <= params.sm_count);

  // The kernel splits the units evenly over its warps; a size it cannot split (MODEL_PARAMS_HEADER) is not compiled.
  if constexpr (NUM_UNITS % NUM_WARPS != 0) {
    throw std::runtime_error("the device-wide lstm_m kernel does not support this LSTM_M_NUM_UNITS on this device");
  } else {
    DEV_KERNEL_RUN(gemv_lstm_dev_subwarp_regbuf_t_kernel, DATA_KIND)
  }
}

void run_lstm_m_dev_t(Lstm_m_dev_t_params &params, Data_kind dtype, cudaStream_t stream) {
  kernel_dispatch::dispatch_kernel_variants<true, false>(
      dtype,                        // dtype
      params.host_addr == nullptr,  // stage
      params.always_spin != 0,      // device_always_spin
      params.use_gdrcopy != 0,      // gdrcopy_enabled
      0,                            // sm_arch ignored
      [&]<Arch A, Data_kind DT, bool S, bool P, bool G>() { launch_lstm_m_dev_t<A, DT, S, P, G>(params, stream); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Arch Arch_traits, Data_kind DATA_TYPE, bool STAGE, bool DEVICE_ALWAYS_SPIN, bool GDRCOPY_ENABLED>
void launch_lstm_l_dev_t(Lstm_l_dev_t_params &params, cudaStream_t stream) {
  constexpr int NUM_UNITS = Lstm_l_dev_t_params::NUM_UNITS;
  constexpr int NUM_INPUTS = Lstm_l_dev_t_params::NUM_INPUTS;
  constexpr int NUM_LAYERS = Lstm_l_dev_t_params::NUM_LAYERS;

  constexpr int WARP_SIZE = 32;

  constexpr int NUM_WARPS = Arch_traits.DEV_LSTM_L_NUM_WARPS_PER_BLOCK;
  constexpr int WARP_GROUP_SIZE = Arch_traits.DEV_LSTM_L_WARP_GROUP_SIZE;

  constexpr int GRID_SIZE = cuda::ceil_div(NUM_UNITS, WARP_GROUP_SIZE);

  constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;
  constexpr int NUM_GROUPS = BLOCK_THREADS / WARP_GROUP_SIZE;

  constexpr int WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS;

  constexpr int NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE;
  // to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS
  constexpr int NUM_FEATURES_UP = cuda::round_up(NUM_FEATURES, NUM_GROUPS);

  assert(GRID_SIZE <= params.sm_count);

  constexpr int ELEM_SIZE = (DATA_TYPE == DATA_TYPE_FP32) ? sizeof(float) : sizeof(uint16_t);

  // size per block
  constexpr int SMEM_SIZE =
      WARP_GROUP_SIZE * WEIGHT_INPUTS * ELEM_SIZE * 4 + 2 * NUM_FEATURES_UP * ELEM_SIZE + BLOCK_THREADS * ELEM_SIZE * 4;

  // The kernel splits the units evenly over its warps; a size it cannot split (MODEL_PARAMS_HEADER, e.g. one only the
  // FP16 register-resident kernel runs) is not compiled.
  if constexpr (NUM_UNITS % NUM_WARPS != 0) {
    throw std::runtime_error("the device-wide lstm_l kernel does not support this LSTM_L_NUM_UNITS on this device");
  } else {
    DEV_KERNEL_RUN(gemv_lstm_dev_subwarp_regbuf_t_kernel, DATA_TYPE)
  }
}

void run_lstm_l_dev_t(Lstm_l_dev_t_params &params, Data_kind dtype, cudaStream_t stream) {
  kernel_dispatch::dispatch_kernel_variants<false, true>(
      dtype,                        // dtype
      params.host_addr == nullptr,  // stage
      params.always_spin != 0,      // device_always_spin
      params.use_gdrcopy != 0,      // gdrcopy_enabled
      params.sm_arch,               // select heavy/light arch based on SM
      [&]<Arch A, Data_kind DT, bool S, bool P, bool G>() { launch_lstm_l_dev_t<A, DT, S, P, G>(params, stream); });
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv
