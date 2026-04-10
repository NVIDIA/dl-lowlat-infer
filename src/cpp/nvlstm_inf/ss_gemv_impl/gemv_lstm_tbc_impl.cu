/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pthread.h>
#include <time.h>

#include <cooperative_groups.h>
#include <cuda/ptx>
#include "gemv_lstm.h"
#include "kernel_params.h"
#include "gemv_lstm_utils.cuh"
#include "dispatch_utils.cuh"

namespace nvidia::quant::lowlat_infer::ssgemv {

#define TBC_16BIT_ELTWISE_MATH_OPT 0
#define TBC_TIME_OVERHEAD_EXP 0

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
using cluster_group_type = cooperative_groups::cluster_group;
#else
using cluster_group_type = int;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the warp group size from the SM architecture
// Tunable parameter, depennds on max number of SMs in a cluster supported by the device
// as well as the shared memory size available per SM
int get_tbc_warpgroup_size_from_sm_arch(int sm_arch) {
  // H200 and B200 support 16 SMs in a cluster
  // Other devices support 8 SMs in a cluster
  return (sm_arch == 900 || sm_arch == 1000)? 16 : 32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if TBC_TIME_OVERHEAD_EXP == 1
static inline __device__ void tbc_early_exit(
                                              volatile int* signal,
                                              cluster_group_type &cluster
                                              ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if( threadIdx.x == 0 && cluster.block_rank() == 0) {
    asm volatile("st.global.wt.s32 [%0], %1;" ::"l"(signal), "r"(0));
  }
  return;
#endif
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_LAYERS, int NUM_INPUTS, int NUM_UNITS, 
          int BLOCK_THREADS, int WARP_GROUP_SIZE, Data_kind DATA_KIND, bool STAGE = false, bool DEVICE_ALWAYS_SPIN = 0, bool GDRCOPY_ENABLED = 0>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void 
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__cluster_dims__(Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE, 1, 1) 
#endif
gemv_lstm_tbc_subwarp_regbuf_t_kernel(Lstm_bpt_params<NUM_LAYERS, NUM_INPUTS, NUM_UNITS> params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

  using data_type = Data_type<DATA_KIND>;
  using elem4_type = typename data_type::Elem4_type;
  using elem2_type = typename data_type::Elem2_type;
  using elem_type  = typename data_type::Elem_type;

  enum { WARP_SIZE = 32 };
  enum { NUM_WARPS = BLOCK_THREADS / WARP_SIZE };
  enum { NUM_GROUPS = BLOCK_THREADS / WARP_GROUP_SIZE };
  enum { NUM_SUBWARP_GROUPS = Div_up<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  enum { WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS };

  enum { NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE };
  // to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS
  enum { NUM_FEATURES_UP = Div_up<NUM_FEATURES, NUM_GROUPS>::VALUE * NUM_GROUPS };

  enum { SMEM_SIZE = WARP_GROUP_SIZE * WEIGHT_INPUTS * sizeof(elem4_type) 
                + 2 * NUM_FEATURES_UP * sizeof(elem_type) + BLOCK_THREADS * sizeof(elem4_type) };

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  static_assert(NUM_UNITS % NUM_WARPS == 0 && NUM_UNITS);
  static_assert(BLOCK_THREADS % WARP_GROUP_SIZE == 0);

  enum { CLUSTER_SIZE = Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE };

  extern __shared__ char smem_storage[]; 

  auto smem_weights = reinterpret_cast<elem4_type*>(smem_storage);

  auto shared_sum 
    = reinterpret_cast<elem4_type*>(smem_weights + WEIGHT_INPUTS * WARP_GROUP_SIZE);

  elem_type* smem_input[2];
  smem_input[0] = reinterpret_cast<elem_type*>(shared_sum + BLOCK_THREADS);
  smem_input[1] = smem_input[0] + NUM_FEATURES_UP;

  constexpr auto zero4 = data_type::zero4;
#if GEMM_H_VECTORIZED == 0
  constexpr elem_type zero = data_type::zero;
#endif


  auto gmem_input = reinterpret_cast<elem_type*>(params.gmem_input);
  auto gmem_h_out = reinterpret_cast<elem_type*>(params.gmem_h_out);
  auto gmem_dot_product_weights = reinterpret_cast<float*>(params.gmem_dot_prod_weights);

  elem4_type weights[NUM_FEATURES_UP / NUM_GROUPS];

  elem4_type bias[NUM_LAYERS];
  elem_type  c_in[NUM_LAYERS];
  elem4_type gemm_h[NUM_LAYERS];

  const int group_idx = threadIdx.x % WARP_GROUP_SIZE;
  const int group_id = threadIdx.x / WARP_GROUP_SIZE;

  namespace cg = cooperative_groups;

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_block_rank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  const int tid = cluster_block_rank * WARP_GROUP_SIZE + group_idx;

  const bool valid_unit = tid < NUM_UNITS;

  float dot_product_weight = valid_unit? gmem_dot_product_weights[tid] : 0.0f;

  auto gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[0]);

  enum { NUM_INPUTS_PER_GROUP = Div_up<NUM_INPUTS, NUM_GROUPS>::VALUE };
  enum { NUM_UNITS_PER_GROUP = Div_up<NUM_UNITS, NUM_GROUPS>::VALUE };
  
  // load first layer to registers
  #pragma unroll
  for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    weights[ii] = (valid_unit && (i < NUM_INPUTS) )?gmem_weights[i * NUM_UNITS + tid]: zero4;
  }
  

  if (valid_unit && threadIdx.x < WARP_GROUP_SIZE) {
    auto s_weights = smem_weights;
    
    // load the rest to smem
    for (int ll = 1; ll < NUM_LAYERS; ll++) {
      gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[ll]);
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
      auto gmem_biases = reinterpret_cast<elem4_type*>(params.gmem_biases[ll]);
      bias[ll] = gmem_biases[tid];
      if (!DEVICE_ALWAYS_SPIN) {
        auto gmem_c_in = reinterpret_cast<elem_type*>(params.gmem_c_in[ll]);
        c_in[ll] = gmem_c_in[tid];
  #if GEMM_H_VECTORIZED == 1
        auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h[ll]);
        gemm_h[ll] = gmem_gemm_h[tid];
  #else
        union Vec4_t{
          elem_type x[4];
          elem4_type x4;
        };
        Vec4_t val;
        auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ll]);
        val.x[0] = gmem_gemm_h[tid] ;
        val.x[1] = gmem_gemm_h[tid + 1 * NUM_UNITS];
        val.x[2] = gmem_gemm_h[tid + 2 * NUM_UNITS];
        val.x[3] = gmem_gemm_h[tid + 3 * NUM_UNITS];
        gemm_h[ll] = val.x4;
  #endif
      }
    }
  }

  gpu_cpu_sync_part_0(params.host_ld_complete, cluster);

  auto s_exit = reinterpret_cast<uint32_t *>(smem_input[1]);

  bool odd_step = false;
  #if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
  #endif
  do
  {
    *s_exit = 0;
    // reset weights' offset
    smem_weights = reinterpret_cast<elem4_type*>(smem_storage);

#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch, s_exit);
#else
    gpu_cpu_sync_part_1<cooperative_groups::cluster_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, s_exit);
#endif

  // shutdown the kernel
  if (DEVICE_ALWAYS_SPIN == 1) {
    if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE){
      if (cluster.thread_rank() == 0) {
        gpu_pingpong_handles::signal_atomic_relaxed<uint>((uint *)params.host_ld_complete,
                                                          HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
      }
      return;
    } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE){
      odd_step = false;
#if AS_STATES_LD_BAR == 1
      // workaround - we must continue with preprocess to signal states load after the initX/reset
#if DISABLE_ATOMIC_EXCHANGE == 1
      device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
      gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch);
#else
      gpu_cpu_sync_part_1<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster);
#endif
#else
      continue;
#endif
    } 

    // load states
    #pragma unroll
    for (int ii = 0; ii < NUM_LAYERS; ii++) {
      auto gmem_c_in = (odd_step)? reinterpret_cast<elem_type*>(params.gmem_c_out[ii]) 
                                 : reinterpret_cast<elem_type*>(params.gmem_c_in[ii]);
  #if GEMM_H_VECTORIZED == 1
      auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h[ii]);
      gemm_h[ii] = valid_unit?gmem_gemm_h[tid] : zero4;
  #else
      union Vec4_t{
        elem_type x[4];
        elem4_type x4;
      };
      Vec4_t val;
      auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ii]);
      val.x[0] = valid_unit?gmem_gemm_h[tid] : zero;
      val.x[1] = valid_unit?gmem_gemm_h[tid + 1 * NUM_UNITS] : zero;
      val.x[2] = valid_unit?gmem_gemm_h[tid + 2 * NUM_UNITS] : zero;
      val.x[3] = valid_unit?gmem_gemm_h[tid + 3 * NUM_UNITS] : zero;
      gemm_h[ii] = val.x4;
  #endif
      c_in[ii] = valid_unit?gmem_c_in[tid] : 0;
    }

#if AS_STATES_LD_BAR == 1
    // notify host that we are ready for compute
    gpu_cpu_sync_part_0(params.host_ld_complete, cluster);

    // // wait for host to signal starting the computation or exit (LSTM Forward on host)
#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch);
#else
    gpu_cpu_sync_part_1<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster);
#endif
#endif
  }

#if TBC_TIME_OVERHEAD_EXP == 1
  tbc_early_exit(params.host_ld_complete, cluster);
#endif

  // Broadcast input from CTA 0 to all CTAs in cluster
  static_assert(NUM_INPUTS % 2 == 0);
  if (cluster_block_rank == 0) {
    if (STAGE == false) {
      auto host_addr = reinterpret_cast<elem_type**>(params.host_addr);
      static_assert(BLOCK_THREADS * sizeof(elem4_type) >= sizeof(void *));
      auto smem_host_ptr = reinterpret_cast<elem_type**>(shared_sum);
      if (threadIdx.x == 0) {
        *smem_host_ptr = host_addr[0];
      }
      __syncthreads();
      #pragma unroll
      for (int idx = threadIdx.x; idx < NUM_INPUTS / 2; idx += BLOCK_THREADS){
        auto in = *(reinterpret_cast<elem2_type*>((*smem_host_ptr)) + idx);
        #pragma unroll
        for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
          auto dst_smem_input = cluster.map_shared_rank(reinterpret_cast<elem2_type*>(smem_input[0]), ii);
          dst_smem_input[idx] = in;
        }
      }
    } else { // STAGE
      #pragma unroll
      for (int idx = threadIdx.x; idx < NUM_INPUTS / 2; idx += BLOCK_THREADS){
        auto in = __ldcv(reinterpret_cast<elem2_type*>(gmem_input) + idx);
        #pragma unroll
        for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
          auto dst_smem_input = cluster.map_shared_rank(reinterpret_cast<elem2_type*>(smem_input[0]), ii);
          dst_smem_input[idx] = in;
        }
      }
    }
  }
  cluster.sync();

  auto acc = zero4;

  // first layer
  #pragma unroll
  for ( int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    acc = fma4<DATA_KIND>(weights[ii], smem_input[0][i], acc);
  }

  shared_sum[threadIdx.x] = acc;

  float hx;

  // need that before we can write to smem_input below
  __syncthreads();

  enum { MAX_WARP_SIZE_GROUP_SIZE = Max<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
    auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
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
    auto gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[0]);
    if (valid_unit) gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif

  }

  if (threadIdx.x < WARP_GROUP_SIZE) {
    if (valid_unit) {
      #pragma unroll
      for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
        auto dst_smem_input = cluster.map_shared_rank(smem_input[1], ii);
        dst_smem_input[tid] = from_float<DATA_KIND>(hx);
      }
    }
  }

  int ibuf = 1;
  #pragma unroll
  for (int ll = 1; ll < NUM_LAYERS; ll++) {

    acc = zero4;

    cluster.sync();

    #pragma unroll
    for ( int ii = 0; ii < NUM_UNITS_PER_GROUP; ii++) {
      int i = group_id * NUM_UNITS_PER_GROUP + ii;
      if (NUM_UNITS % NUM_GROUPS != 0)
        if (i >= NUM_UNITS) continue;
      acc = fma4<DATA_KIND>(smem_weights[i * WARP_GROUP_SIZE + group_idx], smem_input[ibuf][i], acc);
    }

    shared_sum[threadIdx.x] = acc;

    // need that before we can write to smem_input below
    __syncthreads();

    if (ll < NUM_LAYERS - 1)
    {
      smem_weights += NUM_UNITS * WARP_GROUP_SIZE;
    }

    if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
      auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
      #pragma unroll
      for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
        acc = add4<DATA_KIND>(acc, shared_sum[ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
      }

      if (WARP_GROUP_SIZE < WARP_SIZE) {
        // final reduction using schfl_down_sync
        for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
          acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
        }
        //__syncwarp();
      }
    }

    ibuf = (ibuf + 1) % 2;
    if (!valid_unit) continue;

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

#if DEBUG_OUTPUTS == 1
      auto gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[ll]);
      gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif

      if (ll < NUM_LAYERS-1) {
        // broadcast hidden state to all the SMs in the cluster
        if (valid_unit){
          #pragma unroll
          for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
            auto dst_smem_input = cluster.map_shared_rank(smem_input[ibuf], ii);
            dst_smem_input[tid] = from_float<DATA_KIND>(hx);
          }
        }
      } 
#if DEBUG_OUTPUTS == 1
      else {
        gmem_h_out[tid] = from_float<DATA_KIND>(hx);
      }
#endif
    }
  } // layers loop

  // Below we need to find the scallar product between dot_prod weights and hx
  enum { MIN_WARP_SIZE_GROUP_SIZE = Min<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };
  enum { NUM_WARPS_IN_GROUP = Div_up<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };

  float dot_product = 0.f;
  // inputs are not needed anymore
  auto smem_tmp = reinterpret_cast<float*>(smem_input[0]);

  // 1) Each block finds its partial sum

  if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {

    dot_product = (valid_unit && threadIdx.x < WARP_GROUP_SIZE)? dot_product_weight * hx : 0.0f;

    #pragma unroll
    for (int offset = MIN_WARP_SIZE_GROUP_SIZE / 2; offset > 0; offset /= 2) {
      dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
    }

    if (lane_id == 0) {
      smem_tmp[warp_id] = dot_product;
    }

    if (MAX_WARP_SIZE_GROUP_SIZE > WARP_SIZE)
      __syncthreads();
  }

  // 2)  Each block writes to block 0 its partial sum
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int ii = 1; ii < NUM_WARPS_IN_GROUP; ii++) {
      dot_product += smem_tmp[ii];
    }

    auto dst_smem_tmp = cluster.map_shared_rank(smem_tmp, 0);
    dst_smem_tmp[cluster_block_rank] = dot_product;
  }

  cluster.sync();

  // 3) Block 0 performs the final reduction
  if (cluster_block_rank == 0) {
    dot_product = (threadIdx.x < CLUSTER_SIZE)? smem_tmp[threadIdx.x] : 0.0f;

    // cluster size can not be lartger than 16
    if (threadIdx.x < WARP_SIZE) {
      #pragma unroll
      for (int offset = 16 / 2; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
      }
      dot_product += params.dot_prod_bias;
    }
  }

  gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, cluster);

  odd_step = !odd_step;
  } while(DEVICE_ALWAYS_SPIN);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// mbarrier-based TBC kernel (enabled with TBC_USE_MBARRIER)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Device function for FP16/BF16 shuffle-based packing - st_async does not support 2 bytes send
template <Data_kind DATA_KIND, int WARP_GROUP_SIZE>
__device__ __forceinline__ uint32_t pack_fp16_shuffle_sync(float hx) {
  using elem_type = typename Data_type<DATA_KIND>::Elem_type;
  constexpr uint32_t WARP_GROUP_MASK = 0xFFFFFFFF;

  uint16_t hx_u16 = 0;
  elem_type hx_elem = from_float<DATA_KIND>(hx);
  hx_u16 = *reinterpret_cast<uint16_t*>(&hx_elem);

  uint16_t hx_u16_even = __shfl_sync(WARP_GROUP_MASK, hx_u16, 2 * threadIdx.x);
  uint16_t hx_u16_odd  = __shfl_sync(WARP_GROUP_MASK, hx_u16, 2 * threadIdx.x + 1);
  uint32_t hx_u32 = static_cast<uint32_t>(hx_u16_even) | (static_cast<uint32_t>(hx_u16_odd) << 16);

  return hx_u32;
}

template <int NUM_LAYERS, int NUM_INPUTS, int NUM_UNITS,
          int BLOCK_THREADS, int WARP_GROUP_SIZE, Data_kind DATA_KIND, bool STAGE = false, bool DEVICE_ALWAYS_SPIN = 0, bool GDRCOPY_ENABLED = 0>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__cluster_dims__(Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE, 1, 1)
#endif
gemv_lstm_tbc_subwarp_regbuf_t_kernel_mbar(Lstm_bpt_params<NUM_LAYERS, NUM_INPUTS, NUM_UNITS> params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

  using data_type = Data_type<DATA_KIND>;
  using elem4_type = typename data_type::Elem4_type;
  using elem2_type = typename data_type::Elem2_type;
  using elem_type  = typename data_type::Elem_type;

  enum { WARP_SIZE = 32 };
  enum { NUM_WARPS = BLOCK_THREADS / WARP_SIZE };
  enum { NUM_GROUPS = BLOCK_THREADS / WARP_GROUP_SIZE };
  enum { NUM_SUBWARP_GROUPS = Div_up<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  enum { WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS };

  enum { NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE };
  // to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS
  enum { NUM_FEATURES_UP = Div_up<NUM_FEATURES, NUM_GROUPS>::VALUE * NUM_GROUPS };

  static_assert(NUM_UNITS % NUM_WARPS == 0 && NUM_UNITS);
  static_assert(BLOCK_THREADS % WARP_GROUP_SIZE == 0);

  enum { CLUSTER_SIZE = Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE };

  // For FP16 packing with mbarrier, need larger buffer
  enum { INPUT_BUF_COUNT = Div_up<Max<NUM_FEATURES, WARP_GROUP_SIZE * CLUSTER_SIZE>::VALUE, NUM_GROUPS>::VALUE * NUM_GROUPS };

  // Add 16 bytes for mbarriers + 2x BLOCK_THREADS for double-buffered shared_sum
  // Note: hx_buffer removed - now using warp shuffle for FP16 packing
  enum { SMEM_SIZE = 16 + WARP_GROUP_SIZE * WEIGHT_INPUTS * sizeof(elem4_type)
                + 2 * INPUT_BUF_COUNT * sizeof(elem_type) + 2 * BLOCK_THREADS * sizeof(elem4_type) };

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  extern __shared__ char smem_storage[];

  // Allocate mbarriers at start (8-byte aligned)
  uint64_t* mbar[2];
  mbar[0] = reinterpret_cast<uint64_t*>(smem_storage);
  mbar[1] = reinterpret_cast<uint64_t*>(smem_storage + 8);

  // Weights start after barriers (offset by 16 bytes)
  auto smem_weights = reinterpret_cast<elem4_type*>(smem_storage + 16);

  // Double-buffered shared_sum to avoid WAR hazards across layers
  auto shared_sum_base
    = reinterpret_cast<elem4_type*>(smem_weights + WEIGHT_INPUTS * WARP_GROUP_SIZE);

  elem4_type* shared_sum[2];
  shared_sum[0] = shared_sum_base;
  shared_sum[1] = shared_sum_base + BLOCK_THREADS;

  elem_type* smem_input[2];
  smem_input[0] = reinterpret_cast<elem_type*>(shared_sum_base + 2 * BLOCK_THREADS);
  smem_input[1] = smem_input[0] + INPUT_BUF_COUNT;

  constexpr auto zero4 = data_type::zero4;
#if GEMM_H_VECTORIZED == 0
  constexpr elem_type zero = data_type::zero;
#endif


  auto gmem_input = reinterpret_cast<elem_type*>(params.gmem_input);
  auto gmem_h_out = reinterpret_cast<elem_type*>(params.gmem_h_out);
  auto gmem_dot_product_weights = reinterpret_cast<float*>(params.gmem_dot_prod_weights);

  elem4_type weights[NUM_FEATURES_UP / NUM_GROUPS];

  elem4_type bias[NUM_LAYERS];
  elem_type  c_in[NUM_LAYERS];
  elem4_type gemm_h[NUM_LAYERS];

  const int group_idx = threadIdx.x % WARP_GROUP_SIZE;
  const int group_id = threadIdx.x / WARP_GROUP_SIZE;

  namespace cg = cooperative_groups;

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  int cluster_block_rank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  const int tid = cluster_block_rank * WARP_GROUP_SIZE + group_idx;

  const bool valid_unit = tid < NUM_UNITS;

  float dot_product_weight = valid_unit? gmem_dot_product_weights[tid] : 0.0f;

  auto gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[0]);

  enum { NUM_INPUTS_PER_GROUP = Div_up<NUM_INPUTS, NUM_GROUPS>::VALUE };
  enum { NUM_UNITS_PER_GROUP = Div_up<NUM_UNITS, NUM_GROUPS>::VALUE };

  // load first layer to registers
  #pragma unroll
  for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    weights[ii] = (valid_unit && (i < NUM_INPUTS) )?gmem_weights[i * NUM_UNITS + tid]: zero4;
  }


  if (valid_unit && threadIdx.x < WARP_GROUP_SIZE) {
    auto s_weights = smem_weights;

    // load the rest to smem
    for (int ll = 1; ll < NUM_LAYERS; ll++) {
      gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[ll]);
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
      auto gmem_biases = reinterpret_cast<elem4_type*>(params.gmem_biases[ll]);
      bias[ll] = gmem_biases[tid];
      if (!DEVICE_ALWAYS_SPIN) {
        auto gmem_c_in = reinterpret_cast<elem_type*>(params.gmem_c_in[ll]);
        c_in[ll] = gmem_c_in[tid];
  #if GEMM_H_VECTORIZED == 1
        auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h[ll]);
        gemm_h[ll] = gmem_gemm_h[tid];
  #else
        union Vec4_t{
          elem_type x[4];
          elem4_type x4;
        };
        Vec4_t val;
        auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ll]);
        val.x[0] = gmem_gemm_h[tid] ;
        val.x[1] = gmem_gemm_h[tid + 1 * NUM_UNITS];
        val.x[2] = gmem_gemm_h[tid + 2 * NUM_UNITS];
        val.x[3] = gmem_gemm_h[tid + 3 * NUM_UNITS];
        gemm_h[ll] = val.x4;
  #endif
      }
    }
  }

  // Phase tracking for barrier reuse
  int phase[2] = {0, 0};

  gpu_cpu_sync_part_0(params.host_ld_complete, cluster);

  auto s_exit = reinterpret_cast<uint32_t *>(smem_input[1]);

  bool odd_step = false;
  #if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
  #endif
  do
  {
    *s_exit = 0;
    // reset weights' offset
    smem_weights = reinterpret_cast<elem4_type*>(smem_storage + 16 + WARP_SIZE * sizeof(elem_type));  // Offset by mbarriers + hx_buffer

    // Reinitialize mbarriers for each inference iteration (without sync - not needed)
    if (threadIdx.x == 0) {
      cuda::ptx::mbarrier_init(mbar[0], 1);
      cuda::ptx::mbarrier_init(mbar[1], 1);
      cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
    }

    // Reset phase tracking for each inference iteration
    phase[0] = 0;
    phase[1] = 0;

#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch, s_exit);
#else
    gpu_cpu_sync_part_1<cooperative_groups::cluster_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, s_exit);
#endif

  // shutdown the kernel
  if (DEVICE_ALWAYS_SPIN == 1) {
    if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE){
      if (cluster.thread_rank() == 0) {
        gpu_pingpong_handles::signal_atomic_relaxed<uint>((uint *)params.host_ld_complete,
                                                          HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
      }
      return;
    } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE){
      odd_step = false;
#if AS_STATES_LD_BAR == 1
      // workaround - we must continue with preprocess to signal states load after the initX/reset
#if DISABLE_ATOMIC_EXCHANGE == 1
      device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
      gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch);
#else
      gpu_cpu_sync_part_1<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster);
#endif
#else
      continue;
#endif
    }

    // load states
    #pragma unroll
    for (int ii = 0; ii < NUM_LAYERS; ii++) {
      auto gmem_c_in = (odd_step)? reinterpret_cast<elem_type*>(params.gmem_c_out[ii])
                                 : reinterpret_cast<elem_type*>(params.gmem_c_in[ii]);
  #if GEMM_H_VECTORIZED == 1
      auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h[ii]);
      gemm_h[ii] = valid_unit?gmem_gemm_h[tid] : zero4;
  #else
      union Vec4_t{
        elem_type x[4];
        elem4_type x4;
      };
      Vec4_t val;
      auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ii]);
      val.x[0] = valid_unit?gmem_gemm_h[tid] : zero;
      val.x[1] = valid_unit?gmem_gemm_h[tid + 1 * NUM_UNITS] : zero;
      val.x[2] = valid_unit?gmem_gemm_h[tid + 2 * NUM_UNITS] : zero;
      val.x[3] = valid_unit?gmem_gemm_h[tid + 3 * NUM_UNITS] : zero;
      gemm_h[ii] = val.x4;
  #endif
      c_in[ii] = valid_unit?gmem_c_in[tid] : 0;
    }

#if AS_STATES_LD_BAR == 1
    // notify host that we are ready for compute
    gpu_cpu_sync_part_0(params.host_ld_complete, cluster);

    // // wait for host to signal starting the computation or exit (LSTM Forward on host)
#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster, &device_spin_switch);
#else
    gpu_cpu_sync_part_1<cooperative_groups::cluster_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, cluster);
#endif
#endif
  }

#if TBC_TIME_OVERHEAD_EXP == 1
  tbc_early_exit(params.host_ld_complete, cluster);
#endif

  // Broadcast input from CTA 0 to all CTAs in cluster using mbarrier
  static_assert(NUM_INPUTS % 2 == 0);
  if (cluster_block_rank == 0) {
    if (STAGE == false) {
      auto host_addr = reinterpret_cast<elem_type**>(params.host_addr);
      static_assert(BLOCK_THREADS * sizeof(elem4_type) >= sizeof(void *));
      auto smem_host_ptr = reinterpret_cast<elem_type**>(shared_sum[0]);
      if (threadIdx.x == 0) {
        *smem_host_ptr = host_addr[0];
      }
      __syncthreads();

      // Broadcast using st_async with elem2_type
      auto input_elem2 = reinterpret_cast<elem2_type*>(*smem_host_ptr);
      #pragma unroll
      for (int idx = threadIdx.x; idx < NUM_INPUTS / 2; idx += BLOCK_THREADS){
        elem2_type in = input_elem2[idx];
        #pragma unroll
        for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
          elem2_type* dst_smem_input = reinterpret_cast<elem2_type*>(
              cluster.map_shared_rank(smem_input[0], ii));
          uint64_t* peer_mbar = cluster.map_shared_rank(mbar[0], ii);
          cuda::ptx::st_async(&dst_smem_input[idx], in, peer_mbar);
        }
      }
    } else { // STAGE
      auto gmem_input_elem2 = reinterpret_cast<elem2_type*>(gmem_input);
      #pragma unroll
      for (int idx = threadIdx.x; idx < NUM_INPUTS / 2; idx += BLOCK_THREADS){
        elem2_type in = __ldcv(gmem_input_elem2 + idx);
        #pragma unroll
        for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
          elem2_type* dst_smem_input = reinterpret_cast<elem2_type*>(
              cluster.map_shared_rank(smem_input[0], ii));
          uint64_t* peer_mbar = cluster.map_shared_rank(mbar[0], ii);
          cuda::ptx::st_async(&dst_smem_input[idx], in, peer_mbar);
        }
      }
    }
  }

  // All blocks announce expected transaction count (consumers receive, producer also "receives" from itself)
  if (threadIdx.x == 0) {
    const int tx_count = NUM_INPUTS * sizeof(elem_type);
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                          cuda::ptx::scope_cta,
                                          cuda::ptx::space_shared,
                                          mbar[0], tx_count);
  }

  // All blocks wait for input broadcast to complete
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                                               cuda::ptx::scope_cluster,
                                               mbar[0], phase[0])) {}
  phase[0] ^= 1;  // Flip phase for next iteration

  auto acc = zero4;

  // first layer
  #pragma unroll
  for ( int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    acc = fma4<DATA_KIND>(weights[ii], smem_input[0][i], acc);
  }

  shared_sum[0][threadIdx.x] = acc;

  float hx;

  // need that before we can write to smem_input below
  __syncthreads();

  enum { MAX_WARP_SIZE_GROUP_SIZE = Max<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
    auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
      acc = add4<DATA_KIND>(acc, shared_sum[0][ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
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
    auto gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[0]);
    if (valid_unit) gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif

  }

  // Layer 0: PRODUCER broadcasts to all CTAs using st_async
  if (DATA_KIND != DATA_TYPE_FP32) {
    // FP16/BF16: Pack using shuffle instead of shared memory
    constexpr int WRITE_THREADS = WARP_GROUP_SIZE / 2;

    // ALL threads < WARP_SIZE must participate in shuffle (shuffle operates at warp level)
    if (threadIdx.x < WARP_SIZE) {
      uint32_t hx_u32 = pack_fp16_shuffle_sync<DATA_KIND, WARP_GROUP_SIZE>(hx);

      // Only first WRITE_THREADS perform packing and st_async
      if (threadIdx.x < WRITE_THREADS) {
        #pragma unroll
        for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
          uint32_t* peer_vector_u32 = reinterpret_cast<uint32_t*>(
              cluster.map_shared_rank(smem_input[1], ii));
          uint64_t* peer_mbar = cluster.map_shared_rank(mbar[1], ii);
          cuda::ptx::st_async(&peer_vector_u32[cluster_block_rank * WRITE_THREADS + threadIdx.x], hx_u32, peer_mbar);
        }
      }
    }
  } else {
    // FP32: Direct write with float (4 bytes), no buffer needed
    if (threadIdx.x < WARP_GROUP_SIZE && valid_unit) {
      float hx_f32 = hx;
      #pragma unroll
      for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
        float* peer_vector = cluster.map_shared_rank(reinterpret_cast<float*>(smem_input[1]), ii);
        uint64_t* peer_mbar = cluster.map_shared_rank(mbar[1], ii);
        cuda::ptx::st_async(&peer_vector[tid], hx_f32, peer_mbar);
      }
    }
  }

  // Producer announces expected bytes (sets up barrier for layer 1 to wait on)
  if (threadIdx.x == 0) {
    const int tx_count = NUM_UNITS * sizeof(elem_type);
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                          cuda::ptx::scope_cta,
                                          cuda::ptx::space_shared,
                                          mbar[1], tx_count);
  }

  int ibuf = 1;
  #pragma unroll
  for (int ll = 1; ll < NUM_LAYERS; ll++) {

    acc = zero4;

    // CONSUMER: Wait for previous layer's writes at BEGINNING (replaces cluster.sync)
    while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                                                 cuda::ptx::scope_cluster,
                                                 mbar[ibuf], phase[ibuf])) {}
    phase[ibuf] ^= 1;  // Flip phase after wait

    #pragma unroll
    for ( int ii = 0; ii < NUM_UNITS_PER_GROUP; ii++) {
      int i = group_id * NUM_UNITS_PER_GROUP + ii;
      if (NUM_UNITS % NUM_GROUPS != 0)
        if (i >= NUM_UNITS) continue;
      acc = fma4<DATA_KIND>(smem_weights[i * WARP_GROUP_SIZE + group_idx], smem_input[ibuf][i], acc);
    }

    shared_sum[ibuf][threadIdx.x] = acc;

    // need that before we can write to smem_input below
    __syncthreads();

    if (ll < NUM_LAYERS - 1)
    {
      smem_weights += NUM_UNITS * WARP_GROUP_SIZE;
    }

    if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {
      auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
      #pragma unroll
      for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
        acc = add4<DATA_KIND>(acc, shared_sum[ibuf][ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
      }

      if (WARP_GROUP_SIZE < WARP_SIZE) {
        // final reduction using schfl_down_sync
        for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
          acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
        }
        //__syncwarp();
      }
    }

    ibuf = (ibuf + 1) % 2;
    if (!valid_unit) continue;

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

#if DEBUG_OUTPUTS == 1
      auto gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[ll]);
      gmem_c_out[tid] = from_float<DATA_KIND>(cx);
#endif
    }

    // PRODUCER: Broadcast hidden states using st_async at END
    if (ll < NUM_LAYERS-1) {
      if (DATA_KIND != DATA_TYPE_FP32) {
        // FP16/BF16: Pack using shuffle instead of shared memory
        constexpr int WRITE_THREADS = WARP_GROUP_SIZE / 2;

        // ALL threads < WARP_SIZE must participate in shuffle (shuffle operates at warp level)
        if (threadIdx.x < WARP_SIZE) {
          uint32_t hx_u32 = pack_fp16_shuffle_sync<DATA_KIND, WARP_GROUP_SIZE>(hx);

          // Only first WRITE_THREADS perform packing and st_async
          if (threadIdx.x < WRITE_THREADS) {
            #pragma unroll
            for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
              uint32_t* peer_vector_u32 = reinterpret_cast<uint32_t*>(
                  cluster.map_shared_rank(smem_input[ibuf], ii));
              uint64_t* peer_mbar = cluster.map_shared_rank(mbar[ibuf], ii);
              cuda::ptx::st_async(&peer_vector_u32[cluster_block_rank * WRITE_THREADS + threadIdx.x], hx_u32, peer_mbar);
            }
          }
        }
      } else {
        // FP32: Direct write with float (4 bytes), no buffer needed
        if (threadIdx.x < WARP_GROUP_SIZE && valid_unit) {
          float hx_f32 = hx;
          #pragma unroll
          for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
            float* peer_vector = cluster.map_shared_rank(reinterpret_cast<float*>(smem_input[ibuf]), ii);
            uint64_t* peer_mbar = cluster.map_shared_rank(mbar[ibuf], ii);
            cuda::ptx::st_async(&peer_vector[tid], hx_f32, peer_mbar);
          }
        }
      }

      // Producer announces expected transactions after writing
      if (threadIdx.x == 0) {
        const int tx_count = NUM_UNITS * sizeof(elem_type);
        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                              cuda::ptx::scope_cta,
                                              cuda::ptx::space_shared,
                                              mbar[ibuf], tx_count);
      }
    }
#if DEBUG_OUTPUTS == 1
    else {
      gmem_h_out[tid] = from_float<DATA_KIND>(hx);
    }
#endif
  } // layers loop

  // Below we need to find the scallar product between dot_prod weights and hx
  enum { MIN_WARP_SIZE_GROUP_SIZE = Min<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };
  enum { NUM_WARPS_IN_GROUP = Div_up<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };

  float dot_product = 0.f;
  // inputs are not needed anymore
  auto smem_tmp = reinterpret_cast<float*>(smem_input[0]);

  // 1) Each block finds its partial sum

  if (threadIdx.x < MAX_WARP_SIZE_GROUP_SIZE) {

    dot_product = (valid_unit && threadIdx.x < WARP_GROUP_SIZE)? dot_product_weight * hx : 0.0f;

    #pragma unroll
    for (int offset = MIN_WARP_SIZE_GROUP_SIZE / 2; offset > 0; offset /= 2) {
      dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
    }

    if (lane_id == 0) {
      smem_tmp[warp_id] = dot_product;
    }

    if (MAX_WARP_SIZE_GROUP_SIZE > WARP_SIZE)
      __syncthreads();
  }

  // 2)  Each block writes to block 0 its partial sum using st_async
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int ii = 1; ii < NUM_WARPS_IN_GROUP; ii++) {
      dot_product += smem_tmp[ii];
    }

    // Use st_async to write partial sum to block 0 (reuse mbar[ibuf])
    float* dst_smem_tmp = cluster.map_shared_rank(reinterpret_cast<float*>(smem_tmp), 0);
    uint64_t* peer_mbar = cluster.map_shared_rank(mbar[ibuf], 0);
    cuda::ptx::st_async(&dst_smem_tmp[cluster_block_rank], dot_product, peer_mbar);
  }

  // Block 0 announces expected transactions (CLUSTER_SIZE floats × 4 bytes each)
  if (cluster_block_rank == 0 && threadIdx.x == 0) {
    const int tx_count = CLUSTER_SIZE * sizeof(float);
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                          cuda::ptx::scope_cta,
                                          cuda::ptx::space_shared,
                                          mbar[ibuf], tx_count);
  }

  // Block 0 waits for all partial sums to arrive
  if (cluster_block_rank == 0) {
    while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                                                 cuda::ptx::scope_cluster,
                                                 mbar[ibuf], phase[ibuf])) {}
    // No phase flip needed - barriers and phases are reset at start of next iteration
  }

  // 3) Block 0 performs the final reduction
  if (cluster_block_rank == 0) {
    dot_product = (threadIdx.x < CLUSTER_SIZE)? smem_tmp[threadIdx.x] : 0.0f;

    // cluster size can not be lartger than 16
    if (threadIdx.x < WARP_SIZE) {
      #pragma unroll
      for (int offset = 16 / 2; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
      }
      dot_product += params.dot_prod_bias;
    }
  }

  gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, cluster);

  odd_step = !odd_step;
  } while(DEVICE_ALWAYS_SPIN);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_lstm_s_tbc_t(Lstm_s_tbc_t_params& params, Data_kind dtype, cudaStream_t stream) {
  throw std::runtime_error("Not Implemented");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define LSTM_M_CONSTANTS(DATA_TYPE, Arch_traits) \
  constexpr int NUM_UNITS  = Lstm_m_tbc_params::NUM_UNITS; \
  constexpr int NUM_INPUTS = Lstm_m_tbc_params::NUM_INPUTS; \
  constexpr int NUM_LAYERS = Lstm_m_tbc_params::NUM_LAYERS; \
 \
  constexpr int WARP_SIZE = 32;  \
 \
  constexpr int NUM_WARPS = Arch_traits.TBC_LSTM_M_NUM_WARPS_PER_BLOCK;  \
  constexpr int WARP_GROUP_SIZE = Arch_traits.TBC_LSTM_M_WARP_GROUP_SIZE; \
 \
  constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;  \
 \
  constexpr int CLUSTER_SIZE = Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE;  \
  constexpr int NUM_GROUPS = Div_up<BLOCK_THREADS, WARP_GROUP_SIZE>::VALUE;  \
 \
  constexpr int WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS; \
 \
  constexpr int NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE; \
  /* to handle case when NUM_FEATRUES is not divisible by NUM_GROUPS */ \
  constexpr int NUM_FEATURES_UP = Div_up<NUM_FEATURES, NUM_GROUPS>::VALUE * NUM_GROUPS; \
  \
  /* For mbarrier with FP16 packing, need larger buffer for WARP_GROUP_SIZE * CLUSTER_SIZE */ \
  constexpr int INPUT_BUF_COUNT = Div_up<Max<NUM_FEATURES, WARP_GROUP_SIZE * CLUSTER_SIZE>::VALUE, NUM_GROUPS>::VALUE * NUM_GROUPS; \
  \
  static_assert(CLUSTER_SIZE <= 16); \
  \
  constexpr int ELEM_SIZE = (DATA_TYPE == DATA_TYPE_FP32)? sizeof(float) : sizeof(uint16_t); \
  \
  /* Calculate SMEM_SIZE conditionally based on mbarrier usage */ \
  LSTM_M_SMEM_SIZE_CALC

// Define SMEM_SIZE calculation macro (used inside LSTM_M_CONSTANTS)
#ifdef TBC_USE_MBARRIER
#define LSTM_M_SMEM_SIZE_CALC \
  constexpr int SMEM_SIZE = 16 + WARP_GROUP_SIZE * WEIGHT_INPUTS * ELEM_SIZE * 4 \
                  + 2 * INPUT_BUF_COUNT * ELEM_SIZE + 2 * BLOCK_THREADS * ELEM_SIZE * 4;
#define LSTM_M_KERNEL_NAME gemv_lstm_tbc_subwarp_regbuf_t_kernel_mbar
#else
#define LSTM_M_SMEM_SIZE_CALC \
  constexpr int SMEM_SIZE = WARP_GROUP_SIZE * WEIGHT_INPUTS * ELEM_SIZE * 4 \
                  + 2 * NUM_FEATURES_UP * ELEM_SIZE + BLOCK_THREADS * ELEM_SIZE * 4;
#define LSTM_M_KERNEL_NAME gemv_lstm_tbc_subwarp_regbuf_t_kernel
#endif

template <Arch Arch_traits, Data_kind DATA_TYPE>
bool verify_lstm_m_occupancy_reqs_(int num_instances) {

  constexpr bool STAGE = true;
  constexpr bool DEVICE_ALWAYS_SPIN = true;
  constexpr bool GDRCOPY_ENABLED = false;

  LSTM_M_CONSTANTS(DATA_TYPE, Arch_traits)

  int max_cluster_size  = 0;
  cudaLaunchConfig_t config = {0};
  config.gridDim = CLUSTER_SIZE;
  config.blockDim = BLOCK_THREADS;
  config.dynamicSmemBytes = SMEM_SIZE;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = CLUSTER_SIZE;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute; 
  config.numAttrs = 1;

  if (SMEM_SIZE > 0)
    CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      BLOCK_THREADS, WARP_GROUP_SIZE,
                                      DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  if (CLUSTER_SIZE > 8)
    CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      BLOCK_THREADS, WARP_GROUP_SIZE,
                                      DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                        BLOCK_THREADS, WARP_GROUP_SIZE,
                                        DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                                        &config));
  if (max_cluster_size < CLUSTER_SIZE) { fprintf(stderr, "Desired cluster size is not supported: %d < %d\n", max_cluster_size, CLUSTER_SIZE);}

  int max_active_clusters = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveClusters(&max_active_clusters, (void *)LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                        BLOCK_THREADS, WARP_GROUP_SIZE,
                                        DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>, &config));

  if (max_active_clusters < num_instances) { fprintf(stderr, "%d active clusters of size %d can not be supported on this device\n", num_instances, CLUSTER_SIZE);}

  return (max_cluster_size >= CLUSTER_SIZE && max_active_clusters >= num_instances);
}

bool verify_lstm_m_occupancy_reqs(int num_instances, int sm_arch, int elem_size) {
  if (is_sm_heavy_arch(sm_arch)) {
    if (elem_size == 2) {
      return verify_lstm_m_occupancy_reqs_<SM_HEAVY_ARCH, DATA_TYPE_FP16>(num_instances);
    } else {
      return verify_lstm_m_occupancy_reqs_<SM_HEAVY_ARCH, DATA_TYPE_FP32>(num_instances);
    }
  } else {
    if (elem_size == 2) {
      return verify_lstm_m_occupancy_reqs_<SM_LIGHT_ARCH, DATA_TYPE_FP16>(num_instances);
    } else {
      return verify_lstm_m_occupancy_reqs_<SM_LIGHT_ARCH, DATA_TYPE_FP32>(num_instances);
    }
  }
}

template <Arch Arch_traits, Data_kind DATA_TYPE>
bool verify_lstm_m_cluster_size_supported_() {
  constexpr bool STAGE = true;
  constexpr bool DEVICE_ALWAYS_SPIN = true;
  constexpr bool GDRCOPY_ENABLED = false;

  LSTM_M_CONSTANTS(DATA_TYPE, Arch_traits)

  int max_cluster_size  = 0;
  cudaLaunchConfig_t config = {0};
  config.gridDim = CLUSTER_SIZE;
  config.blockDim = BLOCK_THREADS;
  config.dynamicSmemBytes = SMEM_SIZE;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = CLUSTER_SIZE;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute; 
  config.numAttrs = 1;

  if (SMEM_SIZE > 0)
    CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      BLOCK_THREADS, WARP_GROUP_SIZE,
                                      DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  if (CLUSTER_SIZE > 8)
    CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      BLOCK_THREADS, WARP_GROUP_SIZE,
                                      DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                        BLOCK_THREADS, WARP_GROUP_SIZE,
                                        DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                                        &config));
  if (max_cluster_size < CLUSTER_SIZE) { fprintf(stderr, "Desired cluster size is not supported: %d < %d\n", max_cluster_size, CLUSTER_SIZE);}

  return (max_cluster_size >= CLUSTER_SIZE);
}

bool verify_lstm_m_cluster_size_supported(int sm_arch, int elem_size) {
  if (is_sm_heavy_arch(sm_arch)) {
    if (elem_size == 2) {
      return verify_lstm_m_cluster_size_supported_<SM_HEAVY_ARCH, DATA_TYPE_FP16>();
    } else {
      return verify_lstm_m_cluster_size_supported_<SM_HEAVY_ARCH, DATA_TYPE_FP32>();
    }
  } else {
    if (elem_size == 2) {
      return verify_lstm_m_cluster_size_supported_<SM_LIGHT_ARCH, DATA_TYPE_FP16>();
    } else {
      return verify_lstm_m_cluster_size_supported_<SM_LIGHT_ARCH, DATA_TYPE_FP32>();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<Arch Arch_traits, Data_kind DATA_TYPE, bool STAGE, bool DEVICE_ALWAYS_SPIN, bool GDRCOPY_ENABLED>
void launch_lstm_m_tbc_t(Lstm_m_tbc_t_params& params, cudaStream_t stream) {

  LSTM_M_CONSTANTS(DATA_TYPE, Arch_traits)

  if (SMEM_SIZE > 0) 
  CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                    BLOCK_THREADS, WARP_GROUP_SIZE,
                                    DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  if (CLUSTER_SIZE > 8) \
    CHECK_CUDA( cudaFuncSetAttribute(LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      BLOCK_THREADS, WARP_GROUP_SIZE, \
                                      DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  { 
    int max_cluster_size  = 0;
    cudaLaunchConfig_t config = {0};
    config.gridDim = CLUSTER_SIZE;
    config.blockDim = BLOCK_THREADS;
    config.dynamicSmemBytes = SMEM_SIZE;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                          BLOCK_THREADS, WARP_GROUP_SIZE,
                                          DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                                          &config);
    if (max_cluster_size < CLUSTER_SIZE) { fprintf(stderr, "Desired cluster size is not supported: %d < %d\n", max_cluster_size, CLUSTER_SIZE); exit(1);}
  }
  void *kernel_args[] = { &params };
  cudaLaunchCooperativeKernel(
    (const void *)LSTM_M_KERNEL_NAME<NUM_LAYERS, NUM_INPUTS, NUM_UNITS, BLOCK_THREADS, WARP_GROUP_SIZE, DATA_TYPE, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
    dim3(CLUSTER_SIZE, 1, 1),
    dim3(BLOCK_THREADS, 1, 1),
    kernel_args,
    SMEM_SIZE,
    stream);
}

void run_lstm_m_tbc_t(Lstm_m_tbc_t_params& params, Data_kind dtype, cudaStream_t stream) {
  kernel_dispatch::dispatch_kernel_variants<true, true>(
    dtype,                                   // dtype
    params.host_addr == nullptr,             // stage
    params.always_spin != 0,                 // device_always_spin
    params.use_gdrcopy != 0,                 // gdrcopy_enabled
    params.sm_arch,                          // sm_arch
    [&]<Arch A, Data_kind DT, bool S, bool P, bool G>() {
      launch_lstm_m_tbc_t<A, DT, S, P, G>(params, stream);
    }
  );
  CHECK_CUDA(cudaGetLastError());
}

} // namespace ssgemv