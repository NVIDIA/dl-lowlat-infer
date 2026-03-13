/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <vector>
#include <stdint.h>

#include <cooperative_groups.h>
#include "gemv_lstm.h"
#include "kernel_params.h"
#include "gemv_lstm_utils.cuh"

#include "dispatch_utils.cuh"

namespace nvidia::quant::lowlat_infer::ssgemv {

#define BP_TIME_OVERHEAD_EXP 0
#define BP_16BIT_ELTWISE_MATH_OPT 0

////////////////////////////////////////////////////////////////////////////////////////////////////

#if BP_TIME_OVERHEAD_EXP == 1
static inline __device__ void early_exit(volatile int* signal) {
  if (threadIdx.x == 0) {
    asm volatile("st.global.wt.s32 [%0], %1;" ::"l"(signal), "r"(1));
  }
  return;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void read_only_kernel(const size_t n, const int flag, int* data) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  int sum = 0;
  for (size_t i = idx; i < n; i += stride) {
    sum += data[i];
  }

  if (sum * flag == 13) {
    data[idx] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// FP32/FP16/BP16 implementation
template <int NUM_LAYERS, int NUM_INPUTS, int NUM_UNITS, 
          int WARP_GROUP_SIZE, int BLOCK_THREADS, Data_kind DATA_KIND, bool STAGE = false, 
          bool DEVICE_ALWAYS_SPIN = 0, bool GDRCOPY_ENABLED = 0>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void gemv_lstm_s_bpt_kernel(Lstm_bpt_params<NUM_LAYERS, NUM_INPUTS, NUM_UNITS> params) {

  static_assert(NUM_LAYERS == 2);

  using data_type = Data_type<DATA_KIND>;
  using elem4_type = typename data_type::Elem4_type;
  using elem_type  = typename data_type::Elem_type;

  auto block = cooperative_groups::this_thread_block();

  enum { WARP_SIZE  = 32 };
  enum { NUM_WARPS  = BLOCK_THREADS / WARP_SIZE };

  enum { NUM_SUBWARP_GROUPS = Div_up<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  enum { CLUSTER_SIZE_DOWN = Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE };
  enum { CLUSTER_SIZE = Div_up<CLUSTER_SIZE_DOWN * WARP_GROUP_SIZE, WARP_SIZE>::VALUE * NUM_SUBWARP_GROUPS };

  enum { NUM_GROUPS = NUM_WARPS * NUM_SUBWARP_GROUPS / CLUSTER_SIZE }; // numm warp groups per acu

  enum { NUM_INPUTS_PER_GROUP = Div_up<NUM_INPUTS, NUM_GROUPS>::VALUE };
  enum { NUM_UNITS_PER_GROUP = Div_up<NUM_UNITS, NUM_GROUPS>::VALUE };
  enum { NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE };
  enum { NUM_FEATURES_UP = Div_up<NUM_FEATURES, NUM_GROUPS>::VALUE * NUM_GROUPS };
  enum { NUM_INPUTS_UP = Div_up<NUM_INPUTS, NUM_GROUPS>::VALUE * NUM_GROUPS };

  enum { WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS };

  enum { SMEM_SIZE = WARP_GROUP_SIZE * CLUSTER_SIZE * WEIGHT_INPUTS * sizeof(elem4_type) 
                 + BLOCK_THREADS * sizeof(elem4_type) + 2 * NUM_FEATURES_UP * sizeof(elem_type)
                 + NUM_WARPS * sizeof(float)};

  // Shared memory buffer to store weights and input.
  extern __shared__ char smem_storage[];

  auto smem_weights = reinterpret_cast<elem4_type*>(smem_storage);

  auto shared_sum 
    = reinterpret_cast<elem4_type*>(smem_weights + WEIGHT_INPUTS * WARP_GROUP_SIZE * CLUSTER_SIZE);
  auto smem_input 
    = reinterpret_cast<elem_type*>(shared_sum + BLOCK_THREADS);

  auto gmem_input = (STAGE)?reinterpret_cast<elem_type*>(params.gmem_input) : nullptr;

  auto gmem_h_out = reinterpret_cast<elem_type*>(params.gmem_h_out);

  auto gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[0]);
  auto gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[0]);

  auto gmem_dot_product_weights = reinterpret_cast<float*>(params.gmem_dot_prod_weights);

  float dot_product_weight;

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  constexpr elem4_type zero4 = data_type::zero4;
#if GEMM_H_VECTORIZED == 0
  constexpr elem_type zero = data_type::zero;
#endif

  elem4_type bias[NUM_LAYERS];
  elem_type c_in[NUM_LAYERS];
  elem4_type gemm_h[NUM_LAYERS];

  const int group_idx = threadIdx.x % WARP_GROUP_SIZE;
  const int cluster_id = threadIdx.x / (WARP_GROUP_SIZE * NUM_GROUPS);
  const int group_id = (threadIdx.x / WARP_GROUP_SIZE) % NUM_GROUPS;
  const int cluster_idx = threadIdx.x % (WARP_GROUP_SIZE * NUM_GROUPS);

  const int tid = cluster_id * WARP_GROUP_SIZE + group_idx;

  const bool p = tid < NUM_UNITS;

  elem4_type weights[NUM_INPUTS_UP / NUM_GROUPS];

  // offset smem_weights
  smem_weights += cluster_id * WARP_GROUP_SIZE * NUM_UNITS;

  #pragma unroll
  for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    weights[ii] = (p && (i < NUM_INPUTS) )?gmem_weights[i * NUM_UNITS + tid]: zero4;
  }

  if (p) {
    gmem_weights = reinterpret_cast<elem4_type*>(params.gmem_weights[1]);
    #pragma unroll
    for (int ii = 0; ii < NUM_UNITS; ii++) {
      smem_weights[ii * WARP_GROUP_SIZE + group_idx] = gmem_weights[ii * NUM_UNITS + tid];
    }
  }

  #pragma unroll
  for (int ii = 0; ii < NUM_LAYERS; ii++) {
    auto gmem_biases = reinterpret_cast<elem4_type*>(params.gmem_biases[ii]);
    if (!DEVICE_ALWAYS_SPIN) {
      auto gmem_c_in = reinterpret_cast<elem_type*>(params.gmem_c_in[ii]);
  #if GEMM_H_VECTORIZED == 1
      auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h[ii]);
      gemm_h[ii] = p?gmem_gemm_h[tid] : zero4;
  #else
      union Vec4_t{
        elem_type x[4];
        elem4_type x4;
      };
      Vec4_t val;
      auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ii]);
      val.x[0] = p?gmem_gemm_h[tid] : zero;
      val.x[1] = p?gmem_gemm_h[tid + 1 * NUM_UNITS] : zero;
      val.x[2] = p?gmem_gemm_h[tid + 2 * NUM_UNITS] : zero;
      val.x[3] = p?gmem_gemm_h[tid + 3 * NUM_UNITS] : zero;
      gemm_h[ii] = val.x4;
  #endif
      c_in[ii] = p?gmem_c_in[tid] : 0;
    }
    bias[ii] = p?gmem_biases[tid] : zero4;
  }

  dot_product_weight = p? gmem_dot_product_weights[tid] : 0.0f;

  // notify host that we are ready for compute
  gpu_cpu_sync_part_0(params.host_ld_complete, block);

  auto s_exit = reinterpret_cast<uint *>(shared_sum);

  bool odd_step = false;
  #if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
  #endif
  do
  {

    // reset increment
    smem_input 
    = reinterpret_cast<elem_type*>(shared_sum + BLOCK_THREADS);
  // wait for host to signal starting the computation or exit
  // it is only relevant for always spin case

#if DISABLE_ATOMIC_EXCHANGE == 1
  gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch, s_exit);
#else
  gpu_cpu_sync_part_1<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, s_exit);
#endif

  if (DEVICE_ALWAYS_SPIN == 1) {
    if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE){
      if (block.thread_rank() == 0) {
        gpu_pingpong_handles::signal_atomic_relaxed<uint>((uint *)params.host_ld_complete, 
                                                          HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
      }
      //block.sync();
      return;
    } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE){
      odd_step = false;
#if AS_STATES_LD_BAR == 1
      // woraround - we must continue with preprocess to signal states load after the initX/reset
#if DISABLE_ATOMIC_EXCHANGE == 1

      // reset device spin switch
      device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
      gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch);
#else
      gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block);
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
      gemm_h[ii] = p?gmem_gemm_h[tid] : zero4;
  #else
      union Vec4_t{
        elem_type x[4];
        elem4_type x4;
      };
      Vec4_t val;
      auto gmem_gemm_h = reinterpret_cast<elem_type*>(params.gmem_gemm_h[ii]);
      val.x[0] = p?gmem_gemm_h[tid] : zero;
      val.x[1] = p?gmem_gemm_h[tid + 1 * NUM_UNITS] : zero;
      val.x[2] = p?gmem_gemm_h[tid + 2 * NUM_UNITS] : zero;
      val.x[3] = p?gmem_gemm_h[tid + 3 * NUM_UNITS] : zero;
      gemm_h[ii] = val.x4;
  #endif
      c_in[ii] = p?gmem_c_in[tid] : 0;
    }

#if AS_STATES_LD_BAR == 1
    // notify host that we are ready for compute
    gpu_cpu_sync_part_0(params.host_ld_complete, block);

    // // wait for host to signal starting the computation or exit
#if DISABLE_ATOMIC_EXCHANGE == 1

    gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch);
#else
    gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block);
#endif
#endif
  }

  // Code below is timed

#if BP_TIME_OVERHEAD_EXP == 1
  early_exit(params.host_ld_complete);
#endif

  if (STAGE == 0) {
    auto host_addr = reinterpret_cast<elem_type**>(params.host_addr);
    static_assert(BLOCK_THREADS * sizeof(elem4_type) >= sizeof(void *));
    auto smem_host_ptr = reinterpret_cast<elem_type**>(shared_sum);
    if (threadIdx.x == 0) {
      *smem_host_ptr = host_addr[0];
    }
    __syncthreads();
      for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS) {
        smem_input[ii] = (*smem_host_ptr)[ii];
    }
  } else {
      for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS) {
        smem_input[ii] = __ldcv(&gmem_input[ii]);
    }
  }

  block.sync();


  elem4_type acc = zero4;

#if BP_16BIT_ELTWISE_MATH_OPT == 1
  elem_type hx, cx;
#else
  float hx, cx;
#endif

  #pragma unroll
  for (int ii = 0; ii < NUM_INPUTS_PER_GROUP; ii++) {
    int i = group_id * NUM_INPUTS_PER_GROUP + ii;
    acc = fma4<DATA_KIND>(weights[ii], smem_input[i], acc);
  }

  enum { MAX_WARP_SIZE_GROUP_SIZE = Max<WARP_SIZE, WARP_GROUP_SIZE>::VALUE};

  if (NUM_GROUPS > 1) {
    shared_sum[threadIdx.x] = acc;
    __syncthreads();

    // offset shared_sum
    auto shared_sum_local = shared_sum + cluster_id * WARP_GROUP_SIZE * NUM_GROUPS;

    if (cluster_idx < MAX_WARP_SIZE_GROUP_SIZE) {
      auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
      #pragma unroll
      for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
        acc = add4<DATA_KIND>(acc, shared_sum_local[ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
      }
    }

    if (WARP_GROUP_SIZE < WARP_SIZE) {
      // final reduction using schfl_down_sync
      for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
        acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
      }
    }
  }

  if (cluster_idx < WARP_GROUP_SIZE) {
    acc = add4<DATA_KIND>(acc, bias[0]); 
    acc = add4<DATA_KIND>(acc, gemm_h[0]);

    lstm_eltwise<DATA_KIND>(hx, cx, acc, c_in[0]);

#if DEBUG_OUTPUTS == 1
    assert(params.always_spin == 0);
    if (p) gmem_c_out[threadIdx.x] = from_float<DATA_KIND>(cx);
#endif
  }

  {
   __syncthreads();

  }

  smem_input += NUM_FEATURES_UP;

  if (tid < NUM_FEATURES_UP && group_id == 0) {
    smem_input[cluster_id * WARP_GROUP_SIZE + group_idx] = from_float<DATA_KIND>(hx);
  }

  // second layer

  gmem_c_out = reinterpret_cast<elem_type*>(params.gmem_c_out[1]);

  acc = zero4;

  __syncthreads();

  #pragma unroll
  for (int ii = 0; ii < NUM_UNITS_PER_GROUP; ii++) {
    int i = group_id * NUM_UNITS_PER_GROUP + ii;
    acc = fma4<DATA_KIND>(smem_weights[i * WARP_GROUP_SIZE + group_idx], smem_input[i], acc);
  }

  if (NUM_GROUPS > 1) {
    shared_sum[threadIdx.x] = acc;
    __syncthreads();

    // offset shared_sum
    auto shared_sum_local = shared_sum + cluster_id * WARP_GROUP_SIZE * NUM_GROUPS;

    if (cluster_idx < MAX_WARP_SIZE_GROUP_SIZE) {
      auto gidx = (WARP_GROUP_SIZE >= WARP_SIZE)? group_idx : threadIdx.x % WARP_SIZE;
      #pragma unroll
      for (int ii = 1; ii < NUM_GROUPS / NUM_SUBWARP_GROUPS; ii++) {
        acc = add4<DATA_KIND>(acc, shared_sum_local[ii * MAX_WARP_SIZE_GROUP_SIZE + gidx]);
      }

      if (WARP_GROUP_SIZE < WARP_SIZE) {
        // final reduction using schfl_down_sync
        for (int offset = WARP_SIZE / 2; offset >= WARP_GROUP_SIZE; offset /= 2) {
          acc = add4<DATA_KIND>(acc, shfl_down_elem4_sync(0xffffffff, acc, offset, WARP_SIZE));
        }
      }
    }
  }

  float dot_product;
  auto smem_tmp = reinterpret_cast<float*>(smem_input); 
  if (cluster_idx < WARP_GROUP_SIZE) {
    acc = add4<DATA_KIND>(acc, bias[1]); 
    acc = add4<DATA_KIND>(acc, gemm_h[1]);

    lstm_eltwise<DATA_KIND>(hx, cx, acc, c_in[1]);

  #if DEBUG_OUTPUTS == 1
    if (p) {
      gmem_c_out[tid] = from_float<DATA_KIND>(cx);
      gmem_h_out[tid] = from_float<DATA_KIND>(hx);
    }
  #endif
  }

  // Below we need to find the scallar product between dot_prod weights and hx
  enum { MIN_WARP_SIZE_GROUP_SIZE = Min<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };
  enum { NUM_WARPS_IN_GROUP = Div_up<WARP_GROUP_SIZE, WARP_SIZE>::VALUE };

  if (cluster_idx < MAX_WARP_SIZE_GROUP_SIZE) {

    dot_product = (p && cluster_idx < WARP_GROUP_SIZE)? dot_product_weight * hx : 0.0f;

    #pragma unroll
    for (int offset = MIN_WARP_SIZE_GROUP_SIZE / 2; offset > 0; offset /= 2) {
      dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
    }

    if (lane_id == 0) {
      smem_tmp[cluster_id * MAX_WARP_SIZE_GROUP_SIZE / WARP_SIZE + group_idx / WARP_SIZE] = dot_product;
    }
  }

  {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    #pragma unroll
    for (int ii = 1; ii < CLUSTER_SIZE * MAX_WARP_SIZE_GROUP_SIZE / WARP_SIZE; ii++) {
      dot_product += smem_tmp[ii];
    }
    dot_product += params.dot_prod_bias;

    // if (isnan(dot_product)) {
    //   printf("dot_product is nan: %f\n", dot_product);
    // }
  }

  gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, block);
  odd_step = !odd_step;
  } while(DEVICE_ALWAYS_SPIN);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// FP32/FP16/BP16 implementation
template <int NUM_LAYERS, int NUM_INPUTS, int NUM_UNITS, int BLOCK_THREADS, Data_kind DATA_KIND, 
          bool STAGE=false, bool DEVICE_ALWAYS_SPIN=false, bool GDRCOPY_ENABLED=false>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void gemv_lstm_ping_pong_bpt_kernel(Lstm_bpt_params<NUM_LAYERS, NUM_INPUTS, NUM_UNITS> params) {

  using data_type = Data_type<DATA_KIND>;
  using elem_type  = typename data_type::Elem_type;

  enum { WARP_SIZE  = 32 };
  enum { NUM_WARPS  = BLOCK_THREADS / WARP_SIZE };

  enum { NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE };

  __shared__ elem_type shared_sum[NUM_WARPS];

  auto gmem_input = reinterpret_cast<elem_type*>(params.gmem_input);

  auto block = cooperative_groups::this_thread_block();

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  enum { ITEMS_PER_THREAD = Div_up<NUM_INPUTS, BLOCK_THREADS>::VALUE };

  __shared__ uint32_t s_exit[1];

  gpu_cpu_sync_part_0(params.host_ld_complete, block);

#if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
#endif
do
{

// wait for host to signal starting the computation or exit
// it is only relevant for always spin case
#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch, s_exit);
#else
    gpu_cpu_sync_part_1<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, s_exit);
#endif
if (DEVICE_ALWAYS_SPIN == 1) {
  if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE){
    if (block.thread_rank() == 0) {
      gpu_pingpong_handles::signal_atomic_relaxed<uint>((uint *)params.host_ld_complete, 
                                                        HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
    }
    //block.sync();
    return;
  } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE){
    // workaround - we must continue with preprocess to signal states load after the initX/reset
#if AS_STATES_LD_BAR_IN_NULL_MODEL == 0
    // notify host that we are ready for compute
    gpu_cpu_sync_part_0(params.host_ld_complete, block);
    continue;
#else
#if DISABLE_ATOMIC_EXCHANGE == 1
    // reset device spin switch
    device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
    gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch);
#else
    gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block);
#endif
#endif
  }

#if AS_STATES_LD_BAR_IN_NULL_MODEL == 1
// notify host that we are ready for compute
  gpu_cpu_sync_part_0(params.host_ld_complete, block);

// // wait for host to signal starting the computation or exit
#if DISABLE_ATOMIC_EXCHANGE == 1

  gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block, &device_spin_switch);
#else
  gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(params.gmem_ld_complete, block);
#endif
#endif
}

  // Code below is timed

#if BP_TIME_OVERHEAD_EXP == 1
  early_exit(params.host_ld_complete);
#endif

  elem_type input[ITEMS_PER_THREAD];

  if (STAGE == 0) {
    auto host_addr = reinterpret_cast<elem_type**>(params.host_addr);
    __shared__ elem_type* smem_host_ptr[1];
    if (threadIdx.x == 0) {
      *smem_host_ptr = host_addr[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ii = 0; ii < ITEMS_PER_THREAD; ii++) {
      int idx = ii * BLOCK_THREADS + threadIdx.x;
      input[ii] = (idx < NUM_INPUTS)? (*smem_host_ptr)[idx] : elem_type(0);
    }
  } else {
    #pragma unroll
    for (int ii = 0; ii < ITEMS_PER_THREAD; ii++) {
      int idx = ii * BLOCK_THREADS + threadIdx.x;
      input[ii] = (idx < NUM_INPUTS)? gmem_input[idx] : elem_type(0);
    }
  }

  elem_type sum = elem_type(0);

  #pragma unroll
  for (int ii = 0; ii < ITEMS_PER_THREAD; ii++) {
    sum = add<DATA_KIND>(sum, input[ii]);
  }

  #pragma unroll
  for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum = add<DATA_KIND>(
            sum,
            shfl_down_elem_sync(0xffffffff, sum, offset, WARP_SIZE));
  }

  float sum_f = 0.0f;
  if (NUM_WARPS > 1) {
    if (lane_id == 0) {
      shared_sum[warp_id] = sum;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      #pragma unroll
      for (int ii = 1; ii < NUM_WARPS; ii++) {
        sum = add<DATA_KIND>(sum, shared_sum[ii]);
      }
    }
  }

  if (threadIdx.x == 0) {
    sum_f = to_float<DATA_KIND>(sum);
  }

  gpu_cpu_sync_part_2(params.host_ld_complete, sum_f, block);

  #if AS_STATES_LD_BAR_IN_NULL_MODEL == 0
  // notify host that we are ready for compute
  // then we wait for host to signal starting the computation or exit in 
  // the sync_part_1 right after "do" start
  if (DEVICE_ALWAYS_SPIN){
    gpu_cpu_sync_part_0(params.host_ld_complete, block);
  }
  #endif

  } while(DEVICE_ALWAYS_SPIN);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<Arch A, Data_kind DATA_KIND, bool STAGE, bool DEVICE_ALWAYS_SPIN, bool GDRCOPY_ENABLED>
inline void launch_lstm_s_bpt(Lstm_s_bpt_params& params, cudaStream_t stream) {

  constexpr int NUM_UNITS  = Lstm_s_bpt_params::NUM_UNITS;
  constexpr int NUM_INPUTS = Lstm_s_bpt_params::NUM_INPUTS;
  constexpr int NUM_LAYERS = Lstm_s_bpt_params::NUM_LAYERS;
  constexpr int WARP_SIZE = 32;

  constexpr int WARP_GROUP_SIZE = 32; //WARP_SIZE, 16;

  constexpr int NUM_SUBWARP_GROUPS = Div_up<WARP_SIZE, WARP_GROUP_SIZE>::VALUE;

  constexpr int NUM_GROUPS = NUM_SUBWARP_GROUPS * 2; // NUM_WARPS_PER_ACU

  static_assert(NUM_UNITS % NUM_GROUPS == 0);
  static_assert(NUM_INPUTS % NUM_GROUPS == 0);
  static_assert(NUM_GROUPS % NUM_SUBWARP_GROUPS == 0);

  constexpr int CLUSTER_SIZE_DOWN = Div_up<NUM_UNITS, WARP_GROUP_SIZE>::VALUE ;
  constexpr int CLUSTER_SIZE = Div_up<CLUSTER_SIZE_DOWN * WARP_GROUP_SIZE, WARP_SIZE>::VALUE * NUM_SUBWARP_GROUPS;

  constexpr int NUM_WARPS = NUM_GROUPS * CLUSTER_SIZE / NUM_SUBWARP_GROUPS;
  
  constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;

  constexpr int WEIGHT_INPUTS = (NUM_LAYERS - 1) * NUM_UNITS;

  constexpr int NUM_FEATURES = Max<NUM_UNITS, NUM_INPUTS>::VALUE;
  constexpr int NUM_FEATURES_UP = Div_up<NUM_FEATURES, NUM_GROUPS>::VALUE * NUM_GROUPS;

  constexpr int ELEM_SIZE = (DATA_KIND == DATA_TYPE_FP32)? sizeof(float) : sizeof(uint16_t);

  constexpr int  SMEM_SIZE = WEIGHT_INPUTS * CLUSTER_SIZE * WARP_GROUP_SIZE * 4 * ELEM_SIZE
                          + BLOCK_THREADS * 4 *  ELEM_SIZE
                          + 2 * NUM_FEATURES_UP * ELEM_SIZE + NUM_WARPS * sizeof(float);

  if (SMEM_SIZE > 0) {
    CHECK_CUDA( cudaFuncSetAttribute(gemv_lstm_s_bpt_kernel<NUM_LAYERS, NUM_INPUTS, NUM_UNITS,
                                      WARP_GROUP_SIZE, BLOCK_THREADS, DATA_KIND, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE)); }
  gemv_lstm_s_bpt_kernel<NUM_LAYERS, NUM_INPUTS, NUM_UNITS, WARP_GROUP_SIZE, BLOCK_THREADS, DATA_KIND, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>
  <<<1, BLOCK_THREADS, SMEM_SIZE, stream>>> (params);
}

void run_lstm_s_bpt(Lstm_s_bpt_params& params, Data_kind dtype, cudaStream_t stream) {
  kernel_dispatch::dispatch_kernel_variants<true, false>(
    dtype,                                   // dtype
    params.host_addr == nullptr,             // stage
    params.always_spin != 0,                 // device_always_spin
    params.use_gdrcopy != 0,                 // gdrcopy_enabled
    0,                                       // sm_arch ignored
    [&]<Arch A, Data_kind DT, bool S, bool P, bool G>() {
      launch_lstm_s_bpt<A, DT, S, P, G>(params, stream);
    }
  );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Params, Arch A, Data_kind DATA_KIND, bool STAGE, bool DEVICE_ALWAYS_SPIN, bool GDRCOPY_ENABLED>
inline void launch_lstm_ping_pong_bpt(Params& params, cudaStream_t stream) {
  constexpr int NUM_UNITS  = Params::NUM_UNITS;
  constexpr int NUM_INPUTS = Params::NUM_INPUTS;
  constexpr int NUM_LAYERS = Params::NUM_LAYERS;
  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_THREADS = (NUM_INPUTS > 1000)? Div_up<NUM_INPUTS, WARP_SIZE>::VALUE * WARP_SIZE : WARP_SIZE;

  gemv_lstm_ping_pong_bpt_kernel<NUM_LAYERS, NUM_INPUTS, NUM_UNITS, BLOCK_THREADS, DATA_KIND, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED> 
                                                <<<1, BLOCK_THREADS, 0, stream>>>(params);
}

template<typename Params>
void dispatch_lstm_ping_pong_bpt(Params& params, Data_kind dtype, cudaStream_t stream) {
  kernel_dispatch::dispatch_kernel_variants<true, false>(
    dtype,                                   // dtype
    params.host_addr == nullptr,             // stage
    params.always_spin != 0,                 // device_always_spin
    params.use_gdrcopy != 0,                 // gdrcopy_enabled
    0,                                       // sm_arch ignored
    [&]<Arch A, Data_kind DT, bool S, bool P, bool G>() {
      launch_lstm_ping_pong_bpt<Params, A, DT, S, P, G>(params, stream);
    }
  );
}

void run_lstm_s_ping_pong_bpt(Lstm_s_bpt_params& params, Data_kind dtype, cudaStream_t stream) {
  dispatch_lstm_ping_pong_bpt<Lstm_s_bpt_params>(params, dtype, stream);
}

void run_lstm_m_ping_pong_bpt(Lstm_m_bpt_params& params, Data_kind dtype, cudaStream_t stream) {
  dispatch_lstm_ping_pong_bpt<Lstm_m_bpt_params>(params, dtype, stream);
}

void run_lstm_l_ping_pong_bpt(Lstm_l_bpt_params& params, Data_kind dtype, cudaStream_t stream) {
  dispatch_lstm_ping_pong_bpt<Lstm_l_bpt_params>(params, dtype, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void clear_l2_cache(void *data, size_t num_bytes, cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 256;
  assert(num_bytes % (BLOCK_THREADS * sizeof(int)) == 0);
  read_only_kernel<<<num_bytes / (sizeof(int) * BLOCK_THREADS), BLOCK_THREADS, 0, stream>>>
                                                        (num_bytes / sizeof(int), 0, (int *)data);
}

} // namespace ssgemv