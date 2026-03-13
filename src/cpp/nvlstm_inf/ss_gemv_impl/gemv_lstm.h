/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <chrono>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>

#include <atomic>
#include <iostream>
#include <cstring>

#include "kernel_params.h"

namespace nvidia::quant::lowlat_infer::ssgemv {

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t status_ = call; \
  if( status_ != CUBLAS_STATUS_SUCCESS ) { \
    fprintf(stderr, "CUBLAS Error at line %d: %s\n", __LINE__, cublasGetStatusName(status_)); \
    exit(1); \
  } \
} while(0)

// CPU ping ///////////////////////////////////////////////////////////////////////////////////////

namespace cpu_pingpong_handles{

template<typename datatype_t>
inline void signal_atomic_release(datatype_t* x, datatype_t val){

  std::atomic_ref<datatype_t> flag(*x);
  flag.exchange(val, std::memory_order_release);

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename datatype_t>
inline void signal_release(datatype_t* x, datatype_t val) {
  // indicate that the GPU can load data
  cpu_pingpong_handles::signal_atomic_release<datatype_t>(x, val);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename datatype_t>
inline void signal_release(datatype_t* x, datatype_t val, uint32_t* device_spin_switch) {
  // indicate that the GPU can load data
  cpu_pingpong_handles::signal_atomic_release<datatype_t>(x, val * *device_spin_switch);
  *device_spin_switch = !(*device_spin_switch);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename datatype_t>
inline void poll_atomic(datatype_t* x, datatype_t expected_val, datatype_t reset_val){

  std::atomic_ref<datatype_t> flag(*x);

  bool test;
  do {
          datatype_t _expected=expected_val;  
          test = flag.compare_exchange_strong(_expected,reset_val,std::memory_order_acq_rel);
          //printf("poll_atomic %#010x %#010x\n", _expected, expected_val);
  } while(!test);
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename datatype_t, bool EQUAL = false>
    inline uint poll_load(datatype_t* x, datatype_t expected_val) {

#if NV_PROFILE == 1
    return expected_val;
#endif

    std::atomic_ref<datatype_t> flag(*x);
    if (EQUAL) {
      while(flag.load(std::memory_order_acquire) == expected_val);
      // bool test = false;
      // do {
      //     datatype_t res = flag.load(std::memory_order_acquire);
      //     printf("poll load TRUE %lx\n",res);
      //     test = (res != expected_val);
      // } while(!test);
    } else {
      while(flag.load(std::memory_order_acquire) != expected_val);
      // bool test = false;
      // do {
      //     datatype_t res = flag.load(std::memory_order_acquire);
      //     printf("%lx\n",res);
      //     test = (res == expected_val);
      // } while(!test);
    }
    return expected_val;
}

} // namespace cpu_pingpong_handles

////////////////////////////////////////////////////////////////////////////////////////////////////

// time in ms
inline double get_time() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000000.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void random_init(float *dst, int n, bool force_one = false, float scale = 1.f, int range = 3) {
  for( int i = 0; i < n; ++i ) {
    float x = 1.f;
    if( !force_one ) {
      x = (float) ((rand() % range) - (range / 2)) * scale;
    }

    dst[i] = x;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// assumes columnwise storage
//
template <typename data_type>
inline void gate_transpose(data_type* out, const data_type* in, int m, int n) {
  assert(m % NUM_GATES == 0);
  int m_out = m / NUM_GATES;

  for (int jj = 0; jj < n; jj++) {
    for (int ii = 0; ii < m_out; ii++) {
      out[m * jj + NUM_GATES * ii + 0] = in[m * jj + ii + 0 * m_out];
      out[m * jj + NUM_GATES * ii + 1] = in[m * jj + ii + 1 * m_out];
      out[m * jj + NUM_GATES * ii + 2] = in[m * jj + ii + 2 * m_out];
      out[m * jj + NUM_GATES * ii + 3] = in[m * jj + ii + 3 * m_out];
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

// assumes columnwise storage
//
template <typename data_type>
inline void weights_permute(data_type* out, const data_type* in, int num_gates, int num_input, 
                            int num_hidden) {
  // [0, 1, 2] -> [2, 1, 0]
  // e.g. [4, num_hidden, num_input] - > [num_input, num_hidden, 4]
  assert(out != in);

  // see ModelSingleStep.cuh and ModelDef.cpp for the permutation definition
  int gate_perm[4] = {0, 2, 3, 1};
  assert(num_gates == 4);

  for (int ig = 0; ig < num_gates; ig++) {
    for (int j = 0; j < num_hidden; j++) {
      for (int i = 0; i < num_input; i++) {
        size_t in_idx = gate_perm[ig] * num_input * num_hidden + j * num_input + i;
        size_t out_idx = i * num_gates * num_hidden + j * num_gates + ig;
        out[out_idx] = in[in_idx];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// assumes columnwise storage
//
template <typename data_type>
inline void weights_permute_v2(data_type* out, const data_type* in, int num_gates, int num_input, 
                            int num_hidden) {
  // [0, 1, 2] -> [1, 0, 2]
  // e.g. [4, num_hidden, num_input] - > [num_input, 4, num_hidden]
  assert(out != in);

  // see ModelSingleStep.cuh and ModelDef.cpp for the permutation definition
  int gate_perm[4] = {0, 2, 3, 1};
  assert(num_gates == 4);

  for (int ig = 0; ig < num_gates; ig++) {
    for (int j = 0; j < num_hidden; j++) {
      for (int i = 0; i < num_input; i++) {
        size_t in_idx = gate_perm[ig] * num_input * num_hidden + j * num_input + i;
        size_t out_idx = j * num_gates * num_input + ig * num_input + i;
        out[out_idx] = in[in_idx];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static bool compare(float a, float b, float &err) {
    static const float EPSILON = 1.e-1f;
    float den = fabsf(a) + fabsf(b); 
    err = den <= EPSILON ? fabsf(a-b) : fabsf(a-b) / den;
    return isnan(a) || isnan(b) || err >= EPSILON;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static void check_results(const float *out, const float *ref, int n) {
    int failed = 0;
    for( int i = 0; i < n && failed < 64; ++i, ++out, ++ref ) {
        float err;
        if( compare(*out, *ref, err) ) {
            fprintf(stderr, "Invalid result for i=%d:\n", i);
            fprintf(stderr, "    Found...: (%10.6f)\n", *out);
            fprintf(stderr, "    Expected: (%10.6f)\n", *ref);
            fprintf(stderr, "    Error...: %10.6f\n", err);
            failed++;
        }
    }

    if( !failed ) {
        printf("Test OK\n");
    } else {
        fprintf(stderr, "Test FAILED with %d failures\n", failed);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_float_to_fp16_(uint16_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        reinterpret_cast<__half*>(dst)[ii] = __float2half_rn(src[ii]);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_float_to_bf16_(uint16_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        reinterpret_cast<__nv_bfloat16*>(dst)[ii] = __float2bfloat16_rn(src[ii]);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float(void *dst, const float *src, size_t n, Data_kind dtype) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        memmove(dst, src, n*sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_float_to_fp16_(reinterpret_cast<uint16_t*>(dst), src, n);
        break;
    case DATA_TYPE_BF16:
        convert_float_to_bf16_(reinterpret_cast<uint16_t*>(dst), src, n);
        break;
    default:
        assert(false); // Not implemented!
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_fp16_to_float_(float *dst, const uint16_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __half2float(reinterpret_cast<const __half*>(src)[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_bf16_to_float_(float *dst, const uint16_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(src)[ii]);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float(float *dst, const void *src, size_t n, Data_kind dtype) {
    float* tmp = (float *)malloc(n * sizeof(float));
    switch( dtype ) {
    case DATA_TYPE_FP32:
        memcpy(tmp, src, n*sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_fp16_to_float_(tmp, reinterpret_cast<const uint16_t*>(src), n);
        break;
    case DATA_TYPE_BF16:
        convert_bf16_to_float_(tmp, reinterpret_cast<const uint16_t*>(src), n);
        break;
    default:
        assert(false); // Not implemented!
    }
    memcpy(dst, tmp, n*sizeof(float));
    free(tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void clear_l2_cache(void *data, size_t num_bytes, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_lstm_s_bpt(Lstm_s_bpt_params& params, Data_kind dtype, cudaStream_t stream);

void run_lstm_s_tbc_t(Lstm_s_tbc_t_params& params, Data_kind dtype, cudaStream_t stream);
void run_lstm_m_tbc_t(Lstm_m_tbc_t_params& params, Data_kind dtype, cudaStream_t stream);

bool verify_lstm_m_occupancy_reqs(int num_instances, int sm_arch, int elem_size);
bool verify_lstm_m_cluster_size_supported(int sm_arch, int elem_size);

void run_lstm_s_ping_pong_bpt(Lstm_s_bpt_params& params, Data_kind dtype, cudaStream_t stream);
void run_lstm_m_ping_pong_bpt(Lstm_m_bpt_params& params, Data_kind dtype, cudaStream_t stream);
void run_lstm_l_ping_pong_bpt(Lstm_l_bpt_params& params, Data_kind dtype, cudaStream_t stream);

void run_lstm_m_dev_t(Lstm_m_dev_t_params& params, Data_kind dtype, cudaStream_t stream);
void run_lstm_l_dev_t(Lstm_l_dev_t_params& params, Data_kind dtype, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace ssgemv