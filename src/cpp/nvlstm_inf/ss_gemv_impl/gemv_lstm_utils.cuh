/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_fp16.h>
#include <stdint.h>

#include "kernel_params.h"

namespace nvidia::quant::lowlat_infer::ssgemv {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gpu_pingpong_handles {

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ uint32_t poll_atomic(datatype_t* x, datatype_t expected_val, datatype_t reset_val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);

  bool test;
  datatype_t res = (datatype_t)0;
  do {
    datatype_t _expected = expected_val;
    test = flag.compare_exchange_strong(_expected, reset_val, cuda::memory_order_relaxed);
    res = _expected;
    // printf("%lx\n",_expected);
  } while (!test);
  return res;
}

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ uint32_t
poll_atomic(datatype_t* x, datatype_t expected_val, datatype_t reset_val, datatype_t exit_val, datatype_t init_val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);

  bool test;
  datatype_t res = (datatype_t)0;
  do {
    datatype_t _expected = expected_val;
    // test = flag.compare_exchange_strong(_expected,reset_val,cuda::memory_order_relaxed);
    test = flag.compare_exchange_strong(_expected, reset_val, cuda::memory_order_acq_rel);
    res = _expected;
    // printf("%lx\n",_expected);
    test = test || _expected == exit_val || _expected == init_val;
  } while (!test);
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ datatype_t poll_load(datatype_t* x, datatype_t expected_val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);

  while (flag.load(cuda::memory_order_acquire) != expected_val);
  // bool test = false;
  // do {
  //     datatype_t res = flag.load(cuda::memory_order_acquire);
  //     //printf("%lx\n",res);
  //     test = (res == expected_val);
  // } while(!test);
  return expected_val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ datatype_t
poll_load(datatype_t* x, datatype_t expected_val, datatype_t exit_val, datatype_t init_val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);

  bool test;
  datatype_t res = (datatype_t)0;
  do {
    res = flag.load(cuda::memory_order_acquire);
    // printf("%lx\n",res);
    test = (res == expected_val) || (res == exit_val) || (res == init_val);
  } while (!test);
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ void signal_atomic_relaxed(datatype_t* x, datatype_t val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);
  flag.exchange(val, cuda::memory_order_relaxed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ void store_atomic_relaxed(datatype_t* x, datatype_t val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);
  flag.store(val, cuda::std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename datatype_t, cuda::thread_scope threadscope = cuda::thread_scope_system>
__device__ __forceinline__ void store_atomic_release(datatype_t* x, datatype_t val) {
  cuda::atomic_ref<datatype_t, threadscope> flag(*x);
  flag.store(val, cuda::std::memory_order_release);
}

}  // namespace gpu_pingpong_handles

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type>
static inline __device__ void gpu_cpu_sync_part_0(volatile int* signal, group_type& group) {
  group.sync();

  if (group.thread_rank() == 0) {
#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_pingpong_handles::store_atomic_relaxed<uint>((uint*)signal, HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
#else
    gpu_pingpong_handles::signal_atomic_relaxed<uint>((uint*)signal, HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type>
static inline __device__ void gpu_cpu_sync_broadcast_flag_(
    group_type& group, volatile uint32_t* s_exit_flag, uint32_t flag) {};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
static inline __device__ void gpu_cpu_sync_broadcast_flag_(
    cooperative_groups::cluster_group& group, volatile uint32_t* s_exit_flag, uint32_t flag) {
  if (flag) {
    // broadcast exit flag
    if (group.thread_rank() == 0) {
#pragma unroll
      for (int ii = 1; ii < group.num_blocks(); ii++) {
        uint32_t* dst_smem = group.map_shared_rank((uint32_t*)s_exit_flag, ii);
        dst_smem[0] = flag;
      }
    }
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type, bool DEVICE_ALWAYS_SPIN>
static inline __device__ void gpu_cpu_sync_store_broadcast_group_sync_(
    group_type& group, volatile uint32_t* s_exit_flag, uint32_t flag) {
  if (DEVICE_ALWAYS_SPIN) {
    __syncwarp();
    if (group.thread_rank() == 0) {
      *s_exit_flag = flag;
    }
    gpu_cpu_sync_broadcast_flag_(group, s_exit_flag, flag);
  }
  group.sync();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type, bool DEVICE_ALWAYS_SPIN = 0, bool GDRCOPY_ENABLED = 0>
static inline __device__ void gpu_cpu_sync_part_1_no_aex(
    volatile int* signal,
    group_type& group,
    uint32_t* device_spin_switch = nullptr,
    volatile uint32_t* s_exit_flag = nullptr) {
  uint32_t flag = 0U;
  constexpr cuda::thread_scope THREAD_SCOPE = (GDRCOPY_ENABLED) ? cuda::thread_scope_device : cuda::thread_scope_system;
  if (group.thread_rank() == 0) {
    if (DEVICE_ALWAYS_SPIN) {
      flag = gpu_pingpong_handles::poll_load<uint, THREAD_SCOPE>(
          (uint*)signal,
          DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE * *device_spin_switch,
          DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE,
          DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE);
    } else {
      flag = gpu_pingpong_handles::poll_load<uint, THREAD_SCOPE>(
          (uint*)signal, DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE * *device_spin_switch);
    }
    *device_spin_switch = !(*device_spin_switch);
  }
  gpu_cpu_sync_store_broadcast_group_sync_<group_type, DEVICE_ALWAYS_SPIN>(group, s_exit_flag, flag);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type, bool DEVICE_ALWAYS_SPIN = 0, bool GDRCOPY_ENABLED = 0>
static inline __device__ void gpu_cpu_sync_part_1(
    volatile int* signal, group_type& group, volatile uint32_t* s_exit_flag = nullptr) {
  uint32_t flag = 0U;
  constexpr cuda::thread_scope THREAD_SCOPE = (GDRCOPY_ENABLED) ? cuda::thread_scope_device : cuda::thread_scope_system;
  if (group.thread_rank() == 0) {
    if (DEVICE_ALWAYS_SPIN) {
      flag = gpu_pingpong_handles::poll_atomic<uint, THREAD_SCOPE>(
          (uint*)signal,
          DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE,
          0u,
          DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE,
          DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE);
    } else {
      flag = gpu_pingpong_handles::poll_atomic<uint, THREAD_SCOPE>((uint*)signal, DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);
    }
  }

  gpu_cpu_sync_store_broadcast_group_sync_<group_type, DEVICE_ALWAYS_SPIN>(group, s_exit_flag, flag);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename group_type>
static inline __device__ void gpu_cpu_sync_part_2(int* signal, float value, group_type& group) {
  if (group.thread_rank() == 0) {
#if DO_DEBUG == 1
    assert(isnan(value) == false);
#endif
#if DISABLE_ATOMIC_EXCHANGE == 1
    gpu_pingpong_handles::store_atomic_relaxed<float>(reinterpret_cast<float*>(signal), value);
#else
    gpu_pingpong_handles::signal_atomic_relaxed<float>(reinterpret_cast<float*>(signal), value);
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// #define TWO_FP16 __ushort_as_half((unsigned short)0x4000)
#define TWO_FP16 __float2half(2.f)

static __forceinline__ __device__ float sigma_g(float in) { return __fdividef(1.f, (1.f + __expf(-in))); }
static __forceinline__ __device__ float sigma_c(float in) { return tanhf(in); }

static __forceinline__ __device__ __half sigma_g_16bit(__half in) {
  __half res = hrcp(CUDART_ONE_FP16 + hexp(-in));
  return res;
}
static __forceinline__ __device__ __half sigma_c_16bit(__half in) {
  __half res = hexp(TWO_FP16 * in);
  res = (res - CUDART_ONE_FP16) * hrcp(res + CUDART_ONE_FP16);
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline __device__ T shfl_down_elem_sync(unsigned int mask, T& val, unsigned int delta, int width);

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint16_t
shfl_down_elem_sync<uint16_t>(unsigned int mask, uint16_t& val, unsigned int delta, int width) {
  auto res = __shfl_down_sync(mask, *reinterpret_cast<__half*>(&val), delta, width);
  return *reinterpret_cast<uint16_t*>(&res);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float shfl_down_elem_sync<float>(unsigned int mask, float& val, unsigned int delta, int width) {
  return __shfl_down_sync(mask, val, delta, width);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline __device__ T shfl_down_elem4_sync(unsigned int mask, T& val, unsigned int delta, int width);

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint2 shfl_down_elem4_sync<uint2>(unsigned int mask, uint2& val, unsigned int delta, int width) {
  auto res = __shfl_down_sync(mask, *reinterpret_cast<unsigned long long*>(&val), delta, width);
  return *reinterpret_cast<uint2*>(&res);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float4 shfl_down_elem4_sync<float4>(unsigned int mask, float4& val, unsigned int delta, int width) {
  auto f0 = make_float2(val.x, val.y);
  auto f1 = make_float2(val.z, val.w);
  auto u0 = __shfl_down_sync(mask, *reinterpret_cast<unsigned long long*>(&f0), delta, width);
  auto u1 = __shfl_down_sync(mask, *reinterpret_cast<unsigned long long*>(&f1), delta, width);
  f0 = *reinterpret_cast<float2*>(&u0);
  f1 = *reinterpret_cast<float2*>(&u1);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Max {
  enum { VALUE = A >= B ? A : B };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Min {
  enum { VALUE = A <= B ? A : B };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float half_to_float(uint16_t h) {
  float f;
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
  return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float bf16_to_float(uint16_t h) {
  float f;
  asm volatile("cvt.f32.bf16 %0, %1;\n" : "=f"(f) : "h"(h));
  return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void half2_to_float2(float& x, float& y, uint32_t h) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(h));
  x = half_to_float(lo);
  y = half_to_float(hi);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void bf162_to_float2(float& x, float& y, uint32_t h) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(h));
  x = bf16_to_float(lo);
  y = bf16_to_float(hi);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float4 half4_to_float4(uint2 x) {
  float4 res;
  half2_to_float2(res.x, res.y, x.x);
  half2_to_float2(res.z, res.w, x.y);
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float4 bf164_to_float4(uint2 x) {
  float4 res;
  bf162_to_float2(res.x, res.y, x.x);
  bf162_to_float2(res.z, res.w, x.y);
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ float4 to_float4(float4 x) {
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ float4 to_float4(uint2 x) {
  if constexpr (data_kind == DATA_TYPE_FP16) {
    return half4_to_float4(x);
  } else {
    return bf164_to_float4(x);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ float to_float(float x) {
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ float to_float(uint16_t x) {
  if constexpr (data_kind == DATA_TYPE_FP16) {
    return half_to_float(x);
  } else {
    return bf16_to_float(x);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t float_to_half(float f) {
  uint16_t h;
  asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t float_to_bf16(float f) {
  uint16_t h;
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ typename Data_type<data_kind>::Elem_type from_float(float x) {
  if constexpr (data_kind == DATA_TYPE_FP32) {
    return x;
  } else if constexpr (data_kind == DATA_TYPE_FP16) {
    return float_to_half(x);
  } else {
    return float_to_bf16(x);
  }
}

template <Data_kind data_kind>
static inline __device__ uint16_t from_float(uint16_t x) {
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// template <Data_kind DATA_KIND>
// static inline __device__ typename Data_type<DATA_KIND>::Elem4_type from_float4(float4 x) {
//     // todo condition by DATA_KIND for BF16
//     return float4_to_half4(x);
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float ffma(float a, float b, float c) { return a * b + c; }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float2 ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = a.x * b.x + c.x;
  d.y = a.y * b.y + c.y;
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float4 ffma4(const float4 a, const float4 b, float4 c) {
  float4 d;
  d.x = a.x * b.x + c.x;
  d.y = a.y * b.y + c.y;
  d.z = a.z * b.z + c.z;
  d.w = a.w * b.w + c.w;
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float4 ffma4(const float4 a, const float b, float4 c) {
  float4 d;
  d.x = a.x * b + c.x;
  d.y = a.y * b + c.y;
  d.z = a.z * b + c.z;
  d.w = a.w * b + c.w;
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t hfma(uint16_t a, uint16_t b, uint16_t c) {
  uint16_t d;
  asm volatile("fma.rn.f16 %0, %1, %2, %3;" : "=h"(d) : "h"(a), "h"(b), "h"(c));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hfma4(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = hfma2(a.x, b.x, c.x);
  d.y = hfma2(a.y, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hfma4(uint2 a, uint16_t b, uint2 c) {
  uint2 d;
  uint32_t bxy = ((uint32_t)b << 16) | (uint32_t)b;
  d.x = hfma2(a.x, bxy, c.x);
  d.y = hfma2(a.y, bxy, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t bfma(uint16_t a, uint16_t b, uint16_t c) {
  uint16_t d = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(d) : "h"(a), "h"(b), "h"(c));
#endif
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t bfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 bfma4(uint2 a, uint16_t b, uint2 c) {
  uint2 d;
  uint32_t bxy = ((uint32_t)b << 16) | (uint32_t)b;
  d.x = bfma2(a.x, bxy, c.x);
  d.y = bfma2(a.y, bxy, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ typename Data_type<data_kind>::Elem_type fma(
    typename Data_type<data_kind>::Elem_type a,
    typename Data_type<data_kind>::Elem_type b,
    typename Data_type<data_kind>::Elem_type c) {
  if constexpr (data_kind == DATA_TYPE_FP32) {
    return ffma(a, b, c);
  } else if constexpr (data_kind == DATA_TYPE_FP16) {
    return hfma(a, b, c);
  } else {
    return bfma(a, b, c);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ typename Data_type<data_kind>::Elem4_type fma4(
    typename Data_type<data_kind>::Elem4_type a,
    typename Data_type<data_kind>::Elem_type b,
    typename Data_type<data_kind>::Elem4_type c) {
  if constexpr (data_kind == DATA_TYPE_FP32) {
    return ffma4(a, b, c);
  } else if constexpr (data_kind == DATA_TYPE_FP16) {
    return hfma4(a, b, c);
  } else {
    return bfma4(a, b, c);
  }
}

// static inline __device__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
//     return hfma2(a, b, c);
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t hadd(uint16_t a, uint16_t b) {
  uint16_t d;
  asm volatile("add.f16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t badd(uint16_t a, uint16_t b) {
  uint16_t d = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("add.bf16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
#endif
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t badd2(uint32_t a, uint32_t b) {
  uint32_t c = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("add.bf16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#endif
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t add2(uint32_t a, uint32_t b) { return hadd2(a, b); }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hadd4(uint2 a, uint2 b) { return make_uint2(hadd2(a.x, b.x), hadd2(a.y, b.y)); }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float fadd(float a, float b) { return a + b; }

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float4 fadd4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 badd4(uint2 a, uint2 b) { return make_uint2(badd2(a.x, b.x), badd2(a.y, b.y)); }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ typename Data_type<data_kind>::Elem_type add(
    typename Data_type<data_kind>::Elem_type a, typename Data_type<data_kind>::Elem_type b) {
  if constexpr (data_kind == DATA_TYPE_FP32) {
    return fadd(a, b);
  } else if constexpr (data_kind == DATA_TYPE_FP16) {
    return hadd(a, b);
  } else {
    return badd(a, b);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind data_kind>
static inline __device__ typename Data_type<data_kind>::Elem4_type add4(
    typename Data_type<data_kind>::Elem4_type a, typename Data_type<data_kind>::Elem4_type b) {
  if constexpr (data_kind == DATA_TYPE_FP32) {
    return fadd4(a, b);
  } else if constexpr (data_kind == DATA_TYPE_FP16) {
    return hadd4(a, b);
  } else {
    return badd4(a, b);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind DATA_KIND>
__forceinline__ __device__ void lstm_eltwise(
    float& hx,
    float& cx,
    typename Data_type<DATA_KIND>::Elem4_type acc,
    typename Data_type<DATA_KIND>::Elem_type c_in) {
  float4 gate = to_float4<DATA_KIND>(acc);
  gate.x = sigma_g(gate.x);
  gate.y = sigma_g(gate.y);
  gate.z = sigma_c(gate.z);
  gate.w = sigma_g(gate.w);

  cx = gate.y * to_float<DATA_KIND>(c_in) + gate.x * gate.z;
  hx = gate.w * sigma_c(cx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <Data_kind DATA_KIND>
__forceinline__ __device__ void lstm_eltwise(uint16_t& hx, uint16_t& cx, uint2 acc, uint16_t c_in) {
  // works with fp16 only for now
  union {
    __half x[4];
    uint2 val;
  } gate;

  gate.val = acc;
  gate.x[0] = sigma_g_16bit(gate.x[0]);
  gate.x[1] = sigma_g_16bit(gate.x[1]);
  gate.x[2] = sigma_c_16bit(gate.x[2]);
  gate.x[3] = sigma_g_16bit(gate.x[3]);

  auto cx_ = gate.x[1] * __ushort_as_half(c_in) + gate.x[0] * gate.x[2];
  auto hx_ = gate.x[3] * sigma_c_16bit(cx_);
  cx = *reinterpret_cast<uint16_t*>(&cx_);
  hx = *reinterpret_cast<uint16_t*>(&hx_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace nvidia::quant::lowlat_infer::ssgemv
