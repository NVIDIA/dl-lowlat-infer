/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cuda_helpers.cuh"

namespace {

__constant__ half kZeroHalf;
__constant__ float kZeroFloat;
__constant__ half kOneHalf;
__constant__ float kOneFloat;
half *kZeroHalfAddr, *kOneHalfAddr;
float *kZeroFloatAddr, *kOneFloatAddr;

bool const kConstantInit = []() {  // NOLINT(cert-err58-cpp)
  float constexpr zero_float{0};
  half const zero_half{static_cast<half>(zero_float)};
  float constexpr one_float{1};
  half const one_half{static_cast<half>(one_float)};
  if (nvidia::quant::lowlat_infer::cex::DeviceCount() > 0) {
    return cudaMemcpyToSymbol(kZeroFloat, &zero_float, sizeof(float)) == cudaSuccess &&
           cudaMemcpyToSymbol(kZeroHalf, &zero_half, sizeof(half)) == cudaSuccess &&
           cudaMemcpyToSymbol(kOneFloat, &one_float, sizeof(float)) == cudaSuccess &&
           cudaMemcpyToSymbol(kOneHalf, &one_half, sizeof(half)) == cudaSuccess &&
           cudaGetSymbolAddress(reinterpret_cast<void **>(&kZeroFloatAddr), kZeroFloat) == cudaSuccess &&
           cudaGetSymbolAddress(reinterpret_cast<void **>(&kZeroHalfAddr), kZeroHalf) == cudaSuccess &&
           cudaGetSymbolAddress(reinterpret_cast<void **>(&kOneFloatAddr), kOneFloat) == cudaSuccess &&
           cudaGetSymbolAddress(reinterpret_cast<void **>(&kOneHalfAddr), kOneHalf) == cudaSuccess;
  }
  return false;
}();

}  // namespace

namespace nvidia::quant::lowlat_infer::cex {

int DeviceCount() {
  int device_count{0};
  auto status = cudaGetDeviceCount(&device_count);
  if (!(status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)) CudaCheck(status);
  return device_count;
}

std::string DeviceInfo() {
  auto const device_count = DeviceCount();
  if (device_count == 0) return "";
  int device_id;
  CudaCheck(cudaGetDevice(&device_id));
  cudaDeviceProp prop{};
  CudaCheck(cudaGetDeviceProperties(&prop, device_id));
  return std::string{prop.name} + " (id: " + std::to_string(device_id) + " of " + std::to_string(device_count) + ")";
}

void FloatToHalf(half *dst, float const *src, std::size_t size) {
  std::transform(src, src + size, dst, __float2half_rn);
}

template <>
half *ConstZero() {
  StateCheck(kConstantInit);
  return kZeroHalfAddr;
}

template <>
float *ConstZero() {
  StateCheck(kConstantInit);
  return kZeroFloatAddr;
}

template <>
half *ConstOne() {
  StateCheck(kConstantInit);
  return kOneHalfAddr;
}

template <>
float *ConstOne() {
  StateCheck(kConstantInit);
  return kOneFloatAddr;
}

namespace {
template <typename T>
__global__ __launch_bounds__(1) void AtomicStore_kernel(T &flag, bool value, cuda::std::memory_order mo) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    flag.store(value, mo);
  }
}

template <typename T>
__global__ __launch_bounds__(1) void AtomicSpin_kernel(T &flag, bool expect, cuda::std::memory_order mo) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
#ifdef USE_PCIE_OUTBOUND_ATOMICS
    auto e = expect;
    while (!flag.compare_exchange_weak(e, !expect, mo)) {
      e = expect;
      // wait
    }
#else
    auto const mo_rd = mo == cuda::std::memory_order_acq_rel ? cuda::std::memory_order_acquire : mo;
    auto const mo_wt = mo == cuda::std::memory_order_acq_rel ? cuda::std::memory_order_release : mo;
    while (flag.load(mo_rd) != expect) {
      // wait
    }
    flag.store(!expect, mo_wt);
#endif
  }
}

__global__ __launch_bounds__(1) void VolatileStore_kernel(bool volatile *flag, bool value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *flag = value;
  }
}

__global__ __launch_bounds__(1) void VolatileSpin_kernel(bool volatile &flag, bool expect) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (flag != expect) {
      // wait
    }
  }
}
}  // namespace

void spsc::Store(bool volatile *flag, bool value, cudaStream_t stream) {
  dim3 const dim_grid{1}, dim_block{1};
  CudaCheck(cudaGetLastError());  // reset errors
  VolatileStore_kernel<<<dim_grid, dim_block, 0, stream>>>(flag, value);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

void spsc::WaitUntil(bool volatile &flag, bool value, cudaStream_t stream) {
  dim3 const dim_grid{1}, dim_block{1};
  CudaCheck(cudaGetLastError());  // reset errors
  VolatileSpin_kernel<<<dim_grid, dim_block, 0, stream>>>(flag, value);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

__host__ void Event::WaitReset(cudaStream_t stream, cuda::std::memory_order mo) {
  StateCheck(mo == cuda::std::memory_order_acq_rel || mo == cuda::std::memory_order_seq_cst);
  dim3 const dim_grid{1}, dim_block{1};
  CudaCheck(cudaGetLastError());  // reset errors
  AtomicSpin_kernel<<<dim_grid, dim_block, 0, stream>>>(ready_, true, mo);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}
__host__ void Event::Notify(cudaStream_t stream, cuda::std::memory_order mo) {
  dim3 const dim_grid{1}, dim_block{1};
  CudaCheck(cudaGetLastError());  // reset errors
  AtomicStore_kernel<<<dim_grid, dim_block, 0, stream>>>(ready_, true, mo);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

}  // namespace nvidia::quant::lowlat_infer::cex
