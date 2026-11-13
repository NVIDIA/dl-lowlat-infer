/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_COPY_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_COPY_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cstdint>
#include <cuda/cmath>

#include "cuda_helpers.cuh"

namespace nvidia::quant::lowlat_infer::copy {

namespace {
auto constexpr kWarpSize = 32u;
auto constexpr kWarpSizeLog2 = 5u;
auto constexpr kMaxGridSizeF2H = 108u;
auto constexpr kBlockSizeF2H = 128u;

dim3 constexpr BlockDim(std::size_t num_elements) {
  auto const num_elements_padded = static_cast<unsigned>(cuda::round_up(num_elements, kWarpSize));
  return dim3{num_elements_padded <= kBlockSizeF2H ? num_elements_padded : kBlockSizeF2H};
}

__device__ __forceinline__ unsigned int constexpr DivWarpSize(unsigned int x) { return x >> kWarpSizeLog2; }

__device__ __forceinline__ unsigned int constexpr ModWarpSize(unsigned int x) { return x & (kWarpSize - 1); }

template <typename T>
__device__ __forceinline__ T FloatOrHalf(float v) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half_rn(v);
  } else if constexpr (std::is_same_v<T, cex::TensorFloat32>) {
    return cex::TensorFloat32::Round(v);
  } else {
    static_assert(std::is_same_v<T, float>);
    return v;
  }
}

template <typename T>
__device__ __forceinline__ T FloatOrHalf(half v) {
  if constexpr (std::is_same_v<T, float>) {
    return __half2float(v);
  } else {
    static_assert(std::is_same_v<T, half>);
    return v;
  }
}

template <std::uint32_t kBlockSize, std::uint32_t kElemsPerThread, typename T, typename U>
__global__ __launch_bounds__(kBlockSize) void Convert_ker(
    T *__restrict__ tgt, U const *__restrict__ src, std::size_t const num_elements) {
  auto const id_lane = ModWarpSize(threadIdx.x);
  auto const id_warp_init = DivWarpSize(threadIdx.x + blockIdx.x * blockDim.x);
  auto const stride = DivWarpSize(blockDim.x * gridDim.x);
  auto constexpr kChunkSize = kWarpSize * kElemsPerThread;
  auto const num_warps = cuda::ceil_div(num_elements, kChunkSize);

  U tmp[kElemsPerThread];

  for (auto id_warp = id_warp_init; id_warp < num_warps; id_warp += stride) {
    auto const idx_warp = id_warp * kChunkSize;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        tmp[i] = src[idx];
      }
    }
#pragma unroll
    for (std::uint32_t i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        tgt[idx] = FloatOrHalf<T>(tmp[i]);
      }
    }
  }
}

}  // namespace

template <bool allow_copy = true, typename T, typename U>
void Convert(T *tgt, U const *src, std::size_t num_elements, cudaStream_t stream) {
  if constexpr (allow_copy && std::is_same_v<T, U>) {
    cex::Copy(tgt, src, num_elements, stream);
  } else {
    auto const blockDim = BlockDim(num_elements);

    CudaCheck(cudaGetLastError());  // reset errors
    if (num_elements <= blockDim.x) {
      Convert_ker<kBlockSizeF2H, 1><<<1, blockDim, 0, stream>>>(tgt, src, num_elements);
    } else if (cuda::ceil_div(num_elements, blockDim.x * 2) <= kMaxGridSizeF2H) {
      dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 2))};
      Convert_ker<kBlockSizeF2H, 2><<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements);
    } else if (cuda::ceil_div(num_elements, blockDim.x * 4) <= kMaxGridSizeF2H) {
      dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 4))};
      Convert_ker<kBlockSizeF2H, 4><<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements);
    } else {
      dim3 const gridDim{
          std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 8)), kMaxGridSizeF2H)};
      Convert_ker<kBlockSizeF2H, 8><<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements);
    }
    CudaCheck(cudaGetLastError());  // detect synchronous errors
  }
}

namespace {
template <std::uint32_t kBlockSize, std::uint32_t kElemsPerThread, typename T, typename U>
__global__ __launch_bounds__(kBlockSize) void Convert_ind(
    T *__restrict__ tgt,
    U const **__restrict__ src,
    std::size_t const batch_size,
    std::size_t const num_elem_per_batch) {
  auto const num_elements = batch_size * num_elem_per_batch;
  auto const id_lane = ModWarpSize(threadIdx.x);
  auto const id_warp_init = DivWarpSize(threadIdx.x + blockIdx.x * blockDim.x);
  auto const stride = DivWarpSize(blockDim.x * gridDim.x);
  auto constexpr kChunkSize = kWarpSize * kElemsPerThread;
  auto const num_warps = cuda::ceil_div(num_elements, kChunkSize);

  U tmp[kElemsPerThread];

  for (auto id_warp = id_warp_init; id_warp < num_warps; id_warp += stride) {
    auto const idx_warp = id_warp * kChunkSize;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        auto const idx_batch = idx / num_elem_per_batch;
        auto const idx_feature = idx % num_elem_per_batch;
        tmp[i] = src[idx_batch][idx_feature];
      }
    }
#pragma unroll
    for (std::uint32_t i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        tgt[idx] = FloatOrHalf<T>(tmp[i]);
      }
    }
  }
}
}  // namespace

template <typename T, typename U>
void Convert(T *tgt, U const **src, std::size_t batch_size, std::size_t num_elem_per_batch, cudaStream_t stream) {
  auto const num_elements = batch_size * num_elem_per_batch;
  auto const blockDim = BlockDim(num_elements);

  CudaCheck(cudaGetLastError());  // reset errors

  // This breaks the benchmark unit test for standard LSTM model
  // since the memcpy assumes contigues memory layout. Comment for now.
  // if (std::is_same<T, U>::value && batch_size == 1){
  //   CudaCheck(cudaMemcpyAsync(tgt, *src, sizeof(T) * num_elements, cudaMemcpyDefault, stream));
  //   return;
  // }

  if (num_elements <= blockDim.x) {
    Convert_ind<kBlockSizeF2H, 1><<<1, blockDim, 0, stream>>>(tgt, src, batch_size, num_elem_per_batch);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 2) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 2))};
    Convert_ind<kBlockSizeF2H, 2><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_elem_per_batch);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 4) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 4))};
    Convert_ind<kBlockSizeF2H, 4><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_elem_per_batch);
  } else {
    dim3 const gridDim{
        std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 8)), kMaxGridSizeF2H)};
    Convert_ind<kBlockSizeF2H, 8><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_elem_per_batch);
  }
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

namespace {
template <std::uint32_t kBlockSize, std::uint32_t kElemsPerThread, typename T, typename U>
__global__ __launch_bounds__(kBlockSize) void Convert_trn(
    T *__restrict__ tgt,
    U *__restrict__ src,
    std::size_t const batch_size,
    std::size_t const num_steps,
    std::size_t const num_features) {
  auto const num_steps_features = num_steps * num_features;
  auto const num_batch_features = batch_size * num_features;
  auto const num_elements = batch_size * num_steps_features;

  auto const id_lane = ModWarpSize(threadIdx.x);
  auto const id_warp_init = DivWarpSize(threadIdx.x + blockIdx.x * blockDim.x);
  auto const stride = DivWarpSize(blockDim.x * gridDim.x);
  auto constexpr kChunkSize = kWarpSize * kElemsPerThread;
  auto const num_warps = cuda::ceil_div(num_elements, kChunkSize);

  using src_type = std::remove_cv_t<std::remove_pointer_t<U>>;
  src_type tmp[kElemsPerThread];

  for (auto id_warp = id_warp_init; id_warp < num_warps; id_warp += stride) {
    auto const idx_warp = id_warp * kChunkSize;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        if constexpr (std::is_pointer_v<U>) {
          auto const idx_batch = idx / num_steps_features;
          auto const idx_step_feature = idx % num_steps_features;
          tmp[i] = src[idx_batch][idx_step_feature];
        } else {
          static_assert(std::is_same_v<U, src_type const>);
          tmp[i] = src[idx];
        }
      }
    }
#pragma unroll
    for (std::uint32_t i = 0; i < kElemsPerThread; ++i) {
      auto const idx = idx_warp + i * kWarpSize + id_lane;
      if (idx < num_elements) {
        auto const idx_batch = idx / num_steps_features;
        auto const idx_step_feature = idx % num_steps_features;
        auto const idx_step = idx_step_feature / num_features;
        auto const idx_feature = idx_step_feature % num_features;
        auto const idx_tgt = idx_step * num_batch_features + idx_batch * num_features + idx_feature;
        tgt[idx_tgt] = FloatOrHalf<T>(tmp[i]);
      }
    }
  }
}
}  // namespace

template <typename T, typename U>
void Convert(
    T *tgt,
    U const *src,
    std::size_t batch_size,
    std::size_t num_steps,
    std::size_t num_features,
    cudaStream_t stream) {
  if (batch_size == 1) {
    Convert(tgt, src, num_steps * num_features, stream);
    return;
  }

  auto const num_elements = batch_size * num_steps * num_features;
  auto const blockDim = BlockDim(num_elements);

  CudaCheck(cudaGetLastError());  // reset errors
  if (num_elements <= blockDim.x) {
    Convert_trn<kBlockSizeF2H, 1><<<1, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 2) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 2))};
    Convert_trn<kBlockSizeF2H, 2><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 4) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 4))};
    Convert_trn<kBlockSizeF2H, 4><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else {
    dim3 const gridDim{
        std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 8)), kMaxGridSizeF2H)};
    Convert_trn<kBlockSizeF2H, 8><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  }
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

template <typename T, typename U>
void Convert(
    T *tgt,
    U const **src,
    std::size_t batch_size,
    std::size_t num_steps,
    std::size_t num_features,
    cudaStream_t stream) {
  auto const num_elements = batch_size * num_steps * num_features;
  auto const blockDim = BlockDim(num_elements);

  CudaCheck(cudaGetLastError());  // reset errors
  if (num_elements <= blockDim.x) {
    Convert_trn<kBlockSizeF2H, 1><<<1, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 2) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 2))};
    Convert_trn<kBlockSizeF2H, 2><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else if (cuda::ceil_div(num_elements, blockDim.x * 4) <= kMaxGridSizeF2H) {
    dim3 const gridDim{static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 4))};
    Convert_trn<kBlockSizeF2H, 4><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  } else {
    dim3 const gridDim{
        std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x * 8)), kMaxGridSizeF2H)};
    Convert_trn<kBlockSizeF2H, 8><<<gridDim, blockDim, 0, stream>>>(tgt, src, batch_size, num_steps, num_features);
  }
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

namespace {

template <typename T>
__global__ __launch_bounds__(kBlockSizeF2H) void Memset_ker(
    T *__restrict__ tgt, T value, std::size_t const num_elements) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (auto idx = tid; idx < num_elements; idx += stride) {
    tgt[idx] = value;
  }
}

}  // namespace

template <typename T>
void Memset(T *tgt, T value, std::size_t num_elements, cudaStream_t stream) {
  auto blockDim = BlockDim(num_elements);
  if (num_elements < blockDim.x) {
    blockDim.x = num_elements;
  }
  dim3 const gridDim{std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x)), kMaxGridSizeF2H)};

  CudaCheck(cudaGetLastError());  // reset errors
  Memset_ker<<<gridDim, blockDim, 0, stream>>>(tgt, value, num_elements);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

namespace {

template <typename T>
__global__ __launch_bounds__(kBlockSizeF2H) void Memset2D_ker(
    T *__restrict__ tgt, std::size_t const pitch, T value, std::size_t const width, std::size_t const height) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  auto const num_elements = width * height;
  for (auto idx = tid; idx < num_elements; idx += stride) {
    auto const idx_h = idx / width;
    auto const idx_w = idx % width;
    tgt[idx_h * pitch + idx_w] = value;
  }
}
}  // namespace

template <typename T>
void Memset2D(T *tgt, std::size_t pitch, T value, std::size_t width, std::size_t height, cudaStream_t stream) {
  auto const num_elements = width * height;
  auto const blockDim = BlockDim(num_elements);
  dim3 const gridDim{std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x)), kMaxGridSizeF2H)};

  CudaCheck(cudaGetLastError());  // reset errors
  Memset2D_ker<<<gridDim, blockDim, 0, stream>>>(tgt, pitch, value, width, height);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

namespace {

template <typename T, typename U>
__global__ __launch_bounds__(kBlockSizeF2H) void Replicate_ker(
    T *__restrict__ tgt, U const *__restrict__ src, std::size_t const num_elements, unsigned int const times) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride_src = blockDim.x * gridDim.x;
  auto const rid = blockIdx.y * blockDim.y + threadIdx.y;
  auto const stride_tgt = blockDim.y * gridDim.y;
  for (auto idx = tid; idx < num_elements; idx += stride_src) {
    auto const value = FloatOrHalf<T>(src[idx]);
    for (auto t = rid; t < times; t += stride_tgt) {
      tgt[idx + t * num_elements] = value;
    }
  }
}

template <unsigned int times, typename T, typename U>
__global__ __launch_bounds__(kBlockSizeF2H) void Replicate_ker(
    T *__restrict__ tgt, U const *__restrict__ src, std::size_t const num_elements) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride_src = blockDim.x * gridDim.x;
  for (auto idx = tid; idx < num_elements; idx += stride_src) {
    auto const value = FloatOrHalf<T>(src[idx]);
#pragma unroll
    for (auto t = 0; t < times; ++t) {
      tgt[idx + t * num_elements] = value;
    }
  }
}

}  // namespace

template <typename T, typename U>
void Replicate(T *tgt, U const *src, std::size_t num_elements, unsigned int const times, cudaStream_t stream) {
  auto const blockDim = BlockDim(num_elements);
  dim3 const gridDim{std::min(static_cast<unsigned int>(cuda::ceil_div(num_elements, blockDim.x)), kMaxGridSizeF2H)};

  CudaCheck(cudaGetLastError());  // reset errors
  if (times == 50) {
    Replicate_ker<50><<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements);
  } else if (times == 100) {
    Replicate_ker<100><<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements);
  } else {
    Replicate_ker<<<gridDim, blockDim, 0, stream>>>(tgt, src, num_elements, times);
  }
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

namespace {

template <unsigned kNumThreads, typename T>
__global__ __launch_bounds__(kNumThreads) void ScalarProduct_ker(
    float *tgt, const T *x, const T *y, float bias, size_t num_elements) {
  assert(kNumThreads == blockDim.x);
  auto const idx_thread = threadIdx.x;
  float sum = 0.0f;
  for (std::size_t i = idx_thread; i < num_elements; i += kNumThreads) {
    // printf("%d %f %f\n", i, FloatOrHalf<float>(x[i]), FloatOrHalf<float>(y[i]));
    sum += FloatOrHalf<float>(x[i]) * FloatOrHalf<float>(y[i]);
  }

  auto cta = cg::this_thread_block();
  auto const sum_block = cg::reduce(cg::tiled_partition<kNumThreads>(cta), sum, cg::plus<float>());

  if (idx_thread == 0) {
    tgt[0] = sum_block + bias;
  }
}

}  // namespace

template <typename T>
void ScalarProduct(float *tgt, T const *x, T const *y, float bias, std::size_t num_elements, cudaStream_t stream) {
  CudaCheck(cudaGetLastError());  // reset errors
  if (num_elements <= 128) {
    ScalarProduct_ker<32><<<1, 32, 0, stream>>>(tgt, x, y, bias, num_elements);
  } else if (num_elements <= 256) {
    ScalarProduct_ker<64><<<1, 64, 0, stream>>>(tgt, x, y, bias, num_elements);
  } else {
    ScalarProduct_ker<128><<<1, 128, 0, stream>>>(tgt, x, y, bias, num_elements);
  }
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}
}  // namespace nvidia::quant::lowlat_infer::copy

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_COPY_CUH_
