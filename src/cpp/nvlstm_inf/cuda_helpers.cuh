/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_HELPERS_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_HELPERS_CUH_

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <cuda.h>
#include <cuda/atomic>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "util.h"
#include "cuda_error_util.h"
namespace nvidia::quant::lowlat_infer::cex {

int DeviceCount();

std::string DeviceInfo();

template <typename T>
class CudaHostDeleter {
 public:
  constexpr void operator()(T *ptr) const {
    if (ptr != nullptr) CudaCheck(cudaFreeHost(ptr));
  }
};

template <typename T>
class PagedDeleter {
 public:
  constexpr void operator()(T *ptr) const {
    if (ptr != nullptr) std::free(ptr);
  }
};

template <typename T>
using PinnedPtr = std::unique_ptr<T[], CudaHostDeleter<T>>;

template <typename T>
using PagedPtr = std::unique_ptr<T[], PagedDeleter<T>>;

template <typename T>
PinnedPtr<T> MallocHost(std::size_t size, unsigned int flags = cudaHostAllocMapped) {
  T *ptr;
  CudaCheck(cudaHostAlloc(&ptr, size * sizeof(T), flags));
  return PinnedPtr<T>{ptr};
}

template <typename T>
PagedPtr<T> MallocPaged(std::size_t size, unsigned int flags = 0) {
  T *ptr;
  ptr = (T *) std::malloc(size * sizeof(T));
  return PagedPtr<T>{ptr};
}

template <typename T>
class CudaObjectDeleter {
 public:
  constexpr void operator()(T *ptr) const {
    if (ptr != nullptr) {
      if constexpr (std::is_destructible_v<T>) {
        ptr->~T();
      }
      CudaCheck(cudaFreeHost(const_cast<std::remove_cv_t<T> *>(ptr)));
    }
  }
};

template <typename T>
using PinnedObjectPtr = std::unique_ptr<T, CudaObjectDeleter<T>>;

template <typename T, typename... Args>
PinnedObjectPtr<T> MakePinned(Args &&...args) {
  T *ptr;
  CudaCheck(cudaHostAlloc(&ptr, sizeof(T), cudaHostAllocMapped));
  try {
    if constexpr (std::is_scalar_v<std::remove_cv_t<T>>) {
      *ptr = T{std::forward<Args>(args)...};
    } else {
      ptr = new (ptr) T{std::forward<Args>(args)...};
    }
  } catch (...) {
    CudaCheck(cudaFreeHost(const_cast<std::remove_cv_t<T> *>(ptr)));
    throw;
  }
  return PinnedObjectPtr<T>{ptr};
}

template <typename T>
class CudaDeleter {
 public:
  constexpr void operator()(T *ptr) const {
    if (ptr != nullptr) CudaCheck(cudaFree(const_cast<std::remove_cv_t<T> *>(ptr)));
  }
};

template <typename T>
using DevicePtr = std::unique_ptr<T[], CudaDeleter<T>>;

template <typename T>
DevicePtr<T> Malloc(std::size_t size) {
  T *ptr;
  CudaCheck(cudaMalloc(&ptr, size * sizeof(T)));
  return DevicePtr<T>{ptr};
}

template <typename TArray>
class Array2D {
 public:
  using ArrayType = TArray;
  using ValueType = std::remove_reference_t<decltype(std::declval<ArrayType>()[0])>;
  using PointerType = std::add_pointer_t<ValueType>;

  static constexpr std::size_t kValueSize = sizeof(ValueType);

  Array2D() : array_{}, pitch_{} {};

  Array2D(TArray &&array, std::size_t pitch) : array_{std::forward<TArray>(array)}, pitch_{pitch} {
    StateCheck(pitch % kValueSize == 0);
  }

  Array2D(Array2D &&that) noexcept : array_{std::move(that.array_)}, pitch_{that.pitch_} {}

  Array2D &operator=(Array2D &&that) noexcept {
    this->array_ = std::move(that.array_);
    this->pitch_ = that.pitch_;
    return *this;
  }

  [[nodiscard]] std::size_t Pitch() const { return pitch_; }

  TArray &Array() { return array_; };

  [[nodiscard]] std::size_t LeadingDim() const { return pitch_ / kValueSize; }

  TArray const &Array() const { return array_; };

  ValueType &operator[](std::size_t i) { return array_[i]; }

  ValueType const &operator[](std::size_t i) const { return array_[i]; }

  ValueType *Get() { return &array_[0]; };

  ValueType const *Get() const { return &array_[0]; };

  ValueType *Row(std::size_t idx) { return &array_[idx * LeadingDim()]; }

  ValueType const *Row(std::size_t idx) const { return &array_[idx * LeadingDim()]; }

  Array2D<PointerType> AtRow(std::size_t idx) { return Array2D<PointerType>{Row(idx), pitch_}; }

  operator bool() const { return static_cast<bool>(array_); }  // NOLINT(google-explicit-constructor)

 private:
  TArray array_;
  std::size_t pitch_;
};

template <typename T>
using DeviceArray2D = Array2D<DevicePtr<T>>;

template <typename T>
DeviceArray2D<T> Malloc2D(std::size_t num_rows, std::size_t num_cols) {
  T *ptr;
  std::size_t pitch;
  //CudaCheck(cudaMallocPitch(&ptr, &pitch, num_cols * sizeof(T), num_rows));
  pitch = (num_cols * sizeof(T) + 128 - 1) / 128 * 128;
  CudaCheck(cudaMalloc(&ptr, pitch * num_rows));
  return Array2D{DevicePtr<T>{ptr}, pitch};
}

template <typename T>
using ManagedObjectPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T, typename... Args>
ManagedObjectPtr<T> MakeManaged(Args &&...args) {
  T *ptr;
  CudaCheck(cudaMallocManaged(&ptr, sizeof(T)));
  int device_id;
  CudaCheck(cudaGetDevice(&device_id));
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = device_id};
  cudaMemLocation cpu_loc = {.type = cudaMemLocationTypeHost, .id = cudaCpuDeviceId};
  CudaCheck(cudaMemAdvise(ptr, sizeof(T), cudaMemAdviseSetPreferredLocation, loc));
  CudaCheck(cudaMemAdvise(ptr, sizeof(T), cudaMemAdviseSetAccessedBy, cpu_loc));
  try {
    if constexpr (std::is_scalar_v<std::remove_cv_t<T>>) {
      *ptr = T{std::forward<Args>(args)...};
    } else {
      ptr = new (ptr) T{std::forward<Args>(args)...};
    }
  } catch (...) {
    CudaCheck(cudaFree(const_cast<std::remove_cv_t<T> *>(ptr)));
    throw;
  }
  return ManagedObjectPtr<T>{ptr};
}

template <typename T, typename S, std::enable_if_t<std::is_same_v<T, std::remove_cv_t<S>>, bool> = false>
void Copy(T *tgt, S *src, std::size_t size, cudaStream_t stream) {
  CudaCheck(cudaMemcpyAsync(tgt, src, size * sizeof(T), cudaMemcpyDefault, stream));
}

template <typename T, typename D>
void Copy(std::unique_ptr<T[], D> &tgt, T const *src, std::size_t size, cudaStream_t stream) {
  Copy(tgt.get(), src, size, stream);
}

template <typename T, typename D>
void Copy(T *tgt, std::unique_ptr<T[], D> const &src, std::size_t size, cudaStream_t stream) {
  Copy(tgt, src.get(), size, stream);
}

template <typename T, typename S, std::enable_if_t<std::is_same_v<T, std::remove_cv_t<S>>, bool> = false>
void Copy2D(T *tgt,
            std::size_t tgt_pitch,
            S *src,
            std::size_t src_pitch,
            std::size_t num_rows,
            std::size_t num_cols,
            cudaStream_t stream) {
  CudaCheck(
      cudaMemcpy2DAsync(tgt, tgt_pitch, src, src_pitch, num_cols * sizeof(T), num_rows, cudaMemcpyDefault, stream));
}

template <typename T>
void Copy2D(Array2D<T> &tgt,
            typename Array2D<T>::ValueType const *src,
            std::size_t src_pitch,
            std::size_t num_rows,
            std::size_t num_cols,
            cudaStream_t stream) {
  Copy2D(tgt.Get(), tgt.Pitch(), src, src_pitch, num_rows, num_cols, stream);
}

template <typename T>
void Copy2D(Array2D<T> &tgt,
            typename Array2D<T>::ValueType const *src,
            std::size_t num_rows,
            std::size_t num_cols,
            cudaStream_t stream) {
  Copy2D(tgt, src, num_cols * Array2D<T>::kValueSize, num_rows, num_cols, stream);
}

template <typename T>
void Copy2D(typename Array2D<T>::ValueType *tgt,
            std::size_t tgt_pitch,
            Array2D<T> const &src,
            std::size_t num_rows,
            std::size_t num_cols,
            cudaStream_t stream) {
  Copy2D(tgt, tgt_pitch, src.Get(), src.Pitch(), num_rows, num_cols, stream);
}

template <typename T>
void Copy2D(typename Array2D<T>::ValueType *tgt,
            Array2D<T> const &src,
            std::size_t num_rows,
            std::size_t num_cols,
            cudaStream_t stream) {
  Copy2D(tgt, num_cols * Array2D<T>::kValueSize, src, num_rows, num_cols, stream);
}

void FloatToHalf(half *dst, float const *src, std::size_t size);

template <typename TAlloc = std::vector<half>::allocator_type>
std::vector<half, TAlloc> FloatToHalf(float const *src, std::size_t size) {
  std::vector<half, TAlloc> dst(size);
  FloatToHalf(dst.data(), src, size);
  return dst;
}

void inline CopyTransposeConvert(
    half *dst, float const *src, std::uint32_t batch_size, std::uint32_t num_steps, std::uint32_t num_features) {
  auto const num_batch_features = batch_size * num_features;
  for (std::uint32_t i = 0; i < num_steps; ++i) {
    FloatToHalf(dst + i * num_batch_features, src + i * num_features, num_features);
  }
}

struct alignas(4) TensorFloat32 {
  float value;

  __device__ __forceinline__ static TensorFloat32 Round(float const a) {
    float ret;
    asm("{\n  .reg .b32 __$1;"
        "\n   cvt.rna.tf32.f32 __$1, %1;"
        "\n   mov.b32 %0, __$1;\n}\n"
        : "=f"(ret)
        : "f"(a));
    return {ret};
  }

  __host__ __device__ explicit operator float() const { return value; }
};

template <typename T>
class PinnedAllocator {
 public:
  using value_type = T;
  using pointer = value_type *;
  using size_type = std::size_t;

  PinnedAllocator() noexcept = default;

  template <typename U>
  explicit PinnedAllocator(PinnedAllocator<U> const &) noexcept {}

  T *allocate(size_type n) {
    T *tmp;
#if 0 
// AARCH64_GH == 1
    tmp = (T*) std::malloc(n * sizeof(T));
#else
    CudaCheck(cudaHostAlloc(&tmp, n * sizeof(T), cudaHostAllocDefault));
#endif
    return tmp;
  }

  void deallocate(T *p, size_type n) {
#if 0
//AARCH64_GH == 1
    if (p) std::free(p);
#else
    if (p) CudaCheck(cudaFreeHost(p));
#endif
  }
};

template <class T, class U>
auto operator==(PinnedAllocator<T> const &, PinnedAllocator<U> const &) -> bool {
  return true;
}

template <class T, class U>
auto operator!=(PinnedAllocator<T> const &, PinnedAllocator<U> const &) -> bool {
  return false;
}

using PinnedVector16 = std::vector<half, PinnedAllocator<half>>;
using PinnedTensor16 = util::Tensor<PinnedVector16>;
using PinnedVector32 = std::vector<float, PinnedAllocator<float>>;
using PinnedTensor32 = util::Tensor<PinnedVector32>;

template <typename T, typename U>
auto constexpr CeilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

class CudaStreamDeleter {
 public:
  constexpr void operator()(cudaStream_t stream) const {
    if (stream != nullptr) CudaCheck(cudaStreamDestroy(stream));
  }
};

using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

inline StreamPtr CreateStream(unsigned int flags = cudaStreamNonBlocking, int priority = 0) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreateWithPriority(&stream, flags, priority));
  return StreamPtr{stream};
}

class CudaEventDeleter {
 public:
  constexpr void operator()(cudaEvent_t stream) const {
    if (stream != nullptr) CudaCheck(cudaEventDestroy(stream));
  }
};

using EventPtr = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, CudaEventDeleter>;

inline EventPtr CreateEvent(unsigned int flags = cudaEventDisableTiming) {
  cudaEvent_t event;
  CudaCheck(cudaEventCreate(&event, flags));
  return EventPtr{event};
}

inline bool EventReady(cudaEvent_t event) {
  auto success = cudaEventQuery(event);
  if (success == cudaErrorNotReady) return false;
  if (success == cudaSuccess) return true;
  CudaCheck(success);
  return false;
}

template <typename TData>
TData *ConstZero();

template <typename TData>
TData *ConstOne();

template <typename TData>
std::vector<TData, PinnedAllocator<TData>> Fetch(TData const *ptr, std::size_t size, cudaStream_t stream) {
  std::vector<TData, cex::PinnedAllocator<TData>> result(size);
  cex::Copy(result.data(), ptr, size, stream);
  CudaCheck(cudaStreamSynchronize(stream));
  return result;
}

template <typename TData>
std::vector<TData, PinnedAllocator<TData>> Fetch(DevicePtr<TData> const &ptr, std::size_t size, cudaStream_t stream) {
  return Fetch(ptr.get(), size, stream);
}

class SingleProducerSingleConsumer;

namespace spsc {

void Store(bool volatile *flag, bool value, cudaStream_t stream);

void WaitUntil(bool volatile &flag, bool value, cudaStream_t stream);

namespace {

template <typename TSemaphore>
__global__ __launch_bounds__(1) void Acquire_ker(TSemaphore &semaphore) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    semaphore.acquire();
  }
}

template <typename TSemaphore>
__global__ __launch_bounds__(1) void Release_ker(TSemaphore &semaphore, std::ptrdiff_t update) {
  if (threadIdx.x == 0 && blockIdx.x == 0) semaphore.release(update);
}

}  // namespace

template <typename TSemaphore>
void Acquire(TSemaphore &semaphore, cudaStream_t stream) {
  CudaCheck(cudaGetLastError());  // reset errors
  Acquire_ker<<<1, 1, 0, stream>>>(semaphore);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

template <typename TSemaphore>
void Release(TSemaphore &semaphore, cudaStream_t stream, std::ptrdiff_t update = 1) {
  CudaCheck(cudaGetLastError());  // reset errors
  Release_ker<<<1, 1, 0, stream>>>(semaphore, update);
  CudaCheck(cudaGetLastError());  // detect synchronous errors
}

}  // namespace spsc

class Event {
 public:
  __device__ __host__ explicit Event(bool ready = false) : ready_{ready} {}

  [[nodiscard]] __device__ __host__ bool IsReady(
      cuda::std::memory_order mo = cuda::std::memory_order_acquire) const noexcept {
    return ready_.load(mo);
  }

  __host__ bool TryReset(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    bool expect = true;
    return ready_.compare_exchange_weak(expect, false, mo);
  }

  __host__ void WaitReset(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    while (!TryReset(mo)) {
      // wait
    }
  }

  __host__ void WaitReset(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_acq_rel);

  __device__ __host__ void Notify(cuda::std::memory_order mo = cuda::std::memory_order_release) {
    ready_.store(true, mo);
  }

  __host__ void Notify(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_release);

  [[nodiscard]] __device__ __host__ bool IsLockFree() const { return ready_.is_lock_free(); }

 private:
  using cuda_atomic_bool = cuda::atomic<bool, cuda::thread_scope::thread_scope_system>;
  cuda_atomic_bool ready_;
};

class SingleProducerSingleConsumer {
 public:
  __device__ __host__ explicit SingleProducerSingleConsumer(bool empty = true, bool full = false)
      : empty_{empty}, full_{full} {}

  [[nodiscard]] __device__ __host__ bool CanProduce(
      cuda::std::memory_order mo = cuda::std::memory_order_acquire) const noexcept {
    return empty_.IsReady(mo);
  }

  __host__ bool TryProduce(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    return empty_.TryReset(mo);
  }

  __host__ void WaitToProduce(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    empty_.WaitReset(mo);
  }

  __host__ void WaitToProduce(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    empty_.WaitReset(stream, mo);
  }

  __device__ __host__ void NotifyProduced(cuda::std::memory_order mo = cuda::std::memory_order_release) {
    full_.Notify(mo);
  }

  __host__ void NotifyProduced(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_release) {
    full_.Notify(stream, mo);
  }

  [[nodiscard]] __device__ __host__ bool CanConsume(
      cuda::std::memory_order mo = cuda::std::memory_order_acquire) const noexcept {
    return full_.IsReady(mo);
  }

  __host__ bool TryConsume(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    return full_.TryReset(mo);
  }

  __host__ void WaitToConsume(cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    full_.WaitReset(mo);
  }

  __host__ void WaitToConsume(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_acq_rel) {
    full_.WaitReset(stream, mo);
  }

  __device__ __host__ void NotifyConsumed(cuda::std::memory_order mo = cuda::std::memory_order_release) {
    empty_.Notify(mo);
  }

  __host__ void NotifyConsumed(cudaStream_t stream, cuda::std::memory_order mo = cuda::std::memory_order_release) {
    empty_.Notify(stream, mo);
  }

  [[nodiscard]] __device__ __host__ bool IsLockFree() const { return empty_.IsLockFree() && full_.IsLockFree(); }

 private:
  Event empty_;
  Event full_;
};

}  // namespace nvidia::quant::lowlat_infer::cex

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_CUDA_HELPERS_CUH_
