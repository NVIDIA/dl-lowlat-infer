/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// clang-format off

// on aarch64 we have to set it manually
#if defined(__x86_64__) && not defined(__pcie__)
#define __pcie__
#endif

#if not defined(DISABLE_ATOMIC_EXCHANGE)

#if defined(__pcie__)
#define DISABLE_ATOMIC_EXCHANGE 1
#define SIGNAL_SWITCH_INIT_VALUE 1
#else
#define DISABLE_ATOMIC_EXCHANGE 0
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define AARCH64_GH 1

#define GH_PAGED_INPUT 0

// defined in util.cpp
// can not inline, since neon is not supported in nvcc
void _mm_stream_si64(long long *p, long long a);
void _mm_sfence(void);
#else
#define AARCH64_GH 0
#endif

// Performance hints
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Runtime assertions
#define StateCheck(state)                                             \
  do {                                                                \
    nvidia::quant::lowlat_infer::util::StateCheck_((state), #state, __FILE__, __LINE__); \
  } while (0)

// Check for system calls.
#define SystemCheck(rc)                                          \
  do {                                                           \
    nvidia::quant::lowlat_infer::util::SystemCheck_((rc), #rc, __FILE__, __LINE__); \
  } while (0)

namespace nvidia::quant::lowlat_infer::util {

void StateThrow(bool state, const char *msg, const char *file, int line);

void inline StateCheck_(bool state, const char *msg, const char *file, int line) {
  if (!state) StateThrow(state, msg, file, line);
}

void SystemThrow(int rc, const char *msg, const char *file, int line);

void inline SystemCheck_(int rc, const char *msg, const char *file, int line) {
  if (rc) SystemThrow(rc, msg, file, line);
}

using ShapeVector = std::vector<std::size_t>;

using Vector32 = std::vector<float>;

Vector32 LoadData(char const *filename, ShapeVector *shape);

inline Vector32 LoadData(std::filesystem::path const &path, ShapeVector *shape) {
  return LoadData(path.c_str(), shape);
}

template <typename Scalar>
[[maybe_unused]] std::vector<Scalar> LoadVector(char const *filename);

template <typename Scalar>
inline std::vector<Scalar> LoadVector(std::filesystem::path const &path) {
  return LoadVector<Scalar>(path.c_str());
}

void SaveData(char const *filename, ShapeVector const &shape, Vector32 const &data);

void SaveData(char const *filename, ShapeVector const &shape, std::vector<double> const &data);

template <typename T, typename A>
std::vector<T, A> &EnsureCapacity(std::vector<T, A> &x, typename std::vector<T, A>::size_type capacity) {
  if (x.capacity() < capacity) {
    auto const size = x.size();
    x.insert(x.end(), capacity - size, T{});
    x.resize(size);
  }
  return x;
}

template <typename T, typename A, typename... Args>
std::vector<T, A> &Append(std::vector<T, A> &x, T &&first, Args &&...args) {
  x.push_back(std::forward<T>(first));
  if constexpr (sizeof...(Args) > 0) {
    return Append(x, std::forward<Args>(args)...);
  } else {
    return x;
  }
}

template <typename T, typename A = std::allocator<T>, typename... Args>
std::vector<T, A> MakeVector(T &&first, Args &&...args) {
  std::vector<T, A> result;
  result.reserve(1 + sizeof...(Args));
  Append(result, std::forward<T>(first), std::forward<T>(args)...);
  return result;
}

template <typename TShape, typename TData>
void CheckShape(TShape const &shape, TData const &data) {
  StateCheck(!shape.empty());

  typename TShape::value_type total{1};
  for (auto const s : shape) total *= s;
  StateCheck(data.size() == total);
}

template <typename TVector = Vector32, typename TShape = ShapeVector>
class Tensor {
 public:
  using VectorType = TVector;
  using ShapeType = TShape;
  using value_type = typename VectorType::value_type;

  static_assert(std::is_same_v<typename ShapeType::value_type, std::size_t>);

  Tensor() = default;

  Tensor(VectorType &&data, ShapeType &&shape)
      : data_{std::make_shared<VectorType>(std::move(data))}, shape_{std::move(shape)} {
    CheckShape(shape_, *data_);
  }

  // no copying
  Tensor(Tensor const &) = delete;
  Tensor &operator=(Tensor const &) = delete;

  // moving
  Tensor(Tensor &&tensor) noexcept : data_{std::move(tensor.data_)}, shape_{std::move(tensor.shape_)} {};
  Tensor &operator=(Tensor &&tensor) noexcept {
    this->data_ = std::move(tensor.data_);
    this->shape_ = std::move(tensor.shape_);
    return *this;
  }

  VectorType &Data() { return *data_; }
  [[nodiscard]] VectorType const &Data() const { return *data_; }

  ShapeType &Shape() { return shape_; }
  [[nodiscard]] ShapeType const &Shape() const { return shape_; }

  std::shared_ptr<VectorType> DataShared() { return data_; }
  [[nodiscard]] std::shared_ptr<VectorType const> DataShared() const { return data_; }

  static Tensor<> Load(char const *filename) {
    ShapeVector shape;
    auto data = LoadData(filename, &shape);
    return Tensor{std::move(data), std::move(shape)};
  }

  static Tensor<> Load(std::filesystem::path const &filename) { return Load(filename.c_str()); }

  void Save(char const *filename) { SaveData(filename, shape_, data_); }

  void Save(std::filesystem::path const &filename) { Save(filename.c_str()); }

  template <typename TVectorOther>
  Tensor<TVectorOther, TShape> WithData(TVectorOther &&data) const {
    return Tensor<TVectorOther, TShape>{std::forward<TVectorOther>(data), ShapeVector{shape_}};
  }

  template <typename DataAlloc>
  Tensor<std::vector<value_type, DataAlloc>, ShapeType> Reallocate() const {
    return this->WithData(std::vector<value_type, DataAlloc>{data_->begin(), data_->end()});
  }

 private:
  std::shared_ptr<VectorType> data_;
  ShapeType shape_;
};

using Tensor32 = Tensor<>;

unsigned int Log2(std::uint64_t n);

// clang-format on

/**
 * Interface for classes to sample from a dataset.
 */
class DataSampler {
 public:
  virtual ~DataSampler() = default;

  /**
   * Return the next samples to select
   * Returns a value between `0` and `size() - 1`
   */
  [[nodiscard]] virtual std::uint64_t Next() = 0;

  /**
   * Return the size of the data to be sampled from
   */
  [[nodiscard]] virtual std::size_t Size() const = 0;

  /**
   * Reset to start of sampler
   */
  virtual void Reset() = 0;
};

class RingSampler : public DataSampler {
 public:
  RingSampler(std::uint64_t seed, std::uint64_t instance, std::size_t size);

  ~RingSampler() = default;

  [[nodiscard]] std::uint64_t Next() final override;

  [[nodiscard]] std::size_t Size() const final override;

  void Reset() final override;

 private:
  std::uint64_t start_;
  std::uint64_t next_;
  std::size_t size_;
};

class RandomSampler : public DataSampler {
 public:
  RandomSampler(std::uint64_t seed, std::uint64_t instance, std::size_t size);

  ~RandomSampler() = default;

  [[nodiscard]] std::uint64_t Next() final override;

  [[nodiscard]] std::size_t Size() const final override;

  void Reset() final override;

 private:
  std::uint64_t start_;
  std::size_t size_;
  std::minstd_rand rng_;
  std::uniform_int_distribution<std::uint64_t> dist_;
};

// clang-format off

using Clock = std::chrono::steady_clock;

template <typename TFunc>
Clock::duration TimeIt(std::size_t num_iterations, TFunc func) {
  auto ts_start = Clock::now();
  for (std::size_t i = 0; i < num_iterations; ++i) {
    func();
  }
  return Clock::now() - ts_start;
}

template <typename T>
void CopyTranspose(
    T *dst, T const *src, std::uint32_t batch_size, std::uint32_t num_steps, std::uint32_t num_features) {
  auto const num_batch_features = batch_size * num_features;
  auto const input_size_bytes = num_features * sizeof(T);
  for (std::uint32_t i = 0; i < num_steps; ++i) {
    std::memcpy(dst + i * num_batch_features, src + i * num_features, input_size_bytes);
  }
}

template <typename T>
std::vector<T> ParseList(std::string const &s, char const delimiter = ',') {
  std::vector<T> list{};
  std::stringstream ss(s);

  for (T i; ss >> i;) {
    list.push_back(i);
    if (ss.peek() == delimiter) ss.ignore();
  }

  return list;
}

template <typename T>
std::string ToString(T const &container,
                     std::string const &delimiter = ", ",
                     std::string const &start = "[",
                     std::string const &end = "]") {
  std::stringstream ss{};
  bool is_first = true;
  ss << start;
  for (auto &x : container) {
    if (is_first) {
      is_first = false;
    } else {
      ss << delimiter;
    }
    ss << x;
  }
  ss << end;
  return ss.str();
}

using CpuSet = std::vector<std::uint16_t>;

void SetAffinity(std::thread::native_handle_type thread, CpuSet const &cups);

void inline SetAffinity(std::thread::native_handle_type thread, std::uint16_t core) {
  SetAffinity(thread, CpuSet{core});
}

CpuSet GetAffinity(std::thread::native_handle_type thread);

class Semaphore {
 public:
  explicit Semaphore(unsigned int count = 0) : count_{count} {}

  void release(unsigned int update = 1, std::memory_order mo = std::memory_order_acq_rel) {
    count_.fetch_add(update, mo);
  }

  void acquire(std::memory_order mo = std::memory_order_acq_rel) {
    while (!try_acquire(mo)) {
      // Spin Locking
    }
  }

  bool try_acquire(std::memory_order mo = std::memory_order_acq_rel) noexcept {
    auto count = count_.load(mo == std::memory_order_acq_rel ? std::memory_order_acquire : mo);
    if (count) {
      return count_.compare_exchange_strong(count, count - 1, mo);
    } else {
      return false;
    }
  }

  [[nodiscard]] bool can_acquire(std::memory_order mo = std::memory_order_acquire) const { return count_.load(mo); }

  [[nodiscard]] bool is_lock_free() const { return count_.is_lock_free(); }

  static unsigned int max() { return std::numeric_limits<unsigned int>::max(); }

 private:
  std::atomic_uint count_;
};

class SingleProducerSingleConsumer {
 public:
  explicit SingleProducerSingleConsumer(unsigned int empty = 0, unsigned int full = 0) : empty_{empty}, full_{full} {}

  [[nodiscard]] bool CanProduce(std::memory_order mo = std::memory_order_acquire) const noexcept {
    return empty_.can_acquire();
  }

  [[nodiscard]] bool CanConsume(std::memory_order mo = std::memory_order_acquire) const noexcept {
    return full_.can_acquire();
  }

  bool TryProduce(std::memory_order mo = std::memory_order_acq_rel) { return empty_.try_acquire(mo); }

  void NotifyProduced(std::memory_order mo = std::memory_order_release) { full_.release(1, mo); }

  bool TryConsume(std::memory_order mo = std::memory_order_acq_rel) { return full_.try_acquire(mo); }

  void NotifyConsumed(std::memory_order mo = std::memory_order_release) { empty_.release(1, mo); }

  [[nodiscard]] bool IsLockFree() const { return empty_.is_lock_free() && full_.is_lock_free(); }

 private:
  Semaphore empty_;
  Semaphore full_;
};

}  // namespace nvidia::quant::lowlat_infer::util
