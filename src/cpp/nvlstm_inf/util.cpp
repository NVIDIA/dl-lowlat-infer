/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "util.h"

#include <cstdio>
#include <cstring>
#include <npy.h>
#include <pthread.h>
#include <stdexcept>
#include <vector>

using namespace nvidia::quant::lowlat_infer::util;

#if AARCH64_GH == 1
#include <arm_neon.h>

// Store 64-bit integer a into memory using a non-temporal hint to minimize
// cache pollution. If the cache line containing address mem_addr is already in
// the cache, the cache will be updated.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_stream_si64
void _mm_stream_si64(long long *p, long long a) { vst1_s64((int64_t *)p, vdup_n_s64((int64_t)a)); }

// Perform a serializing operation on all store-to-memory instructions that were
// issued prior to this instruction. Guarantees that every store instruction
// that precedes, in program order, is globally visible before any store
// instruction which follows the fence in program order.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sfence
void _mm_sfence(void) {
  do {
    __asm__ __volatile__("" ::: "memory");
    (void)0;
  } while (0);
  std::atomic_thread_fence(std::memory_order_seq_cst);
}
#endif

namespace {

template <typename T>
void SaveChecked(const char *filename, ShapeVector const &shape, std::vector<T> const &data) {
  CheckShape(shape, data);
  auto constexpr fortran_order = false;
  npy::SaveArrayAsNumpy(filename, fortran_order, shape.size(), shape.data(), data);
}
}  // namespace

namespace nvidia::quant::lowlat_infer::util {

void StateThrow(bool state, const char *msg, const char *file, int line) {
  if (!state) {
    char buf[1024];
    std::snprintf(buf, std::size(buf), "Invalid state: %s in %s:%d", msg, file, line);
    throw std::runtime_error(buf);
  }
}

void SystemThrow(int rc, const char *msg, const char *file, int line) {
  if (rc) {
    char buf[1024];
    std::snprintf(buf, std::size(buf), "System call failed: %s = %s in %s:%d", msg, strerror(rc), file, line);
    throw std::runtime_error(buf);
  }
}

std::vector<float> LoadData(char const *filename, ShapeVector *shape) {
  std::vector<float> data;
  bool fortran_order;
  npy::LoadArrayFromNumpy(filename, *shape, fortran_order, data);
  StateCheck(!fortran_order);
  return data;
}

template <typename Scalar>
std::vector<Scalar> LoadVector(char const *filename) {
  std::vector<Scalar> data;
  ShapeVector shape;
  bool fortran_order;
  npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);
  StateCheck(!fortran_order);
  return data;
}

template std::vector<std::int32_t> LoadVector(char const *filename);

template std::vector<std::uint32_t> LoadVector(char const *filename);

template std::vector<float> LoadVector(char const *filename);

template std::vector<double> LoadVector(char const *filename);

void SaveData(const char *filename, ShapeVector const &shape, Vector32 const &data) {
  SaveChecked(filename, shape, data);
}

void SaveData(const char *filename, ShapeVector const &shape, std::vector<double> const &data) {
  SaveChecked(filename, shape, data);
}

unsigned int Log2(std::uint64_t n) {
  unsigned int result = 0;
  while (n > 1) {
    n >>= 1;
    ++result;
  }
  return result;
}

RingSampler::RingSampler(std::uint64_t seed, std::uint64_t instance, std::size_t size)
    : start_{(seed + instance) % size}, next_{start_}, size_{size} {}

[[nodiscard]] std::uint64_t RingSampler::Next() {
  std::uint64_t r = next_;
  next_ = (next_ + 1) % size_;
  return r;
}

[[nodiscard]] std::size_t RingSampler::Size() const { return size_; }

void RingSampler::Reset() { next_ = start_; }

RandomSampler::RandomSampler(std::uint64_t seed, std::uint64_t instance, std::size_t size)
    : start_{seed + instance}, size_{size}, rng_{start_}, dist_{0, size_ - 1} {}

[[nodiscard]] std::uint64_t RandomSampler::Next() { return dist_(rng_); }

[[nodiscard]] std::size_t RandomSampler::Size() const { return size_; }

void RandomSampler::Reset() { rng_.seed(start_); }

void SetAffinity(std::thread::native_handle_type thread, CpuSet const &cpus) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  for (auto c : cpus) {
    CPU_SET(c, &cpu_set);
  }
  if (thread == 0) {
    thread = pthread_self();
  }
  SystemCheck(pthread_setaffinity_np(thread, sizeof(cpu_set), &cpu_set));
}

CpuSet GetAffinity(std::thread::native_handle_type thread) {
  nvidia::quant::lowlat_infer::util::CpuSet result{};
  if (thread == 0) {
    thread = pthread_self();
  }
  cpu_set_t cpu_set;
  SystemCheck(pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpu_set));
  for (CpuSet::value_type i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpu_set)) result.push_back(i);
  }
  return result;
}

}  // namespace nvidia::quant::lowlat_infer::util
