/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "InferenceResult.h"

#include <cassert>
#include <chrono>

#include "util.h"

namespace nvidia::quant::lowlat_infer {

InferenceResult::InferenceResult(std::size_t init_capacity) {
  output_.reserve(init_capacity);
  ts_start_.reserve(init_capacity);
  ts_stop_.reserve(init_capacity);
}

void InferenceResult::Save(char const *filename, Clock::time_point const &base) const {
  auto const size = Size();
  util::ShapeVector shape{size, 3};
  std::vector<double> data{};
  data.reserve(shape[0] * shape[1]);
  for (std::size_t i = 0; i < size; ++i) {
    data.push_back(output_[i]);
    data.push_back(ToFracSeconds(ts_start_[i], base));
    data.push_back(ToFracSeconds(ts_stop_[i], base));
  }
  util::SaveData(filename, shape, data);
}

void InferenceResult::AddBatch(std::uint32_t batch_size,
                               float const *output,
                               Clock::time_point const *ts_start,
                               Clock::time_point const &ts_stop) {
  output_.insert(output_.end(), output, output + batch_size);
  ts_start_.insert(ts_start_.end(), ts_start, ts_start + batch_size);
  ts_stop_.insert(ts_stop_.end(), batch_size, ts_stop);
}

double InferenceResult::ToFracSeconds(Clock::time_point const &time_point, Clock::time_point const &base) {
  auto const nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(time_point - base).count();
  return static_cast<double>(nanos) * 1e-9;
}

void InferenceResult::Reserve(std::uint32_t batch_size) {
  auto const capacity = Size() + batch_size;
  if (unlikely(output_.capacity() < capacity)) {
    util::EnsureCapacity(output_, capacity);
    util::EnsureCapacity(ts_start_, capacity);
    util::EnsureCapacity(ts_stop_, capacity);
  }
}

InferenceResult::Clock::duration InferenceResult::Duration() const { 
  std::chrono::steady_clock::duration total{};
  auto const size = Size();
  for (std::size_t i = 0; i < size; ++i) {
    total += ts_stop_[i] - ts_start_[i];
  }
  return total;
}

InferenceResult::Clock::duration InferenceResult::P99() const { 
  std::chrono::steady_clock::duration total{};
  auto const size = Size();
  std::vector<std::chrono::steady_clock::duration> d(size);
  for (std::size_t i = 0; i < size; ++i) {
    d[i] = ts_stop_[i] - ts_start_[i];
  }
  std::sort(d.begin(), d.end(), [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(a).count() 
        <  std::chrono::duration_cast<std::chrono::nanoseconds>(b).count();
  });
  int idxp99 = int(99.f / 100.f * size);
  return d[idxp99];
}

void InferenceResult::Reset() {
  output_.clear();
  ts_start_.clear();
  ts_stop_.clear();
}

}  // namespace nvidia::quant::lowlat_infer
