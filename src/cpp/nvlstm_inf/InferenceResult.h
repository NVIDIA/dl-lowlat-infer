/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NVLSTMINF_INFERENCE_RESULTS_H
#define NVLSTMINF_INFERENCE_RESULTS_H

#include <chrono>
#include <cstdint>
#include <vector>

namespace nvidia::quant::lowlat_infer {

class InferenceResult {
 public:
  using Clock = std::chrono::steady_clock;

  static_assert(Clock::is_steady);

  static auto constexpr kDefaultCapacity = 1 << 20;

  explicit InferenceResult(std::size_t init_capacity = kDefaultCapacity);

  // no copy
  InferenceResult(InferenceResult const &) = delete;
  InferenceResult &operator=(InferenceResult const &) = delete;

  void Add(float output, Clock::time_point const &ts_start, Clock::time_point const &ts_stop) {
    output_.push_back(output);
    ts_start_.push_back(ts_start);
    ts_stop_.push_back(ts_stop);
  }

  void AddBatch(
      std::uint32_t batch_size,
      float const *output,
      Clock::time_point const *ts_start,
      Clock::time_point const &ts_stop);

  void Reserve(std::uint32_t batch_size);

  void Save(char const *filename, Clock::time_point const &base) const;

  void Reset();

  [[nodiscard]] std::size_t Size() const { return output_.size(); }
  [[nodiscard]] Clock::duration Duration() const;
  [[nodiscard]] Clock::duration P99() const;
  [[nodiscard]] Clock::duration P50() const;

  static double ToFracSeconds(Clock::time_point const &time_point, Clock::time_point const &base);

  [[nodiscard]] auto const &Output() const { return output_; }
  [[nodiscard]] auto const &TsStart() const { return ts_start_; }
  [[nodiscard]] auto const &TsStop() const { return ts_stop_; }

 private:
  std::vector<float> output_;
  std::vector<Clock::time_point> ts_start_;
  std::vector<Clock::time_point> ts_stop_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif
