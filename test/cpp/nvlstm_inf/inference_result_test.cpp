/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>

#include "nvlstm_inf/InferenceResult.h"
#include "nvlstm_inf/util.h"

class InferenceResultF : public ::testing::Test {
 protected:
  void SetUp() override {
    auto dir_name = (std::filesystem::temp_directory_path() / "inference_test_XXXXXX").string();
    auto dir_name_c = dir_name.data();
    dir_name_c = mkdtemp(dir_name_c);
    ASSERT_TRUE(dir_name_c);
    tmpdir = std::filesystem::path{dir_name_c};
  }

  void TearDown() override { std::filesystem::remove_all(tmpdir); }

  std::filesystem::path tmpdir;
};

using namespace nvidia::quant::lowlat_infer;

TEST(InferenceResult, Add) {
  InferenceResult result{1 << 8};
  EXPECT_EQ(result.Size(), 0);
  using Clock = InferenceResult::Clock;
  auto const start = Clock::now();
  auto const stop = Clock::now();
  result.Add(1.0, start, stop);
  EXPECT_EQ(result.Size(), 1);
  EXPECT_FLOAT_EQ(result.Output().at(0), 1.0);
  auto const nanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(result.TsStop().at(0) - result.TsStart().at(0)).count();
  EXPECT_GT(nanos, 0);
  auto constexpr max_nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();
  EXPECT_LT(nanos, max_nanos);

  auto constexpr batch_size = 4;
  std::vector<float> output(batch_size, 2.0);
  std::vector<Clock::time_point> ts_start(batch_size, start);
  result.AddBatch(batch_size, output.data(), ts_start.data(), stop);
  EXPECT_EQ(result.Size(), 1 + batch_size);
  for (std::size_t i = 1; i < result.Size(); ++i) {
    EXPECT_FLOAT_EQ(result.Output().at(i), 2.0);
    EXPECT_EQ(result.TsStart().at(i), start);
    EXPECT_EQ(result.TsStop().at(i), stop);
  }
}

TEST(InferenceResult, P99RejectsEmptyResult) {
  InferenceResult result{0};
  EXPECT_THROW(static_cast<void>(result.P99()), std::runtime_error);
}

TEST(InferenceResult, P99UsesNearestRank) {
  auto check_p99 = [](int sample_count, int expected_nanos) {
    using Clock = InferenceResult::Clock;
    auto const base = Clock::time_point{};
    InferenceResult result{static_cast<std::size_t>(sample_count)};
    for (int nanos = sample_count - 1; nanos >= 0; --nanos) {
      result.Add(0.0F, base, base + std::chrono::nanoseconds{nanos});
    }
    EXPECT_EQ(result.P99(), std::chrono::nanoseconds{expected_nanos});
  };

  check_p99(1, 0);
  check_p99(100, 98);
  check_p99(101, 99);
}

TEST(InferenceResult, ToFracSeconds) {
  using Clock = InferenceResult::Clock;
  auto const base = Clock::now();
  auto const start = InferenceResult::ToFracSeconds(Clock::now(), base);
  auto const stop = InferenceResult::ToFracSeconds(Clock::now(), base);
  auto const secs = stop - start;
  EXPECT_GT(secs, 0);
  EXPECT_LT(secs, 1);
}

TEST_F(InferenceResultF, ToFracSeconds) {
  using Clock = InferenceResult::Clock;
  auto const base = Clock::now();
  auto const start = Clock::now();
  auto const stop = Clock::now();
  InferenceResult result{1};
  result.Add(1.0, start, stop);
  auto const filename = tmpdir / "result.npy";
  result.Save(filename.c_str(), base);
  auto const data = util::LoadVector<double>(filename.c_str());
  EXPECT_EQ(data.size(), 3);
  EXPECT_DOUBLE_EQ(data[0], 1.0);
  auto const secs = data[2] - data[1];
  EXPECT_GT(secs, 0);
  EXPECT_LT(secs, 1);
}
