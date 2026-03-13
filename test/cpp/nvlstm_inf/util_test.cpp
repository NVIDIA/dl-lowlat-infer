/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <nvlstm_inf/util.h>

#include <filesystem>
#include <vector>

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer::util;

TEST(UtilTest, StateCheck) {
  EXPECT_NO_THROW(StateCheck(true));
  EXPECT_THROW(StateCheck(false), std::runtime_error);
}

TEST(UtilTest, LoadData) {
  ShapeVector shape;
  EXPECT_ANY_THROW(LoadData("not_existing.npy", &shape));
  auto path = std::filesystem::path(TEST_RESOURCES_DIR);
  path /= "lstm_s.npy";
  ASSERT_TRUE(std::filesystem::exists(path));
  auto const data = LoadData(path.c_str(), &shape);
  EXPECT_EQ(shape.size(), 2);
  auto constexpr n_rows = 65664;
  EXPECT_EQ(shape[0], n_rows);
  auto constexpr n_cols = 128;
  EXPECT_EQ(shape[1], n_cols);
  EXPECT_EQ(data.size(), n_rows * n_cols);
  EXPECT_FLOAT_EQ(data[0], 1.3001157);
  EXPECT_FLOAT_EQ(data[n_cols - 1], 1.9756641);
  // row major layout
  EXPECT_FLOAT_EQ(data[42 * n_cols + 28], 1.4296954);
}

TEST(UtilTest, Tensor32) {
  auto path = std::filesystem::path(TEST_RESOURCES_DIR);
  path /= "lstm_s.npy";
  ASSERT_TRUE(std::filesystem::exists(path));
  auto tensor = Tensor32::Load(path.c_str());
  auto &shape = tensor.Shape();
  EXPECT_EQ(shape.size(), 2);
  auto constexpr n_rows = 65664;
  EXPECT_EQ(shape[0], n_rows);
  auto constexpr n_cols = 128;
  EXPECT_EQ(shape[1], n_cols);
  EXPECT_EQ(tensor.Data().size(), n_rows * n_cols);
}

TEST(UtilTest, RandomSampler) {
  std::uint64_t constexpr seed{100};
  RandomSampler seq{seed, 1, 48000};

  EXPECT_EQ(seq.Size(), 48000);

  // See Random Sequence implementation in util.h
  std::array expected_samples = {
      108,
      28237,
      35357,
      2531,
      36649,
      7616,
      602,
      9544,
  };

  for (std::size_t i = 0; i < expected_samples.size(); ++i) {
    EXPECT_EQ(seq.Next(), expected_samples[i]);
  }
  seq.Reset();
  for (std::size_t i = 0; i < expected_samples.size(); ++i) {
    EXPECT_EQ(seq.Next(), expected_samples[i]);
  }
}

TEST(UtilTest, EnsureCapacity) {
  std::size_t constexpr size_init = 31;
  std::vector<int> x(size_init, 1);
  EXPECT_EQ(x.size(), size_init);
  EXPECT_EQ(x.capacity(), size_init);
  auto constexpr capacity = size_init + size_init / 2;
  EnsureCapacity(x, capacity);
  EXPECT_EQ(x.capacity(), 2 * size_init);
  EXPECT_EQ(x.size(), size_init);
}

TEST(UtilTest, MakeVector) {
  EXPECT_EQ(MakeVector(1).size(), 1);
  EXPECT_EQ(MakeVector(1, 2, 3, 4).size(), 4);
  EXPECT_EQ(MakeVector(1, 2, 3, 4).at(3), 4);
  EXPECT_EQ(MakeVector(Tensor32{}).size(), 1);
  EXPECT_EQ(MakeVector(Tensor32{}, Tensor32{}).size(), 2);
}

TEST(UtilTest, SingleProducerSingleConsumer) {
  SingleProducerSingleConsumer spsc{0, 0};
  EXPECT_TRUE(spsc.IsLockFree());
  EXPECT_FALSE(spsc.CanProduce());
  EXPECT_FALSE(spsc.CanConsume());
  EXPECT_FALSE(spsc.TryProduce());
  EXPECT_FALSE(spsc.TryConsume());
  spsc.NotifyConsumed();
  EXPECT_TRUE(spsc.TryProduce());
  EXPECT_FALSE(spsc.TryProduce());
  spsc.NotifyProduced();
  EXPECT_TRUE(spsc.TryConsume());
  EXPECT_FALSE(spsc.TryConsume());
}

TEST(UtilTest, ClockResolution) {
  auto constexpr iterations = 1'000;
  auto min_resolution = Clock::duration::max();

  auto time_0 = Clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const time_1 = Clock::now();
    auto const diff = time_1 - time_0;
    if (Clock::duration::zero() < diff && diff < min_resolution) {
      min_resolution = diff;
    }
    time_0 = time_1;
  }
  EXPECT_LE(std::chrono::duration_cast<std::chrono::nanoseconds>(min_resolution).count(), 30);
}

TEST(UtilTest, Affinity) {
  auto constexpr thread_current = 0;
  auto cpu_set_orig = GetAffinity(thread_current);
  EXPECT_GE(cpu_set_orig.size(), 1);
  CpuSet cpu_set_temp{0u};
  SetAffinity(thread_current, cpu_set_temp);
  auto cpu_set_test = GetAffinity(thread_current);
  EXPECT_EQ(cpu_set_test, cpu_set_temp);
  SetAffinity(thread_current, cpu_set_orig);
  cpu_set_test = GetAffinity(thread_current);
  EXPECT_EQ(cpu_set_test, cpu_set_orig);
}