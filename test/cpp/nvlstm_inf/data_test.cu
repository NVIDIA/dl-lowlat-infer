/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <algorithm>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <numeric>
#include <nvlstm_inf/DataIterator.h>
#include <nvlstm_inf/cuda_helpers.cuh>
#include <nvlstm_inf/util.h>
#include <vector>

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer;

class DataTest : public ::testing::Test {
 public:
  static auto constexpr kNumFeatures = 128;
  static auto constexpr kNumInstances = 16;

 protected:
  void SetUp() override {
    device_count_ = cex::DeviceCount();
    data_ = util::Tensor32::Load(std::filesystem::path(TEST_RESOURCES_DIR) / "lstm_s.npy");
  }

  void TearDown() override {}

  int device_count_{};
  util::Tensor32 data_;
  // clang-format off
  std::array<float, 8> const expected_means_{
    -0.0315759,
    0.0100055,
    0.039832,
    0.00251134,
    -0.000206796,
    0.0241455,
    0.00356965,
    -0.00447728,
  };
  // clang-format on
};

namespace {

template <typename InputIt>
float Mean(InputIt first, InputIt last) {
  return std::accumulate(first, last, 0.0f) / (last - first);
}

}  // namespace

static constexpr size_t N_SAMPLES_USED = 65536;

TEST_F(DataTest, Iterate) {
  DataIterator data_iterator{data_, std::make_unique<util::RandomSampler>(41, 1, N_SAMPLES_USED)};
  EXPECT_EQ(data_iterator.NumSamples(), 65536);
  EXPECT_EQ(data_iterator.NumExcessSamples(), 128);
  EXPECT_EQ(data_iterator.NumFeatures(), 128);

  std::vector<float> output(kNumFeatures * kNumInstances, 0.0f);

  for (std::size_t k = 0; k < 2; ++k) {
    for (float expected_mean : expected_means_) {
      for (std::uint32_t i = 0; i < kNumInstances; ++i) {
        data_iterator.MemcpyNext(output.data() + i * kNumFeatures);
      }
      EXPECT_TRUE(std::all_of(output.cbegin(), output.cend(), [](auto x) { return x != 0.0f; }));
      auto const mean = Mean(output.cbegin(), output.cend());
      EXPECT_NEAR(mean, expected_mean, 1e-6);
    }
    data_iterator.Reset();
  }
}

TEST_F(DataTest, IterateCuda) {
  if (device_count_ == 0) GTEST_SKIP();

  DataIterator data_iterator{data_, std::make_unique<util::RandomSampler>(41, 1, N_SAMPLES_USED)};

  auto constexpr num_elems = kNumFeatures * kNumInstances;
  auto pinned = cex::MallocHost<float>(num_elems);
  auto output = pinned.get();

  for (float expected_mean : expected_means_) {
    for (std::uint32_t i = 0; i < kNumInstances; ++i) {
      data_iterator.MemcpyNext(output + i * kNumFeatures);
    }
    EXPECT_TRUE(std::all_of(output, output + num_elems, [](auto x) { return x != 0.0f; }));
    auto const mean = Mean(output, output + num_elems);
    EXPECT_NEAR(mean, expected_mean, 1e-6);
  }
}

TEST_F(DataTest, IteratePinned) {
  if (device_count_ == 0) GTEST_SKIP();

  DataIterator data_iterator{
      data_.Reallocate<cex::PinnedAllocator<float>>(), std::make_unique<util::RandomSampler>(41, 1, N_SAMPLES_USED)};

  auto constexpr num_elems = kNumFeatures * kNumInstances;
  auto pinned = cex::MallocHost<float>(num_elems);
  auto output = pinned.get();

  for (float expected_mean : expected_means_) {
    for (std::uint32_t i = 0; i < kNumInstances; ++i) {
      data_iterator.MemcpyNext(output + i * kNumFeatures);
    }
    EXPECT_TRUE(std::all_of(output, output + num_elems, [](auto x) { return x != 0.0f; }));
    auto const mean = Mean(output, output + num_elems);
    EXPECT_NEAR(mean, expected_mean, 1e-6);
  }
}
