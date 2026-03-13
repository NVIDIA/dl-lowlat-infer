/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <filesystem>
#include <tuple>
#include <utility>

#include "nvlstm_inf/Benchmark.cuh"
#include "nvlstm_inf/cuda_helpers.cuh"
#include "nvlstm_inf/ss_gemv_impl/kernel_traits.h"

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer;

class BenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_count_ = cex::DeviceCount();
    auto dir_name = (std::filesystem::temp_directory_path() / "inference_test_XXXXXX").string();
    auto dir_name_c = dir_name.data();
    dir_name_c = mkdtemp(dir_name_c);
    ASSERT_TRUE(dir_name_c);
    tmpdir = std::filesystem::path{dir_name_c};
  }

  void TearDown() override { std::filesystem::remove_all(tmpdir); }

  int device_count_{};
  std::filesystem::path tmpdir;
};

using Param = std::tuple<std::uint32_t, std::uint32_t, bool, bool>;

class BenchmarkParamTest : public BenchmarkTest, public ::testing::WithParamInterface<Param> {};

TEST_P(BenchmarkParamTest, Run) {
  if (device_count_ == 0) GTEST_SKIP();

  auto const num_instances = std::get<0>(GetParam());
  auto const batch_size = std::get<1>(GetParam());

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  auto constexpr model_name = "lstm_s";
  auto const path_model = test_resources_path / model_name;
  auto const path_data = test_resources_path / (std::string(model_name) + ".npy");
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::SingleStep;
  config.data_type = ssgemv::DATA_TYPE_FP32;
  auto const stage = std::get<2>(GetParam());
  auto const reap = std::get<3>(GetParam());
  std::chrono::nanoseconds grace_period{1'000};
  Benchmark benchmark{path_data, path_model, {42}, num_instances, batch_size, config, stage, reap, {}, grace_period};
  EXPECT_EQ(benchmark.NumInstances(), num_instances);
  benchmark.Run(1);

  auto const size_zero = static_cast<std::int32_t>(benchmark.Result(0).Size());
  for (std::size_t i = 0; i < num_instances; ++i) {
    auto const size_i = static_cast<std::int32_t>(benchmark.Result(i).Size());
    EXPECT_GE(size_i, 1000);
    EXPECT_LE(std::abs(size_zero - size_i), batch_size);
  }

  auto const prefix = std::string("output.") + std::to_string(num_instances) + "." + std::to_string(batch_size) + ".";
  auto const base_path = tmpdir / prefix;
  benchmark.Save(base_path);
  int count{0};
  for (auto const& entry : std::filesystem::directory_iterator{tmpdir}) {
    if (entry.is_regular_file()) {
      auto const filename = entry.path().filename().string();
      if (filename.rfind(prefix, 0) == 0 && filename.find(".npy", filename.size() - 4) != std::string::npos) ++count;
    }
  }
  EXPECT_EQ(count, num_instances);
}

INSTANTIATE_TEST_SUITE_P(BatchSizeTests,
                         BenchmarkParamTest,
                         ::testing::Values(Param{1, 1, false, false},
                                           Param{8, 1, true, true},
                                           Param{1, 1, false, true},
                                           Param{8, 1, false, false}));