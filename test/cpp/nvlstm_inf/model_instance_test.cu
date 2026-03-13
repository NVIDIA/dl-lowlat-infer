/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <tuple>
#include <utility>

#include "nvlstm_inf/CudaEnv.cuh"
#include "nvlstm_inf/ModelInstance.cuh"
#include "nvlstm_inf/ss_gemv_impl/kernel_traits.h"

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer;

class ModelInstanceTest : public ::testing::Test {
 protected:
  static auto constexpr kMaxCudaEnvs = 8;

  void SetUp() override {
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    for (std::size_t i = 0; i< kMaxCudaEnvs; ++i) {
      cuda_envs_.push_back(std::make_unique<CudaEnv>(lenient, create_stream));
    }
    device_count_ = cuda_envs_[0]->DeviceCount();
    data_ = cex::PinnedTensor32{util::Tensor32::Load(std::filesystem::path(TEST_RESOURCES_DIR) / "lstm_s.npy")
                                    .Reallocate<cex::PinnedAllocator<float>>()};
  }

  void TearDown() override { cuda_envs_.clear(); }

  std::vector<std::unique_ptr<CudaEnv>> cuda_envs_{};
  cex::PinnedTensor32 data_{};
  int device_count_{};
};

class ModelInstanceParamTest
    : public ModelInstanceTest,
      public ::testing::WithParamInterface<std::tuple<std::uint32_t, bool, ssgemv::Data_kind, float>> {};

TEST_P(ModelInstanceParamTest, Loop) {
  if (device_count_ == 0) GTEST_SKIP();

  auto const batch_size = std::get<0>(GetParam());
  auto constexpr num_models = 1;  // Single-step mode only supports num_models = 1
  auto const stage = std::get<1>(GetParam());
  auto constexpr use_omp = false;

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  auto path = test_resources_path / "lstm_s";
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::SingleStep;
  config.data_type = std::get<2>(GetParam());
  ModelInstance::ModelVector models{};
  for (std::size_t i = 0; i < num_models; ++i) {
    models.push_back(ModelInstance::CreateModel(path, *cuda_envs_[i], batch_size, config));
  }
  ModelInstance::AnyDataIterator data{};
  if (config.Type() == ssgemv::DATA_TYPE_FP32) {
     data = std::make_unique<ModelInstance::PinnedDataIterator32>(data_, std::make_unique<util::RandomSampler>(41, 1, data_.Shape()[0]));
  } else {
    auto const data16 =
        data_.WithData(cex::FloatToHalf<cex::PinnedAllocator<half>>(data_.Data().data(), data_.Data().size()));
    data = std::make_unique<ModelInstance::PinnedDataIterator16>(data16, std::make_unique<util::RandomSampler>(41, 1, data16.Shape()[0]));
  }
  auto const ignore = models[0]->NumSteps() - 1;
  auto const iterations = batch_size << 8;
  auto constexpr reap = false;
  ModelInstance instance{std::move(models), std::move(data), iterations / 2, stage, use_omp, reap};
  EXPECT_FALSE(instance.Running());
  EXPECT_FALSE(instance.Stopping());
  EXPECT_EQ(instance.UseOmp(), use_omp);
  auto const& result = instance.Result();
  EXPECT_EQ(result.Size(), 0);

  std::array const expected{0.064819083, 0.073436707, 0.059797354, 0.050716668, 0.073893495, 0.068927355, 0.069986179, 0.083584458,
                            0.080267072, 0.071200296, 0.080009207, 0.087382995, 0.071729451, 0.074027367, 0.090647928, 0.090529755,
                            0.074804552, 0.050946970, 0.030959364, 0.021393511, 0.029250558, 0.037693381, 0.044212788, 0.049397185,
                            0.066885948, 0.061655439, 0.062567383, 0.053002290, 0.033487428, 0.019983493, 0.039805844, 0.063204959,
                            0.068325341, 0.064847589, 0.051150955, 0.065892503, 0.047354836, 0.055802122, 0.037880823, 0.031366859,
                            0.040071156, 0.043838549, 0.056253117, 0.061773267, 0.072327383, 0.072200947, 0.060081877, 0.051098354,
                            0.054549709, 0.044371627, 0.067443728, 0.065045938, 0.081806451, 0.075603664, 0.073458865, 0.064689815,
                            0.056921460, 0.046533268, 0.037616640, 0.059902404, 0.077613138, 0.073271416, 0.089865953, 0.061364599};

  double time_avg_expected = 24e-6;  // batch_size is always 1 in single-step mode

  for (std::size_t k = 0; k < 2; ++k) {
    instance.Start();
    EXPECT_TRUE(instance.Running());
    EXPECT_FALSE(instance.Stopping());

    std::size_t num_launched{0};

    while (instance.Running()) {
      if (instance.Loop()) {
        num_launched += 1;
      } else {
        instance.Reap();
      }
      if (result.Size() >= iterations - num_models * batch_size) {
        instance.Stop();
      }
    }

    EXPECT_FALSE(instance.Running());
    EXPECT_FALSE(instance.Stopping());
    EXPECT_EQ(result.Size(), iterations);
    EXPECT_EQ(num_launched, iterations / batch_size);

    if (num_models == 1) {
      double time_total{0};
      for (std::size_t i = 0; i < result.Size(); ++i) {
        time_total += InferenceResult::ToFracSeconds(result.TsStop()[i], result.TsStart()[i]);
      }

      auto const time_avg = time_total / static_cast<double>(result.Size());
      EXPECT_NEAR(time_avg, time_avg_expected, 10e-6);

    }

    ASSERT_GE(result.Output().size(), expected.size() + ignore);

    auto const precision = std::get<3>(GetParam());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result.Output()[i + ignore], expected[i], precision);
    }
    instance.Reset();
  }
}

INSTANTIATE_TEST_SUITE_P(BatchSizeTests,
                         ModelInstanceParamTest,
                         ::testing::Values(std::make_tuple(1u, false, ssgemv::DATA_TYPE_FP32, 5e-6),
                                           std::make_tuple(1u, true, ssgemv::DATA_TYPE_FP32, 5e-6),
                                           std::make_tuple(1u, false, ssgemv::DATA_TYPE_FP16, 2e-3),
                                           std::make_tuple(1u, true, ssgemv::DATA_TYPE_FP16, 2e-3)));

TEST_F(ModelInstanceTest, OpenMpTest) {
  auto constexpr batch_size = 1;  // single-step mode requires batch_size == 1
  auto constexpr stage = true;

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  auto path = test_resources_path / "lstm_s";
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::SingleStep;
  auto model = ModelInstance::CreateModel(path, *cuda_envs_[0], batch_size, config);
  auto data = std::make_unique<DataIterator<cex::PinnedVector32>>(data_, std::make_unique<util::RandomSampler>(41, 1, data_.Shape()[0]));
  ModelInstance instance{std::move(model), std::move(data), InferenceResult::kDefaultCapacity, stage};
  // OpenMP is not used with batch_size=1 in single-step mode
  EXPECT_FALSE(instance.UseOmp());
}