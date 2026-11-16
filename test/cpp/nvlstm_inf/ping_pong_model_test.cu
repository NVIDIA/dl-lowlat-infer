/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <memory>
#include <nvlstm_inf/CudaEnv.cuh>
#include <nvlstm_inf/ModelConfig.cuh>
#include <nvlstm_inf/PingPongModelPersistent.cuh>
#include <nvlstm_inf/cuda_helpers.cuh>
#include <utility>

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer;

class PingPongModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    cuda_env_ = std::make_unique<CudaEnv>(lenient, create_stream);
    device_count_ = cuda_env_->DeviceCount();
  }

  void TearDown() override { cuda_env_.reset(); }

  std::unique_ptr<CudaEnv> cuda_env_{};
  int device_count_{};
};

class PingPongModelPersistentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    cuda_env_ = std::make_unique<CudaEnv>(lenient, create_stream);
    device_count_ = cuda_env_->DeviceCount();
  }

  void TearDown() override { cuda_env_.reset(); }

  std::unique_ptr<CudaEnv> cuda_env_{};
  int device_count_{};
};

class PingPongModelParamTest : public PingPongModelTest, public ::testing::WithParamInterface<int> {};

TEST_F(PingPongModelTest, Construct) {
  if (device_count_ == 0) GTEST_SKIP();

  auto path = std::filesystem::path(TEST_RESOURCES_DIR) / "lstm_s";
  ASSERT_TRUE(std::filesystem::exists(path));
  ModelDef model_def{path.c_str(), true};  // skip_weights=true for pingpong
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::PingPongPersistent;

  PingPongModelPersistent<float> model{*cuda_env_, model_def, 1, config};
  EXPECT_EQ(1, model.BatchSize());
  EXPECT_EQ(model_def.NumSteps(), model.NumSteps());
  EXPECT_EQ(model_def.NumFeatures(), model.NumFeatures());
}

namespace {

template <typename T>
void TestLstmS(CudaEnv &cuda_env, std::uint32_t batch_size, float abs_error) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / "lstm_s").c_str(), true};  // skip_weights=true for pingpong
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::PingPongPersistent;

  PingPongModelPersistent<T> model{cuda_env, model_def, batch_size, config};
  auto const data = util::Tensor32::Load(test_resources_path / "lstm_s.npy");

  auto const num_features = model_def.NumFeatures();
  ASSERT_EQ(data.Shape().at(1), num_features);

  // See test/python/onnx_inf_test.py
  std::array const expected{2.828571,  2.856524,  2.180112,  3.128277,  2.310311,  1.846714,  0.744907,  0.990764,
                            1.177333,  0.888849,  1.092423,  2.328708,  1.573656,  0.892719,  1.325031,  1.595074,
                            0.947756,  0.929175,  1.585958,  1.726998,  1.788090,  0.806547,  1.691239,  1.857200,
                            1.148445,  1.128221,  1.204678,  0.981003,  -0.262372, -0.885408, -0.504753, -0.950921,
                            0.255792,  -0.838356, 0.268282,  -1.044721, -2.070024, -1.453503, -0.529929, -0.755342,
                            -1.329478, -0.939819, -1.247294, -0.791838, -0.740532, -1.823964, -2.484209, -2.471167,
                            -2.145804, -2.386509, -2.330149, -2.474677, -2.446797, -2.177483, -1.057186, 0.459029,
                            0.635510,  1.350941,  1.026807,  1.519344,  2.259581,  3.067355,  3.907224,  2.460680};

  auto const num_steps = model_def.NumSteps();
  auto const init_size = (num_steps - 1) * num_features;

  auto const expected_size = expected.size();
  ASSERT_LE(expected_size, data.Shape().at(0) - num_steps + 1);
  ASSERT_EQ(expected_size % batch_size, 0);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model.InitX(initial);

  auto input = cex::MallocHost<float>(num_features * batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  auto const num_batches = expected_size / batch_size;
  for (std::size_t batch = 0; batch < num_batches; ++batch) {
    auto const batch_start = batch * batch_size;
    cex::Copy(input, data_start + batch_start * num_features, batch_size * num_features, cuda_env.Stream());
    model.Infer(input.get());
    model.Output(output.get());
    for (std::size_t offset = 0; offset < batch_size; ++offset) {
      EXPECT_NEAR(output[offset], expected[batch_start + offset], abs_error);
    }
  }
}

}  // namespace

TEST_P(PingPongModelParamTest, InferenceLstmSFloat) {
  // TODO: Skipped pending reference output generation and validation logic for ping pong model
  GTEST_SKIP() << "Ping pong model inference validation requires reference outputs";
  TestLstmS<float>(*cuda_env_, GetParam(), 5e-4);
}

TEST_P(PingPongModelParamTest, InferenceLstmSHalf) {
  // TODO: Skipped pending reference output generation and validation logic for ping pong model
  GTEST_SKIP() << "Ping pong model inference validation requires reference outputs";
  TestLstmS<half>(*cuda_env_, GetParam(), 4e-3);
}

INSTANTIATE_TEST_CASE_P(
    BatchSizeTests, PingPongModelParamTest, ::testing::Values(1));  // Single-step requires batch_size=1

TEST_F(PingPongModelPersistentTest, InferenceLstmS) {
  // TODO: Skipped pending reference output generation and validation logic for ping pong model
  GTEST_SKIP() << "Ping pong model inference validation requires reference outputs";
  if (device_count_ == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / "lstm_s").c_str(), true};  // skip_weights=true for pingpong

  auto constexpr batch_size = 1;
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::PingPongPersistent;
  PingPongModelPersistent<float> model{*cuda_env_, model_def, batch_size, config};
  auto const data = util::Tensor32::Load(test_resources_path / "lstm_s.npy");

  auto const num_features = model_def.NumFeatures();

  ASSERT_EQ(data.Shape().at(1), num_features);

  // See test/python/onnx_inf_test.py
  std::array const expected{2.828571,  2.856524,  2.180112,  3.128277,  2.310311,  1.846714,  0.744907,  0.990764,
                            1.177333,  0.888849,  1.092423,  2.328708,  1.573656,  0.892719,  1.325031,  1.595074,
                            0.947756,  0.929175,  1.585958,  1.726998,  1.788090,  0.806547,  1.691239,  1.857200,
                            1.148445,  1.128221,  1.204678,  0.981003,  -0.262372, -0.885408, -0.504753, -0.950921,
                            0.255792,  -0.838356, 0.268282,  -1.044721, -2.070024, -1.453503, -0.529929, -0.755342,
                            -1.329478, -0.939819, -1.247294, -0.791838, -0.740532, -1.823964, -2.484209, -2.471167,
                            -2.145804, -2.386509, -2.330149, -2.474677, -2.446797, -2.177483, -1.057186, 0.459029,
                            0.635510,  1.350941,  1.026807,  1.519344,  2.259581,  3.067355,  3.907224,  2.460680};

  auto const num_steps = model_def.NumSteps();
  auto const init_size = (num_steps - 1) * batch_size * num_features;

  ASSERT_TRUE(expected.size() <= data.Shape().at(0) - num_steps + 1);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model.InitX(initial);

  auto input = cex::MallocHost<float>(num_features * batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  for (std::size_t offset = 0; offset < expected.size(); ++offset) {
    cex::Copy(input, data_start + offset * num_features, num_features, cuda_env_->Stream());
    model.PreProcess();
    model.Infer(input.get());
    model.Output(output.get());
    model.OutputWait();
    model.PostProcess();
    EXPECT_NEAR(output[0], expected[offset], 6e-4);
  }
}
