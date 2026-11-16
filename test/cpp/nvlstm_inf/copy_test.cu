/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cstring>
#include <cutlass/tfloat32.h>
#include <gtest/gtest.h>
#include <nvlstm_inf/CudaEnv.cuh>
#include <nvlstm_inf/cuda_helpers.cuh>
#include <nvlstm_inf/util.h>
#include <random>

#include "nvlstm_inf/cuda_copy.cuh"

using namespace nvidia::quant::lowlat_infer;

namespace {
using PinnedVector = std::vector<float, cex::PinnedAllocator<float>>;

template <typename T>
void RandomVector(T &v) {
  std::mt19937 rnd(42);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> dist{-1, 1};
  for (auto &x : v) {
    x = static_cast<float>(dist(rnd));
  }
}
}  // namespace

class CopyTest : public ::testing::Test {
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

TEST_F(CopyTest, Copy) {
  if (device_count_ == 0) GTEST_SKIP();
  for (auto num_elements : {10ull, 1'000'000ull}) {
    PinnedVector src(num_elements, 0.f);
    std::vector<half, cex::PinnedAllocator<half>> tgt(num_elements, half{});
    RandomVector(src);
    auto stream = cuda_env_->Stream();
    copy::Convert(tgt.data(), src.data(), num_elements, stream);
    CudaCheck(cudaStreamSynchronize(stream));
    auto const expected = cex::FloatToHalf(src.data(), src.size());
    for (std::size_t i = 0; i < num_elements; ++i) {
      EXPECT_EQ(static_cast<float>(expected[i]), static_cast<float>(tgt[i]));
    }
  }
}

TEST_F(CopyTest, TF32) {
  if (device_count_ == 0) GTEST_SKIP();
  for (auto num_elements : {10ull, 1'000'000ull}) {
    PinnedVector src(num_elements, 0.f);
    std::vector<cex::TensorFloat32, cex::PinnedAllocator<cex::TensorFloat32>> tgt(num_elements);
    RandomVector(src);
    auto stream = cuda_env_->Stream();
    copy::Convert(tgt.data(), src.data(), num_elements, stream);
    CudaCheck(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < num_elements; ++i) {
      cutlass::tfloat32_t const expected{src[i]};
      EXPECT_FLOAT_EQ(expected, static_cast<float>(tgt[i]));
    }
  }
}

namespace {
void TestCopyBatch(CudaEnv &cuda_env, std::size_t batch_size, std::size_t num_per_batch) {
  auto const num_elements = batch_size * num_per_batch;
  PinnedVector src(num_elements, 0.f), tgt(num_elements, 0.f);
  RandomVector(src);
  auto batches = cex::MallocHost<float const *>(batch_size);
  for (std::size_t i = 0; i < batch_size; ++i) {
    batches[i] = src.data() + i * num_per_batch;
  }

  auto stream = cuda_env.Stream();
  copy::Convert(tgt.data(), batches.get(), batch_size, num_per_batch, stream);
  CudaCheck(cudaStreamSynchronize(stream));
  EXPECT_EQ(src, tgt);
}
}  // namespace

TEST_F(CopyTest, CopyBatch) {
  if (device_count_ == 0) GTEST_SKIP();
  TestCopyBatch(*cuda_env_, 1, 10);
  TestCopyBatch(*cuda_env_, 10, 1);
  TestCopyBatch(*cuda_env_, 1, 1000000);
  TestCopyBatch(*cuda_env_, 10, 100000);
  TestCopyBatch(*cuda_env_, 100, 10000);
  TestCopyBatch(*cuda_env_, 1000, 1000);
  TestCopyBatch(*cuda_env_, 10000, 100);
  TestCopyBatch(*cuda_env_, 100000, 10);
  TestCopyBatch(*cuda_env_, 1000000, 1);
}

enum class CopyType {
  Direct,
  Indirect,
  Cpu,
  CpuHalf,
};

namespace {
bool HalfEqual(half x, half y) { return static_cast<unsigned short>(x) == static_cast<unsigned short>(y); }
}  // namespace

namespace {
void TestCopyBatchStep(
    CudaEnv &cuda_env, std::size_t batch_size, std::size_t num_steps, std::size_t num_features, CopyType copy_type) {
  auto const num_steps_features = num_steps * num_features;
  auto const num_elements = batch_size * num_steps_features;
  auto const num_batch_features = batch_size * num_features;

  PinnedVector src(num_elements, 0.f), tgt(num_elements, 0.f);
  RandomVector(src);
  std::vector<half> tgt_half;

  auto stream = cuda_env.Stream();
  if (copy_type == CopyType::Indirect) {
    auto src_batches = cex::MallocHost<float const *>(batch_size);
    for (std::size_t idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
      src_batches[idx_batch] = src.data() + idx_batch * num_steps_features;
    }
    copy::Convert(tgt.data(), src_batches.get(), batch_size, num_steps, num_features, stream);
  } else if (copy_type == CopyType::Direct) {
    copy::Convert(tgt.data(), src.data(), batch_size, num_steps, num_features, stream);
  } else if (copy_type == CopyType::Cpu) {
    for (std::size_t idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
      auto const dst_batch = tgt.data() + idx_batch * num_features;
      auto const src_batch = src.data() + idx_batch * num_steps_features;
      util::CopyTranspose(dst_batch, src_batch, batch_size, num_steps, num_features);
    }
  } else if (copy_type == CopyType::CpuHalf) {
    tgt_half.resize(num_elements);
    for (std::size_t idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
      auto const dst_batch = tgt_half.data() + idx_batch * num_features;
      auto const src_batch = src.data() + idx_batch * num_steps_features;
      cex::CopyTransposeConvert(dst_batch, src_batch, batch_size, num_steps, num_features);
    }
  } else {
    FAIL();
  }

  auto const is_half = copy_type == CopyType::CpuHalf;

  CudaCheck(cudaStreamSynchronize(stream));
  for (std::size_t idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
    auto const idx_batch_0 = idx_batch * num_steps_features;
    auto const idx_batch_1 = idx_batch * num_features;
    for (std::size_t idx_step = 0; idx_step < num_steps; ++idx_step) {
      auto const idx_batch_step = idx_batch_0 + idx_step * num_features;
      auto const idx_step_batch = idx_step * num_batch_features + idx_batch_1;
      for (std::size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
        auto const idx_src = idx_batch_step + idx_feature;
        auto const idx_tgt = idx_step_batch + idx_feature;
        if ((!is_half && src[idx_src] != tgt[idx_tgt]) ||
            (is_half && !HalfEqual(__float2half_rn(src[idx_src]), tgt_half[idx_tgt]))) {
          std::cout << "idx_batch: " << idx_batch << " "
                    << "idx_step: " << idx_step << " "
                    << "idx_feature: " << idx_feature << " "
                    << "idx_src: " << idx_src << " "
                    << "idx_tgt: " << idx_tgt << std::endl;
        }
        if (is_half) {
          ASSERT_TRUE(HalfEqual(__float2half_rn(src[idx_src]), tgt_half[idx_tgt]));
        } else {
          ASSERT_EQ(src[idx_src], tgt[idx_tgt]);
        }
      }
    }
  }
}
}  // namespace

TEST_F(CopyTest, CopyBatchStep) {
  if (device_count_ == 0) GTEST_SKIP();
  for (auto copy_type : {CopyType::Direct, CopyType::Indirect, CopyType::Cpu, CopyType::CpuHalf}) {
    TestCopyBatchStep(*cuda_env_, 1000, 10, 100, copy_type);
    TestCopyBatchStep(*cuda_env_, 10, 1000, 100, copy_type);
    TestCopyBatchStep(*cuda_env_, 10, 10, 100000, copy_type);
    TestCopyBatchStep(*cuda_env_, 1, 1000, 1000, copy_type);
    TestCopyBatchStep(*cuda_env_, 1, 4, 4, copy_type);
  }
}

TEST_F(CopyTest, Memset) {
  if (device_count_ == 0) GTEST_SKIP();
  for (auto num_elements : {10ull, 1'000'000ull}) {
    PinnedVector tgt(num_elements, 0.f);
    auto stream = cuda_env_->Stream();
    auto constexpr value = 42.0f;
    copy::Memset(tgt.data(), value, num_elements, stream);
    CudaCheck(cudaStreamSynchronize(stream));
    for (auto x : tgt) {
      EXPECT_EQ(x, value);
    }
  }
}

TEST_F(CopyTest, Memset2D) {
  if (device_count_ == 0) GTEST_SKIP();
  for (auto num_elements : {100ull, 1'000'000ull}) {
    PinnedVector tgt(num_elements, 0.f);
    auto stream = cuda_env_->Stream();
    auto constexpr value = 42.0f;
    auto const width = 10ull;
    auto const pitch = 100ull;
    auto height = num_elements / pitch;
    copy::Memset2D(tgt.data(), pitch, value, width, height, stream);
    CudaCheck(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < num_elements; ++i) {
      auto const idx_col = i % pitch;
      auto const expected = idx_col < width ? value : 0.f;
      if (tgt[i] != expected) {
        std::cout << "i: " << i << " idx_col: " << idx_col << std::endl;
      }
      ASSERT_EQ(tgt[i], expected);
    }
  }
}

namespace {
void TestReplicate(CudaEnv &cuda_env, std::size_t num_elements, unsigned int times) {
  PinnedVector tgt(times * num_elements, 0.f), src(num_elements, 0.f);
  RandomVector(src);
  auto stream = cuda_env.Stream();
  copy::Replicate(tgt.data(), src.data(), num_elements, times, stream);
  CudaCheck(cudaStreamSynchronize(stream));
  for (auto t = 0u; t < times; ++t) {
    auto tgt_ptr = tgt.data() + t * num_elements;
    for (std::size_t i = 0; i < num_elements; ++i) {
      ASSERT_EQ(src[i], tgt_ptr[i]);
    }
  }
}
}  // namespace

TEST_F(CopyTest, Replicate) {
  if (device_count_ == 0) GTEST_SKIP();
  TestReplicate(*cuda_env_, 1, 10);
  TestReplicate(*cuda_env_, 10, 1);
  TestReplicate(*cuda_env_, 100, 1'000);
  TestReplicate(*cuda_env_, 1'000, 100);
  TestReplicate(*cuda_env_, 10'000, 10);
}

TEST_F(CopyTest, ScalarProduct) {
  if (device_count_ == 0) GTEST_SKIP();
  float const bias = 0.5;
  PinnedVector tgt(1, 0.f);
  for (auto num_elements : {100ull, 200ull, 1000ull}) {
    PinnedVector a(num_elements, 0.f), b(num_elements, 0.f);
    RandomVector(a);
    RandomVector(b);
    auto stream = cuda_env_->Stream();
    copy::ScalarProduct(tgt.data(), a.data(), b.data(), bias, num_elements, stream);
    CudaCheck(cudaStreamSynchronize(stream));
    auto expected = 0.f;
    for (std::size_t i = 0; i < num_elements; ++i) {
      expected += a[i] * b[i];
    }
    EXPECT_FLOAT_EQ(tgt[0], expected + bias);
  }
}
