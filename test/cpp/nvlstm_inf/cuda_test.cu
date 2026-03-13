/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <cuda/std/chrono>
#include <cuda/std/semaphore>
#include <nvlstm_inf/CudaEnv.cuh>
#include <thread>

#include "nvlstm_inf/cuda_helpers.cuh"

using namespace cuda::std::chrono_literals;
using namespace nvidia::quant::lowlat_infer;

class CudaTest : public ::testing::Test {
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

TEST_F(CudaTest, SingleProducerSingleConsumer) {
  if (device_count_ == 0) GTEST_SKIP();

  auto spsc_ptr = cex::MakePinned<cex::SingleProducerSingleConsumer>(false, false);
  auto &spsc = *spsc_ptr;
  EXPECT_TRUE(spsc.IsLockFree());

  cuda::std::binary_semaphore semaphore{};

  for (std::size_t t = 0; t < 10'000; ++t) {
    // std=c++17
    // auto constexpr mem_acquire = cuda::std::memory_order::memory_order_acquire;
    // auto constexpr mem_release = cuda::std::memory_order::memory_order_release;
    // auto constexpr mem_acq_rel = cuda::std::memory_order::memory_order_acq_rel;
    // std=c++20
    auto constexpr mem_acquire = cuda::memory_order_acquire;
    auto constexpr mem_release = cuda::memory_order_release;
    auto constexpr mem_acq_rel = cuda::memory_order_acq_rel;
    EXPECT_FALSE(spsc.CanProduce(mem_acquire));
    EXPECT_FALSE(spsc.CanConsume(mem_acquire));
    EXPECT_FALSE(spsc.TryProduce(mem_acq_rel));
    EXPECT_FALSE(spsc.TryConsume(mem_acq_rel));
    spsc.NotifyConsumed(mem_release);
    EXPECT_TRUE(spsc.TryProduce(mem_acq_rel));
    EXPECT_FALSE(spsc.TryProduce(mem_acq_rel));
    spsc.NotifyProduced(mem_release);
    EXPECT_TRUE(spsc.TryConsume(mem_acq_rel));
    EXPECT_FALSE(spsc.TryConsume(mem_acq_rel));

    std::thread wait_for_notify{[&spsc, &semaphore](){
      spsc.WaitToProduce(mem_acq_rel);
      spsc.WaitToConsume(mem_acq_rel);
      semaphore.release();
    }};
    auto stream = cuda_env_->Stream();
    spsc.NotifyConsumed(stream, mem_release);
    spsc.NotifyProduced(stream, mem_release);
    ASSERT_TRUE(semaphore.try_acquire_for(1s));
    wait_for_notify.join();
    EXPECT_FALSE(spsc.TryProduce(mem_acq_rel));
    EXPECT_FALSE(spsc.TryConsume(mem_acq_rel));

    spsc.WaitToConsume(stream, mem_acq_rel);
    spsc.WaitToProduce(stream, mem_acq_rel);
    std::thread wait_for_spin{[&spsc, &semaphore](){
      spsc.NotifyConsumed(mem_release);
      spsc.NotifyProduced(mem_release);
      while (spsc.CanProduce(mem_acquire)) {}
      while (spsc.CanConsume(mem_acquire)) {}
      semaphore.release();
    }};
    ASSERT_TRUE(semaphore.try_acquire_for(1s));
    wait_for_spin.join();
    EXPECT_FALSE(spsc.CanProduce(mem_acquire));
    EXPECT_FALSE(spsc.CanConsume(mem_acquire));
  }
}

TEST_F(CudaTest, VolatileStoreSpinDevice) {
  if (device_count_ == 0) GTEST_SKIP();

  auto flag = cex::MakePinned<bool volatile>(false);
  EXPECT_FALSE(*flag);
  auto stream = cuda_env_->Stream();
  cex::spsc::WaitUntil(*flag, true, stream);
  *flag = true;
  CudaCheck(cudaStreamSynchronize(stream));
  EXPECT_TRUE(*flag);
  cex::spsc::Store(flag.get(), false, stream);
  CudaCheck(cudaStreamSynchronize(stream));
  EXPECT_FALSE(*flag);
}

TEST_F(CudaTest, VolatileStoreSpinHost) {
  if (device_count_ == 0) GTEST_SKIP();

  auto flag = cex::MakePinned<bool volatile>(false);
  EXPECT_FALSE(*flag);
  auto stream = cuda_env_->Stream();
  cex::spsc::Store(flag.get(), true, stream);
  while(!*flag) {
    std::this_thread::yield();
  }
}

TEST_F(CudaTest, Semaphore) {
  if (device_count_ == 0) GTEST_SKIP();

  auto sema = cex::MakeManaged<cuda::std::binary_semaphore>(0);
  auto stream = cuda_env_->Stream();
  auto constexpr num_repetitions = 1 << 8;
  for (std::size_t i = 0; i < num_repetitions; ++i) {
    EXPECT_FALSE(sema->try_acquire());
    cex::spsc::Acquire(*sema, stream);
    std::this_thread::sleep_for(std::chrono::microseconds{10});
    sema->release();
    CudaCheck(cudaStreamSynchronize(stream));
  }
  for (std::size_t i = 0; i < num_repetitions; ++i) {
    EXPECT_FALSE(sema->try_acquire());
    cex::spsc::Release(*sema, stream);
    EXPECT_TRUE(sema->try_acquire_for(cuda::std::chrono::seconds{1}));
    EXPECT_FALSE(sema->try_acquire());
  }
  sema->release();
  sema->release();
  EXPECT_TRUE(sema->try_acquire());
  EXPECT_FALSE(sema->try_acquire());
}

TEST_F(CudaTest, Fetch) {
  if (device_count_ == 0) GTEST_SKIP();
  auto constexpr num_elems = 100;
  auto constexpr value = 42.f;
  cex::PinnedVector32 src(num_elems, value);
  auto dev_ptr = cex::Malloc<float>(num_elems);
  auto stream = cuda_env_->Stream();
  cex::Copy(dev_ptr, src.data(), num_elems, stream);
  auto local = cex::Fetch(dev_ptr, num_elems, stream);
  EXPECT_EQ(local, src);
}

TEST_F(CudaTest, Array2D) {
  auto constexpr num_rows = 100;
  auto constexpr num_cols = 200;
  auto array2D = cex::Malloc2D<float>(num_rows, num_cols);
  EXPECT_TRUE(array2D);
  static_assert(decltype(array2D)::kValueSize == 4);
  static_assert(std::is_same_v<decltype(array2D)::ValueType, float>);
  EXPECT_GE(array2D.LeadingDim(), num_cols);
  EXPECT_EQ(array2D.LeadingDim() % 16, 0);
  EXPECT_EQ(array2D.Pitch(), array2D.LeadingDim() * sizeof(float));
  cex::PinnedVector32 src(num_rows * num_cols, 42.f);
  auto stream = cuda_env_->Stream();
  cex::Copy2D(array2D, src.data(), num_rows, num_cols, stream);
  cex::PinnedVector32 tgt(num_rows * num_cols);
  cex::Copy2D(tgt.data(), array2D, num_rows, num_cols, stream);
  CudaCheck(cudaStreamSynchronize(stream));
  EXPECT_EQ(src, tgt);
}