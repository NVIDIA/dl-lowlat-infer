/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include <nvlstm_inf/CudaEnv.cuh>
#include <nvlstm_inf/cuda_helpers.cuh>

using namespace nvidia::quant::lowlat_infer;

class CutlassTest : public ::testing::Test {
 protected:
  static auto constexpr kMaxCudaEnvs = 8;

  void SetUp() override {
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    for (std::size_t i = 0; i < kMaxCudaEnvs; ++i) {
      cuda_env_ = std::make_unique<CudaEnv>(lenient, create_stream);
    }
    device_count_ = cuda_env_->DeviceCount();
  }

  void TearDown() override { cuda_env_.reset(); }

  std::unique_ptr<CudaEnv> cuda_env_{};
  int device_count_{};
};

TEST_F(CutlassTest, Numeric) {
  if (device_count_ == 0) GTEST_SKIP();

  auto const x = 2.25_hf;
  EXPECT_EQ(static_cast<float>(x), 2.25f);
}

TEST_F(CutlassTest, Gemm) {
  if (device_count_ == 0) GTEST_SKIP();

  using data_type = float;
  using layout = cutlass::layout::ColumnMajor;

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<data_type,                   // ElementA
                                           layout,                      // LayoutA
                                           data_type,                   // ElementB
                                           layout,                      // LayoutB
                                           data_type,                   // ElementOutput
                                           layout,                      // LayoutOutput
                                           float,                       // ElementAccumulator
                                           cutlass::arch::OpClassSimt,  // tag indicating Tensor Cores
                                           cutlass::arch::Sm80  // tag indicating target GPU compute architecture
                                           >;

  Gemm gemm_op;

  //
  // Define the problem size
  //
  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //
  auto A = cex::Malloc2D<data_type>(M, K);
  auto B = cex::Malloc2D<data_type>(K, N);
  auto C = cex::Malloc2D<data_type>(M, N);

  //
  // Launch GEMM on the device
  //

  EXPECT_EQ(gemm_op.initialize({
                {M, N, K},
                {A.Get(), A.LeadingDim()},  // TensorRef to A device tensor
                {B.Get(), B.LeadingDim()},  // TensorRef to B device tensor
                {C.Get(), C.LeadingDim()},  // TensorRef to C device tensor
                {C.Get(), C.LeadingDim()},  // TensorRef to C device tensor
                {alpha, beta}               // epilogue operation arguments
            }),
            cutlass::Status::kSuccess);

  EXPECT_EQ(gemm_op(cuda_env_->Stream()), cutlass::Status::kSuccess);
}
