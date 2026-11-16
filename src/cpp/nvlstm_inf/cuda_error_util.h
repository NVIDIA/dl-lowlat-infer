/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// Define some error checking macros.
#define CuCheck(stat)                                                       \
  do {                                                                      \
    nvidia::quant::lowlat_infer::cex::CuCheck_((stat), __FILE__, __LINE__); \
  } while (0)
#define CudaCheck(stat)                                                       \
  do {                                                                        \
    nvidia::quant::lowlat_infer::cex::CudaCheck_((stat), __FILE__, __LINE__); \
  } while (0)
#define CublasCheck(stat)                                                       \
  do {                                                                          \
    nvidia::quant::lowlat_infer::cex::CublasCheck_((stat), __FILE__, __LINE__); \
  } while (0)

namespace nvidia::quant::lowlat_infer::cex {

void CuThrow(CUresult stat, const char *file, int line);
void CudaThrow(cudaError_t stat, const char *file, int line);
void CublasThrow(cublasStatus_t stat, const char *file, int line);

void inline CuCheck_(CUresult stat, const char *file, int line) {
  if (stat != CUDA_SUCCESS) CuThrow(stat, file, line);
}

void inline CudaCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) CudaThrow(stat, file, line);
}

void inline CublasCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) CublasThrow(stat, file, line);
}

}  // namespace nvidia::quant::lowlat_infer::cex
