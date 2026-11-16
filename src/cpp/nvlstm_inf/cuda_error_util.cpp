/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cuda_error_util.h"

#include <cstdio>
#include <stdexcept>

namespace nvidia::quant::lowlat_infer::cex {

void CuThrow(CUresult stat, const char *file, int line) {
  if (stat != CUDA_SUCCESS) {
    char buf[1024];
    const char *err_buf;
    cuGetErrorString(stat, &err_buf);
    std::snprintf(buf, std::size(buf), "CUDA Driver Error: %s %s %d", err_buf, file, line);
    throw std::runtime_error(buf);
  }
}

void CudaThrow(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    char buf[1024];
    std::snprintf(buf, std::size(buf), "CUDA Error: %s %s %d", cudaGetErrorString(stat), file, line);
    throw std::runtime_error(buf);
  }
}

namespace {
// cuBLAS API errors
const char *_cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
}  // namespace

void CublasThrow(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    char buf[1024];
    std::snprintf(buf, std::size(buf), "cuBLAS Error: %s %s %d", _cublasGetErrorString(stat), file, line);
    throw std::runtime_error(buf);
  }
}
}  // namespace nvidia::quant::lowlat_infer::cex
