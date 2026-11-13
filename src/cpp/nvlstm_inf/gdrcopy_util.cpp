/* clang-format off */
/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * AND
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "gdrcopy_util.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "cuda_error_util.h"

// gdrcopy utility based on the gdrcopy example code
// https://github.com/NVIDIA/gdrcopy/blob/master/tests/common.hpp

#define PAGE_ROUND_UP(x, n) (((x) + ((n) - 1)) & ~((n) - 1))
namespace gdrcopy_utils {

// Needs at least CUDA 11.3 for the ability to check GPUDirect RDMA support.
bool check_gdr_support(unsigned int device_id) {
  CUdevice dev;
  CuCheck(cuDeviceGet(&dev, device_id));
  int drv_version;
  CuCheck(cuDriverGetVersion(&drv_version));

  // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA support.
  if (drv_version >= 11030) {
    int gdr_support = 0;
    CuCheck(cuDeviceGetAttribute(&gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));

    if (!gdr_support) fprintf(stderr, "This GPU does not support GPUDirect RDMA.\n");

    return !!gdr_support;
  }
  return false;
}

namespace {

// GDRCopy 2.6's DMA-BUF backend (no gdrdrv module) needs a driver supporting CUDA 13.3 or newer.
int constexpr kDmaBufMinDriverVersion = 13030;

std::string CudaVersionString(int version) {
  return std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
}

}  // namespace

GdrProbe probe_gdrcopy() {
  CudaCheck(cudaFree(nullptr));  // the primary context, current for the driver API calls below
  int driver_version = 0;
  CuCheck(cuDriverGetVersion(&driver_version));
  auto const driver = CudaVersionString(driver_version);

  gdr_t g = gdr_open();
  if (g == nullptr) {
    if (driver_version < kDmaBufMinDriverVersion) {
      return {
          false,
          "the gdrdrv module could not be opened, and the driver supports CUDA " + driver +
              "; GDRCopy without gdrdrv needs a driver supporting CUDA 13.3 or newer"};
    }
    return {
        false,
        "neither the gdrdrv module nor GDRCopy's DMA-BUF backend could be opened (the driver supports CUDA " + driver +
            ")"};
  }
  std::string backend = "gdrdrv";
#if GDR_API_VERSION >= ((2 << MAJOR_VERSION_SHIFT) | 6)
  int dma_buf = 0;
  if (gdr_get_attribute(g, GDR_ATTR_USING_DMA_BUF_MMAP, &dma_buf) == 0 && dma_buf != 0) {
    backend = "DMA-BUF, driver CUDA " + driver;
  }
#endif

  // A device page as the buffers are allocated (gdr_buffer_map), pinned, mapped, written and read back.
  std::string failure;
  gpu_mem_handle_t mem{};
  if (gpu_mem_alloc(&mem, GPU_PAGE_SIZE, true, true) != CUDA_SUCCESS) {
    failure = "allocating a device page failed";
  } else {
    gdr_mh_t mh{};
    if (gdr_pin_buffer(g, mem.ptr, GPU_PAGE_SIZE, 0, 0, &mh) != 0 || mh.h == 0) {
      failure = "pinning device memory failed";
    } else {
      void *map_ptr = nullptr;
      gdr_info_t info{};
      if (gdr_map(g, mh, &map_ptr, GPU_PAGE_SIZE) != 0) {
        failure = "mapping device memory failed";
      } else {
        if (gdr_get_info(g, mh, &info) != 0) {
          failure = "querying the mapping failed";
        } else {
          auto *host_ptr = static_cast<unsigned char *>(map_ptr) + (info.va - mem.ptr);
          std::uint32_t const written = 0x5a5a1234u;
          std::uint32_t read = 0;
          if (gdr_copy_to_mapping(mh, host_ptr, &written, sizeof(written)) != 0 ||
              gdr_copy_from_mapping(mh, &read, host_ptr, sizeof(read)) != 0 || read != written) {
            failure = "writing and reading back device memory through the mapping failed";
          }
        }
        gdr_unmap(g, mh, map_ptr, GPU_PAGE_SIZE);
      }
      gdr_unpin_buffer(g, mh);
    }
    gpu_mem_free(&mem);
  }
  gdr_close(g);
  if (!failure.empty()) return {false, "GDRCopy (" + backend + "): " + failure};
  return {true, backend};
}

CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops) {
  CUresult ret = CUDA_SUCCESS;
  CUdeviceptr ptr, out_ptr;
  size_t allocated_size;

  if (aligned_mapping)
    allocated_size = size + GPU_PAGE_SIZE - 1;
  else
    allocated_size = size;

  ret = cuMemAlloc(&ptr, allocated_size);
  if (ret != CUDA_SUCCESS) return ret;

  if (set_sync_memops) {
    unsigned int flag = 1;
    ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
    if (ret != CUDA_SUCCESS) {
      cuMemFree(ptr);
      return ret;
    }
  }

  if (aligned_mapping)
    out_ptr = PAGE_ROUND_UP(ptr, GPU_PAGE_SIZE);
  else
    out_ptr = ptr;

  handle->ptr = out_ptr;
  handle->unaligned_ptr = ptr;
  handle->size = size;
  handle->allocated_size = allocated_size;

  return CUDA_SUCCESS;
}

CUresult gpu_mem_free(gpu_mem_handle_t *handle) {
  CUresult ret = CUDA_SUCCESS;

  ret = cuMemFree(handle->unaligned_ptr);
  if (ret == CUDA_SUCCESS) memset(handle, 0, sizeof(gpu_mem_handle_t));

  return ret;
}

// Map a GPU buffer to CPU address space via GDRCopy
// If existing_gpu_ptr is 0, allocates new GPU memory; otherwise uses existing buffer
void gdr_buffer_map(gh_mem_handle_t *mhandle, size_t size, gdr_t g, CUdeviceptr existing_gpu_ptr) {
  CUdeviceptr gpu_ptr;

  if (existing_gpu_ptr == 0) {
    // Allocate new GPU memory
    gpu_mem_handle_t gmhandle;
    CuCheck(gpu_mem_alloc(&gmhandle, size, true, true));
    mhandle->gmhandle = gmhandle;
    gpu_ptr = gmhandle.ptr;
  } else {
    // Use existing GPU buffer (zero out gmhandle since we don't own this memory)
    memset(&mhandle->gmhandle, 0, sizeof(gpu_mem_handle_t));
    gpu_ptr = existing_gpu_ptr;
  }

  // Pin and map the buffer
  void *host_ptr = NULL;
  gdr_mh_t mh;
  void *map_ptr = NULL;
  gdr_info_t info;
  off_t off;

  StateCheck(gdr_pin_buffer(g, gpu_ptr, size, 0, 0, &mh) == 0);
  StateCheck(mh.h != 0);
  StateCheck(gdr_map(g, mh, &map_ptr, size) == 0);
  StateCheck(gdr_get_info(g, mh, &info) == 0);
  off = info.va - gpu_ptr;
  host_ptr = (void *)((uintptr_t)map_ptr + off);

  mhandle->gpu_ptr = gpu_ptr;
  mhandle->host_ptr = host_ptr;
  mhandle->size = size;
  mhandle->g = g;
  mhandle->mh = mh;
  mhandle->map_ptr = map_ptr;
}

void gdr_buffer_unmap(gh_mem_handle_t *mhandle) {
  StateCheck(gdr_unmap(mhandle->g, mhandle->mh, mhandle->map_ptr, mhandle->size) == 0);
  StateCheck(gdr_unpin_buffer(mhandle->g, mhandle->mh) == 0);
  // Only free GPU memory if we allocated it (gmhandle.ptr != 0)
  if (mhandle->gmhandle.ptr != 0) {
    CuCheck(gpu_mem_free(&mhandle->gmhandle));
  }
}

}  // namespace gdrcopy_utils
