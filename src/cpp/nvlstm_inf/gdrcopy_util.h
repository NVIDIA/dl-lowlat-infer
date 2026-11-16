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

#pragma once

#include <cuda.h>
#include <gdrapi.h>
#include <gdrconfig.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string>

#include "cuda_error_util.h"
#include "util.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

// gdrcopy utility based on the gdrcopy example code
// https://github.com/NVIDIA/gdrcopy/blob/master/tests/common.hpp
namespace gdrcopy_utils {

#if defined(__aarch64__) || defined(_M_ARM64)
inline void storefence() { asm volatile("dmb st" ::: "memory"); }
#endif

#if defined(__x86_64__)
inline void storefence() { _mm_sfence(); }
#endif

typedef struct gpuMemHandle {
  CUdeviceptr ptr;  // aligned ptr if requested; otherwise, the same as unaligned_ptr.
  union {
    CUdeviceptr unaligned_ptr;  // for tracking original ptr; may be unaligned.
    // VMM with GDR support is available from CUDA 11.0
    CUmemGenericAllocationHandle handle;
  };
  size_t size;
  size_t allocated_size;
} gpu_mem_handle_t;

typedef struct {
  CUdeviceptr gpu_ptr;
  void* host_ptr;
  size_t size;

  // GDRCopy-related objects
  gpu_mem_handle_t gmhandle;
  gdr_t g;
  gdr_mh_t mh;
  void* map_ptr;
} gh_mem_handle_t;

bool check_gdr_support(unsigned int device_id);

CUresult gpu_mem_alloc(gpu_mem_handle_t* handle, const size_t size, bool aligned_mapping, bool set_sync_memops);
CUresult gpu_mem_free(gpu_mem_handle_t* handle);

// Whether GDRCopy works here (the current device): it opens GDRCopy (the gdrdrv module, or without it GDRCopy's
// DMA-BUF backend, which needs a driver supporting CUDA 13.3 or newer), then pins, maps, writes and reads back a device
// page. ok: description names the backend; otherwise it says why GDRCopy does not work.
struct GdrProbe {
  bool ok;
  std::string description;
};
GdrProbe probe_gdrcopy();

static inline gdr_t gdr_open_safe() {
  gdr_t g = gdr_open();
  if (!g)
    throw std::runtime_error("gdr_open failed: neither the gdrdrv module nor GDRCopy's DMA-BUF backend is usable");
  return g;
}

void gdr_buffer_map(gh_mem_handle_t* mhandle, size_t size, gdr_t g, CUdeviceptr existing_gpu_ptr = 0);
void gdr_buffer_unmap(gh_mem_handle_t* mhandle);

CUresult prepare_gdr_buffer(gdr_t g, gdr_mh_t& mh, gpu_mem_handle_t& mem_handle, void** buf_ptr);

// as per https://github.com/NVIDIA/gdrcopy/issues/152 comment
// we use a single gdr_t handle for all threads
// but then have one memory handle per thread
class GdrHandleManager {
 public:
  // Get the singleton instance (thread-safe via C++11 magic statics)
  static GdrHandleManager& GetInstance() {
    static GdrHandleManager instance;
    return instance;
  }

  // Get the gdr_t handle (thread-safe, lock-free read after initialization)
  gdr_t GetHandle() const { return gdr_handle_; }

  // Delete copy and move constructors/operators
  GdrHandleManager(const GdrHandleManager&) = delete;
  GdrHandleManager& operator=(const GdrHandleManager&) = delete;
  GdrHandleManager(GdrHandleManager&&) = delete;
  GdrHandleManager& operator=(GdrHandleManager&&) = delete;

  ~GdrHandleManager() {
    if (gdr_handle_) {
      gdr_close(gdr_handle_);
      gdr_handle_ = nullptr;
    }
  }

 private:
  GdrHandleManager() { gdr_handle_ = gdr_open_safe(); }

  gdr_t gdr_handle_;
};

template <typename T>
class GdrcopyBuffer {
 public:
  GdrcopyBuffer(size_t count) {
    g_ = GdrHandleManager::GetInstance().GetHandle();
    gdr_buffer_map(&mhandle_, count * sizeof(T), g_);
    CuCheck(cuMemsetD8_v2(mhandle_.gpu_ptr, 0, count * sizeof(T)));
  }

  GdrcopyBuffer(const GdrcopyBuffer&) = delete;
  GdrcopyBuffer& operator=(const GdrcopyBuffer&) = delete;
  GdrcopyBuffer(GdrcopyBuffer&&) = delete;
  GdrcopyBuffer& operator=(GdrcopyBuffer&&) = delete;

  T* GetHostPtr() { return reinterpret_cast<T*>(mhandle_.host_ptr); }

  T* GetDevicePtr() const { return reinterpret_cast<T*>(static_cast<uintptr_t>(mhandle_.gpu_ptr)); }

  CUdeviceptr GetDeviceCuPtr() const { return mhandle_.gpu_ptr; }

  gh_mem_handle_t* GetMemHandlePtr() { return &mhandle_; }

  ~GdrcopyBuffer() {
    if (mhandle_.host_ptr) {
      gdr_buffer_unmap(&mhandle_);
    }
  }

  gdrcopy_utils::gh_mem_handle_t mhandle_{0};
  gdr_t g_{0};
};

template <typename T>
using GdrcopyBufferPtr = std::unique_ptr<GdrcopyBuffer<T>>;

}  // namespace gdrcopy_utils

namespace nvidia::quant::lowlat_infer::ssgemv::cpu_pingpong_handles {

// The signal write through the GDRCopy mapping, which the device-side kernel spins on before it reads the step's input.
// On aarch64 two store barriers go around it, as around the resident precompute's GDRCopy writes:
// - dmb st before: the stores that the signal publishes (on aarch64 the step's input pointer, stored to pinned host
//   memory in InputImpl, fenced only with dmb ish by _mm_sfence) are ordinary cacheable stores, and the mapping is
//   write-combining or device memory, which dmb ish does not order them against. Without it the kernel can see the
//   signal first and run the step on the previous input.
// - dsb st after: the signal write completes before the host goes on, rather than waiting in the write buffer (without
//   it, P99 latency was about 20 us higher).
// gdr_copy_to_mapping fences only after a write-combining copy. On x86 the step's input itself goes through GDRCopy,
// and its sfence orders it before the signal.
template <typename datatype_t>
inline void signal_release(const gdrcopy_utils::GdrcopyBufferPtr<datatype_t>& x_gdr_buffer_ptr, datatype_t val) {
  datatype_t flag_value = val;
#if defined(__aarch64__) || defined(_M_ARM64)
  asm volatile("dmb st" ::: "memory");
#endif
#if 1
  int ret = gdr_copy_to_mapping(
      x_gdr_buffer_ptr->GetMemHandlePtr()->mh,
      (unsigned char*)x_gdr_buffer_ptr->GetHostPtr(),
      (unsigned char*)&flag_value,
      sizeof(datatype_t));
  StateCheck(ret == 0);
#else
  x_gdr_buffer_ptr->GetHostPtr()[0] = flag_value;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  asm volatile("dsb st" ::: "memory");
#endif
  // cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
}

template <typename datatype_t>
inline void signal_release(
    const gdrcopy_utils::GdrcopyBufferPtr<datatype_t>& x_gdr_buffer_ptr, datatype_t val, uint32_t* device_spin_switch) {
  signal_release(x_gdr_buffer_ptr, val * *device_spin_switch);

  *device_spin_switch = !(*device_spin_switch);
}
}  // namespace nvidia::quant::lowlat_infer::ssgemv::cpu_pingpong_handles
