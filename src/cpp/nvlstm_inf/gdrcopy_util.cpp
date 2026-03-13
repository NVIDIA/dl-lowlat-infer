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

#include <cstring>
#include "cuda_error_util.h"
#include "gdrcopy_util.h"

// gdrcopy utility based on the gdrcopy example code
// https://github.com/NVIDIA/gdrcopy/blob/master/tests/common.hpp

#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))
namespace gdrcopy_utils {

// Needs at least CUDA 11.3 for the ability to check GPUDirect RDMA support.
bool check_gdr_support(unsigned int device_id)
{
    CUdevice dev;
    CuCheck(cuDeviceGet(&dev, device_id));
    int drv_version;
    CuCheck(cuDriverGetVersion(&drv_version));

    // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA support.
    if (drv_version >= 11030) {
        int gdr_support = 0;
        CuCheck(cuDeviceGetAttribute(&gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));

        if (!gdr_support)
            fprintf(stderr, "This GPU does not support GPUDirect RDMA.\n");

        return !!gdr_support;
    }
    return false;
}

CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
{
    CUresult ret = CUDA_SUCCESS;
    CUdeviceptr ptr, out_ptr;
    size_t allocated_size;

    if (aligned_mapping)
        allocated_size = size + GPU_PAGE_SIZE - 1;
    else
        allocated_size = size;

    ret = cuMemAlloc(&ptr, allocated_size);
    if (ret != CUDA_SUCCESS)
        return ret;

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

CUresult gpu_mem_free(gpu_mem_handle_t *handle)
{
    CUresult ret = CUDA_SUCCESS;

    ret = cuMemFree(handle->unaligned_ptr);
    if (ret == CUDA_SUCCESS)
        memset(handle, 0, sizeof(gpu_mem_handle_t));

    return ret;
}

// Map a GPU buffer to CPU address space via GDRCopy
// If existing_gpu_ptr is 0, allocates new GPU memory; otherwise uses existing buffer
void gdr_buffer_map(gh_mem_handle_t *mhandle, size_t size, gdr_t g, CUdeviceptr existing_gpu_ptr){
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

    mhandle->gpu_ptr  = gpu_ptr;
    mhandle->host_ptr = host_ptr;
    mhandle->size     = size;
    mhandle->g        = g;
    mhandle->mh       = mh;
    mhandle->map_ptr  = map_ptr;
}

void gdr_buffer_unmap(gh_mem_handle_t *mhandle){

    StateCheck(gdr_unmap(mhandle->g, mhandle->mh, mhandle->map_ptr, mhandle->size) == 0);
    StateCheck(gdr_unpin_buffer(mhandle->g, mhandle->mh) == 0);
    // Only free GPU memory if we allocated it (gmhandle.ptr != 0)
    if (mhandle->gmhandle.ptr != 0) {
        CuCheck(gpu_mem_free(&mhandle->gmhandle));
    }

}

} // namespace gdrcopy_utils