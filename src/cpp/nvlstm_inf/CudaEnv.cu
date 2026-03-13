/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "CudaEnv.cuh"
#include "ModelDef.h"

#include "ss_gemv_impl/kernel_params.h"

namespace nvidia::quant::lowlat_infer {

// Green Contexts
std::vector<CUdevResource> g_gctx_partitions;
CUdevResource g_gctx_remaining_partition;

GreenContextsVector g_contexts;

//static GreenContextsVector g_green_contexts;

// We manage the contexts outside of the benchmark class, 
// so there is more flexibility in when we delete the contexts.
// e.g. before calling DeviceReset but after the Benchmark class
// is out of scope
GreenContextsVector CreateGreenContexts(std::uint32_t num_units,
                         std::uint32_t num_instances, int sm_arch, int device_id,
                         CUcontext* primary_ctx_ptr, bool use_precompute_green_context, bool is_ping_pong_model) {

  GreenContextsVector green_contexts;

  CudaCheck(cudaGetDevice(&device_id));
  CUdevice current_device;
  CuCheck(cuDeviceGet(&current_device, device_id));

  cudaDeviceProp prop{};
  CudaCheck(cudaGetDeviceProperties(&prop, device_id));
  std::uint32_t sm_count = prop.multiProcessorCount;

  std::uint32_t min_sm_count_per_instance = (num_units <= LSTM_S_NUM_UNITS)? 8:16;
  std::uint32_t part_size = min_sm_count_per_instance;
  unsigned int use_flags = 0;

  if (!use_precompute_green_context) {
    part_size = min_sm_count_per_instance;
    while ((part_size + 8) * num_instances <= sm_count) {
      part_size += 8;
    };
    if (is_ping_pong_model) {
      use_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
    } else {
      use_flags = (num_units <= LSTM_S_NUM_UNITS)? 0: CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE;
      use_flags = (num_units <= LSTM_S_NUM_UNITS && num_instances > 8)? CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING : use_flags;
    }
  } else {
    if (num_units <= LSTM_S_NUM_UNITS || is_ping_pong_model) {
      part_size = 2;
      use_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
    } else {
      // add logic of choosing the cluster size
      part_size = 8; use_flags = 0;
      if (ssgemv::is_sm_heavy_arch(sm_arch)) {
          part_size = 16;
          use_flags = CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE;
      }
    }
  } 

  CUdevResource avail_resources;
  CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM;
  if (!primary_ctx_ptr) {
    CuCheck(cuDeviceGetDevResource(current_device, &avail_resources, default_resource_type));
  } else {
    CuCheck(cuCtxGetDevResource(*primary_ctx_ptr, &avail_resources, default_resource_type));
  }

  printf("Initial resources have %d SMs. Splitting into %d partitions of size %d\n", avail_resources.sm.smCount, num_instances, part_size);

  unsigned int min_SM_count = part_size;

  // only this is supported for now
  if (num_units <= LSTM_S_NUM_UNITS) StateCheck(num_instances <= 32);
  else {
    if (ssgemv::is_sm_heavy_arch(sm_arch)) {
      StateCheck(num_instances <= 8);
    }
    else {
      StateCheck(num_instances <= 16);
    }
  }

  g_gctx_partitions.resize(num_instances);
  unsigned int actual_split_groups = num_instances;

  CuCheck(cuDevSmResourceSplitByCount(&g_gctx_partitions[0],
                                      &actual_split_groups,
                                      &avail_resources,
                                      &g_gctx_remaining_partition,
                                      use_flags,
                                      min_SM_count));

  printf("Split %d SMs into %d groups with %d SMs each and a remaining group with %d SMs\n",
          avail_resources.sm.smCount, actual_split_groups, g_gctx_partitions[0].sm.smCount,
          g_gctx_remaining_partition.sm.smCount);

  StateCheck(num_instances == actual_split_groups);

  for (uint ii = 0; ii < actual_split_groups; ii++) {
    green_contexts.push_back(std::make_shared<GreenContext>(g_gctx_partitions[ii], device_id));
  }
  green_contexts.push_back(std::make_shared<GreenContext>(g_gctx_remaining_partition, device_id));
  StateCheck(green_contexts.size() >= 2);

  return green_contexts;
}

GreenContextsVector CreateGreenContexts(std::filesystem::path const &path_model,
                         std::uint32_t num_instances, int sm_arch, int device_id, bool use_precompute_green_context, bool is_ping_pong_model) {

  ModelDef model_def{path_model.c_str()};

  return CreateGreenContexts(model_def.NumUnits(), num_instances, sm_arch, device_id, nullptr, use_precompute_green_context, is_ping_pong_model);
}

int GetSmCountFromGreenStream(cudaStream_t stream) {
  CUgreenCtx hCtx;
  cuStreamGetGreenCtx ( stream, &hCtx );
  CUcontext context;
  cuCtxFromGreenCtx ( &context, hCtx );
  CUdevResource resource;
  cuCtxGetDevResource ( context, &resource,CU_DEV_RESOURCE_TYPE_SM );

  return resource.sm.smCount;
}

unsigned long long GetCtxIdFromGreenStream(cudaStream_t stream) {
  CUgreenCtx hCtx;
  cuStreamGetGreenCtx ( stream, &hCtx );
  CUcontext context;
  cuCtxFromGreenCtx ( &context, hCtx );

  unsigned long long ctxId;
  cuCtxGetId( context, &ctxId );

  return ctxId;
}

int CudaEnv::GetDevice() {
  int device;
  CudaCheck(cudaGetDevice(&device));
  return device;
}

void CudaEnv::CheckDevice() {
  auto const device = GetDevice();
  cudaDeviceProp props{};
  CudaCheck(cudaGetDeviceProperties(&props, device));
  StateCheck(props.canMapHostMemory);
  StateCheck(props.canUseHostPointerForRegisteredMem);
  StateCheck(props.concurrentManagedAccess);
  int value;
  CudaCheck(cudaDeviceGetAttribute(&value, cudaDevAttrHostRegisterSupported, device));
  StateCheck(value);
  CudaCheck(cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem, device));
  StateCheck(value);
  CudaCheck(cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, device));
  StateCheck(value);
  unsigned int flags;
  CudaCheck(cudaGetDeviceFlags(&flags));
  StateCheck(flags & cudaDeviceMapHost);
}

bool CudaEnv::NativeAtomicSupported() {
  auto const device = GetDevice();
  cudaDeviceProp props{};
  CudaCheck(cudaGetDeviceProperties(&props, device));
  return props.hostNativeAtomicSupported > 0;
}

std::size_t CudaEnv::MaxDynamicSharedMemory() {
  auto const device = GetDevice();
  cudaDeviceProp props{};
  CudaCheck(cudaGetDeviceProperties(&props, device));
  return props.sharedMemPerBlockOptin;
}


void CudaEnv::CreateStreams(unsigned int num_streams, int priority) {
  while (streams_.size() < num_streams) {
    streams_.push_back(cex::CreateStream(cudaStreamNonBlocking, priority));
    events_.push_back(cex::CreateEvent());
    forked_streams_.push_back(false);
  }
}

void CudaEnv::ForkStreams(cudaEvent_t after) {
  CudaCheck(cudaEventRecord(after, Stream()));
  for (unsigned int i = 0; i < streams_.size(); ++i) {
    StateCheck(!forked_streams_.at(i));
    CudaCheck(cudaStreamWaitEvent(streams_[i].get(), after));
    forked_streams_[i] = true;
  }
}

void CudaEnv::JoinStreams() {
  for (unsigned int i = 0; i < streams_.size(); ++i) {
    if (forked_streams_[i]) {
      JoinStream(i);
    }
  }
}

void CudaEnv::JoinStream(unsigned int idx) {
  StateCheck(forked_streams_.at(idx));
  auto event = events_[idx].get();
  CudaCheck(cudaEventRecord(event, streams_[idx].get()));
  CudaCheck(cudaStreamWaitEvent(Stream(), event));
  forked_streams_[idx] = false;
}

cudaStream_t CudaEnv::ForkHighPriority(cudaEvent_t after) {
  StateCheck(!stream_hp_forked_);
  stream_hp_forked_ = true;
#if HIGH_PRIORITY_FORK_DISABLE == 1
  return Stream();
#else
  CudaCheck(cudaEventRecord(after, Stream()));
  CudaCheck(cudaStreamWaitEvent(HighPriorityStream(), after));
  return HighPriorityStream();
#endif
}

void CudaEnv::JoinHighPriority(cudaEvent_t after) {
  StateCheck(stream_hp_forked_);
  stream_hp_forked_ = false;
#if HIGH_PRIORITY_FORK_DISABLE == 1
  return;
#else
  CudaCheck(cudaEventRecord(after, HighPriorityStream()));
  CudaCheck(cudaStreamWaitEvent(Stream(), after));
#endif
}

}  // namespace nvidia::quant::lowlat_infer