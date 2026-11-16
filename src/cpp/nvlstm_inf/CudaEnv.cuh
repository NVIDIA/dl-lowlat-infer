/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_CUDAENV_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_CUDAENV_CUH_

#include <cassert>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>

#include "cuda_helpers.cuh"

namespace nvidia::quant::lowlat_infer {

#define CUDA_ENV_HIGH_PRIORITY_STREAM 1
#define HIGH_PRIORITY_FORK_DISABLE 0

class GreenContext;
using GreenContextsVector = std::vector<std::shared_ptr<GreenContext>>;

extern std::vector<CUdevResource> g_gctx_partitions;
extern CUdevResource g_gctx_remaining_partition;
extern GreenContextsVector g_contexts;

class GreenContext {
 public:
  GreenContext(CUdevResource resource, int current_device_id = 0) : resource_(resource) {
    CuCheck(cuDeviceGet(&current_device_, current_device_id));
    // step 3
    const int num_resources = 1;  // Has to be 1​
    CuCheck(cuDevResourceGenerateDesc(&resource_desc_, &resource_, num_resources));

    // Step (4) ​
    int flags = CU_GREEN_CTX_DEFAULT_STREAM;
    CuCheck(cuGreenCtxCreate(&green_ctx_, resource_desc_, current_device_, flags));

    // Step (5) ​
    CuCheck(cuCtxFromGreenCtx(&ctx_from_green_ctx_, green_ctx_));

    CuCheck(cuCtxGetId(ctx_from_green_ctx_, &ctx_id_));
  }

  ~GreenContext() { CuCheck(cuGreenCtxDestroy(green_ctx_)); }

  CUgreenCtx get() { return green_ctx_; }

  unsigned long long get_id() { return ctx_id_; }

  void SetCurrent() { CuCheck(cuCtxSetCurrent(ctx_from_green_ctx_)); }

 private:
  CUdevice current_device_;
  CUdevResource resource_;
  CUdevResourceDesc resource_desc_;
  CUgreenCtx green_ctx_;
  CUcontext ctx_from_green_ctx_;

  unsigned long long ctx_id_;
};

/// One green context per instance and one for the rest of the device (the precompute), for a model of num_units units.
/// fp16_reg_m: the instances run lstm_m with the FP16 register-resident kernel, whose partitions must hold its CTAs (as
/// one cluster with dsmem_exchange).
GreenContextsVector CreateGreenContexts(
    std::uint32_t num_units,
    std::uint32_t num_instances,
    int sm_arch,
    int device_id = 0,
    CUcontext *primary_ctx_ptr = nullptr,
    bool use_precompute_green_context = false,
    bool is_ping_pong_model = false,
    bool fp16_reg_m = false);

/// As above, for the model at path_model.
GreenContextsVector CreateGreenContexts(
    std::filesystem::path const &path_model,
    std::uint32_t num_instances,
    int sm_arch,
    int device_id = 0,
    bool use_precompute_green_context = false,
    bool is_ping_pong_model = false,
    bool fp16_reg_m = false);

/// SMs of the green context the stream belongs to; 0 for a stream outside green contexts.
int GetSmCountFromGreenStream(cudaStream_t stream);
unsigned long long GetCtxIdFromGreenStream(cudaStream_t stream);

class CudaEnv {
 public:
  explicit CudaEnv(
      bool lenient = false,
      bool create_stream = false,
      bool use_gdrcopy = false,
      std::shared_ptr<GreenContext> inference_context = nullptr,
      std::shared_ptr<GreenContext> precompute_context = nullptr,
      int cpu_core_id = -1)
      : device_count_{cex::DeviceCount()},
        use_gdrcopy_(use_gdrcopy),
        lenient_{lenient},
        create_stream_(create_stream),
        cpu_core_id_(cpu_core_id) {
#if !GDRCOPY_AVAILABLE
    if (use_gdrcopy_) {
      throw std::invalid_argument("--gdrcopy requires a build configured with -DUSE_GDRCOPY=ON");
    }
#endif
    if (device_count_ == 0 && !lenient) throw std::runtime_error("Missing cuda device.");

    if (inference_context) green_contexts_.push_back(inference_context);
    if (precompute_context) green_contexts_.push_back(precompute_context);

    StateCheck(green_contexts_.size() == 0 || green_contexts_.size() == 1 || green_contexts_.size() == 2);

    if (precompute_context) {
      precompute_context->SetCurrent();
    } else if (inference_context) {
      inference_context->SetCurrent();
    };

    Init(create_stream);
  }

  void AssignContexts(GreenContextsVector green_contexts, int device_id = 0) {
    StateCheck(green_contexts_.size() == 0);

    Release(true);

    green_contexts_ = green_contexts;

    CUdevice current_device;
    CuCheck(cuDeviceGet(&current_device, device_id));
    CuCheck(cuDevicePrimaryCtxRelease(current_device));

    // Make sure all the precompute handles are created in the precompute context
    green_contexts_.back()->SetCurrent();

    Init(create_stream_);
  }

  void ReleaseContexts(CUcontext &primary_ctx, int device_id = 0) {
    SetPrecomputeContext();

    CudaCheck(cudaDeviceSynchronize());
    Release(true);

    green_contexts_.resize(0);

    CUdevice current_device;
    CuCheck(cuDeviceGet(&current_device, device_id));
    CuCheck(cuDevicePrimaryCtxRetain(&primary_ctx, current_device));
  }

  ~CudaEnv() { Release(true); }

  // No copying
  CudaEnv(CudaEnv const &) = delete;
  CudaEnv &operator=(CudaEnv const &) = delete;

  // Moving
  CudaEnv(CudaEnv &&env) noexcept
      : device_count_{env.device_count_},
        use_gdrcopy_{env.use_gdrcopy_},
        lenient_{env.lenient_},
        create_stream_(env.create_stream_),
        cublas_handle_{env.cublas_handle_},
        cublaslt_handle_{env.cublaslt_handle_},
        cuda_stream_{std::move(env.cuda_stream_)},
        cuda_high_prio_stream_(std::move(env.cuda_high_prio_stream_)),
        green_contexts_(std::move(env.green_contexts_)),
        cpu_core_id_{env.cpu_core_id_} {
    env.cublas_handle_ = nullptr;
    env.cublaslt_handle_ = nullptr;
    env.cuda_stream_ = nullptr;
    env.cuda_high_prio_stream_ = nullptr;
  }

  CudaEnv &operator=(CudaEnv &&env) {  // NOLINT(performance-noexcept-move-constructor)
    Release();
    assert(device_count_ == env.device_count_);
    cublas_handle_ = env.cublas_handle_;
    cublaslt_handle_ = env.cublaslt_handle_;
    cuda_stream_ = std::move(env.cuda_stream_);
    cuda_high_prio_stream_ = std::move(env.cuda_high_prio_stream_);
    green_contexts_ = std::move(env.green_contexts_);
    env.cublas_handle_ = nullptr;
    env.cublaslt_handle_ = nullptr;
    env.cuda_stream_ = nullptr;
    env.cuda_high_prio_stream_ = nullptr;
    cpu_core_id_ = env.cpu_core_id_;
    use_gdrcopy_ = env.use_gdrcopy_;
    return *this;
  };

  [[nodiscard]] auto DeviceCount() const { return device_count_; };
  [[nodiscard]] auto Cublas() const { return cublas_handle_; };
  [[nodiscard]] auto CublasLt() const { return cublaslt_handle_; };
  [[nodiscard]] auto Stream() const { return cuda_stream_.get(); };
  [[nodiscard]] auto CustomStream() const { return cuda_custom_stream_.get(); };
  [[nodiscard]] auto HighPriorityStream() const { return cuda_high_prio_stream_.get(); };
  [[nodiscard]] bool UseGdrcopy() const {
#if GDRCOPY_AVAILABLE
    return use_gdrcopy_;
#else
    assert(!use_gdrcopy_);
    return false;
#endif
  }

  void CreateStreams(unsigned int num_streams, int priority = 0);

  void ForkStreams(cudaEvent_t after);

  void JoinStreams();

  void JoinStream(unsigned int idx);

  cudaStream_t ForkHighPriority(cudaEvent_t after);

  void JoinHighPriority(cudaEvent_t after);

  auto Streams(unsigned int idx) { return streams_.at(idx).get(); }

  static bool NativeAtomicSupported();

  static std::size_t MaxDynamicSharedMemory();

  static int GetDevice();

  int GetCpuCoreId() { return cpu_core_id_; };

  [[nodiscard]] int getPrioHigh() const { return prio_high_; }
  [[nodiscard]] int getPrioLow() const { return prio_low_; }

  void SetModelContext() const {
    if (green_contexts_.size()) green_contexts_[0]->SetCurrent();
  }
  void SetPrecomputeContext() const {
    if (green_contexts_.size()) green_contexts_.back()->SetCurrent();
  }
  /// A new non-blocking stream in the precompute's partition of the device (its green context, if there is one).
  [[nodiscard]] cex::StreamPtr CreatePrecomputeStream() const {
    if (green_contexts_.size()) {
      CUstream stream;
      CuCheck(cuGreenCtxStreamCreate(&stream, green_contexts_.back()->get(), CU_STREAM_NON_BLOCKING, 0));
      return cex::StreamPtr{stream};
    }
    return cex::CreateStream(cudaStreamNonBlocking);
  }

 private:
  void Init(bool create_stream = true) {
    if (!lenient_ || device_count_ > 0) {
      CheckDevice();
      cudaFree(0);
      CublasCheck(cublasCreate(&cublas_handle_));
      CublasCheck(cublasLtCreate(&cublaslt_handle_));
      if (create_stream) {
        if (green_contexts_.size() >= 1) {
          CUstream stream;
          CuCheck(cuGreenCtxStreamCreate(&stream, green_contexts_.back()->get(), CU_STREAM_NON_BLOCKING, 0));
          cuda_stream_ = cex::StreamPtr{stream};
          CUstream custom_stream;
          CuCheck(cuGreenCtxStreamCreate(&custom_stream, green_contexts_[0]->get(), CU_STREAM_NON_BLOCKING, 0));
          cuda_custom_stream_ = cex::StreamPtr{custom_stream};

        } else {
          cuda_stream_ = cex::CreateStream();
          cuda_custom_stream_ = cex::CreateStream();
        }
        CublasCheck(cublasSetStream(cublas_handle_, cuda_stream_.get()));
        CudaCheck(cudaDeviceGetStreamPriorityRange(&prio_low_, &prio_high_));
        StateCheck(prio_high_ < prio_low_);
        int prio_0, prio_1;
        CudaCheck(cudaStreamGetPriority(cuda_stream_.get(), &prio_0));
#if CUDA_ENV_HIGH_PRIORITY_STREAM == 1
        cuda_high_prio_stream_ = cex::CreateStream(cudaStreamNonBlocking, prio_high_);
        CudaCheck(cudaStreamGetPriority(cuda_high_prio_stream_.get(), &prio_1));
        StateCheck(prio_1 < prio_0);
#else
        cuda_high_prio_stream_ = cex::CreateStream();
        CudaCheck(cudaStreamGetPriority(cuda_high_prio_stream_.get(), &prio_1));
#endif
      }
    }
  }

  void Release(bool release_streams = false) {
    if (device_count_ > 0) cudaDeviceSynchronize();

    if (cublas_handle_) {
      CublasCheck(cublasDestroy(cublas_handle_));
      cublas_handle_ = nullptr;
    }
    if (cublaslt_handle_) {
      CublasCheck(cublasLtDestroy(cublaslt_handle_));
      cublaslt_handle_ = nullptr;
    }

    if (release_streams) {
      cuda_stream_.reset(nullptr);
      SetModelContext();
      cuda_custom_stream_.reset(nullptr);
      SetPrecomputeContext();
      cuda_high_prio_stream_.reset(nullptr);
    }
  }

  static void CheckDevice();

  int const device_count_;
  int const lenient_;
  int const create_stream_;
  cublasHandle_t cublas_handle_ = nullptr;
  cublasLtHandle_t cublaslt_handle_ = nullptr;
  cex::StreamPtr cuda_stream_{};
  cex::StreamPtr cuda_custom_stream_{};
  cex::StreamPtr cuda_high_prio_stream_{};
  std::vector<cex::StreamPtr> streams_{};
  std::vector<cex::EventPtr> events_{};
  std::vector<bool> forked_streams_{};
  bool stream_hp_forked_{false};
  int prio_high_, prio_low_;

  int cpu_core_id_{-1};
  bool use_gdrcopy_{false};

  GreenContextsVector green_contexts_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_CUDAENV_CUH_
