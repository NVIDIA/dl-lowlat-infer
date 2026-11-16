/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_MODELPERSISTENTBASE_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_MODELPERSISTENTBASE_CUH_

#include <cstdint>

#include "CudaEnv.cuh"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "ModelInterface.h"
#include "ss_gemv_impl/gemv_lstm.h"
#include "ss_gemv_impl/kernel_params.h"
#include "util.h"

#if AARCH64_GH == 0
#include <emmintrin.h>
#include <immintrin.h>
#endif

#if GDRCOPY_AVAILABLE
#include "gdrcopy_util.h"
#else
#include "gdrcopy_stub.h"
#endif

// 1: with --gdrcopy, the models that call UseDeviceInput() (the SingleStepGemv models, ping-pong) get each step's
// input in a device buffer written through GDRCopy. Default on PCIe builds; off with AARCH64_GH, where the kernel reads
// the pinned input over the coherent C2C link and the buffer cost latency (GH200: lstm_s / lstm_m mean +0.32 / +0.48
// us). Set it with -DGDRCOPY_DEVICE_INPUT=0|1 to try it on other systems.
#ifndef GDRCOPY_DEVICE_INPUT
#define GDRCOPY_DEVICE_INPUT (AARCH64_GH == 0)
#endif

/*
 We use sliding window last time step implementation for the ping pong model
*/

namespace nvidia::quant::lowlat_infer {

#define GRAPH_CAPTURE_DISABLE 0

// Supported data types are: float, half
template <typename TData = float>
class ModelPersistentBase : public ModelInterface {
 public:
  using data_type = TData;
  using Base = ModelInterface;

  using Base::NumUnits;

  ModelPersistentBase(
      CudaEnv &cuda_env, ModelDef const &model_def, std::uint32_t batch_size, bool full_windows, bool single_step)
      : ModelInterface{batch_size, model_def.NumSteps(), model_def.NumFeatures(), model_def.NumUnits()},
        cuda_env_{cuda_env},
        num_features_(model_def.NumFeatures()),
        full_windows_(full_windows),
        single_step_(single_step),
        is_ping_pong_(model_def.IsPingPong()) {
    StateCheck(batch_size == 1);

    StateCheck(!full_windows_);

    if (Env().UseGdrcopy()) {
      sig_device_gdr_buffer_ptr_ = std::make_unique<gdrcopy_utils::GdrcopyBuffer<uint>>(1);
    }

#if AARCH64_GH == 1 && GH_PAGED_INPUT == 1
    pinned_x_ = cex::MallocPaged<TData>(num_features_);
#else
    pinned_x_ = cex::MallocHost<TData>(num_features_);
#endif
#if AARCH64_GH == 1
    pinned_out_ = cex::MallocPaged<float>(1);
    sig_device_ = cex::MallocPaged<int>(1);
#else
    pinned_out_ = cex::MallocHost<float>(1);
    sig_device_ = cex::MallocHost<int>(1);
#endif
    pinned_input_addr_ = cex::MallocHost<void *>(1);

    host_input_ptr_ = reinterpret_cast<float *>(pinned_x_.get());
  }
  ~ModelPersistentBase() override = default;

  inline int *DeviceSignal() {
    return Env().UseGdrcopy() ? reinterpret_cast<int *>(DeviceSignalGdrBufferPtr()->GetDevicePtr()) : sig_device_.get();
  };

  inline int *HostSignal() { return reinterpret_cast<int *>(pinned_out_.get()); };

  inline gdrcopy_utils::GdrcopyBufferPtr<uint> &DeviceSignalGdrBufferPtr() { return sig_device_gdr_buffer_ptr_; };

  template <bool Switch = false>
  inline void SignalReleaseDevice(uint value) {
    if (Switch) {
#if DISABLE_ATOMIC_EXCHANGE == 1
      if (Env().UseGdrcopy()) {
        ssgemv::cpu_pingpong_handles::signal_release<uint>(DeviceSignalGdrBufferPtr(), value, &DeviceSignalSwitch());
      } else {
        ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)DeviceSignal(), value, &DeviceSignalSwitch());
      }
#endif
    } else {
      if (Env().UseGdrcopy()) {
        ssgemv::cpu_pingpong_handles::signal_release<uint>(DeviceSignalGdrBufferPtr(), value);
      } else {
        ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)DeviceSignal(), value);
      }
    }
  }

  bool IsCapturing() const { return is_capturing_; };

  bool IsStaging() const { return stage_; };

  // Where a staged kernel reads the step's input (params.gmem_input): the device buffer of UseDeviceInput() with
  // --gdrcopy, else the pinned staging buffer. Kernels read it with __ldcv (it changes every step).
  void *KernelInputPtr() const {
#if GDRCOPY_AVAILABLE
    if (KernelInputOnDevice()) return input_gdr_buffer_ptr_->GetDevicePtr();
#endif
    return host_input_ptr_;
  }
  /// Whether KernelInputPtr() is the device buffer (params.input_on_device).
  bool KernelInputOnDevice() const {
#if GDRCOPY_AVAILABLE
    return stage_ && input_gdr_buffer_ptr_ != nullptr;
#else
    return false;
#endif
  }
  /// Both into a kernel's params (gmem_input, input_on_device).
  template <typename Params>
  void SetKernelInput(Params &params) const {
    params.gmem_input = KernelInputPtr();
    params.input_on_device = KernelInputOnDevice() ? 1 : 0;
  }

  bool IsFp32Input() const { return is_fp32_input_; };

  // Pointers
  float *HostInputPtr() { return host_input_ptr_; };
  float const **HostInputAddr() { return host_input_addr_; };

  void **PinedInputAddr() { return pinned_input_addr_.get(); };

  ModelInterface::Graph *GraphPtr() { return graph_; }

  // Should be called from the child classes as well
  virtual void InitParams() {
    is_output_ready_ = false;

    if (stage_ && host_input_ptr_ == nullptr) {
      throw std::runtime_error("Using input before setting it.");
    }

    // Note: we convert fp32 input to fp16 at earlier stage
    StateCheck(!(stage_ && is_fp32_input_ && data_kind_ != ssgemv::DATA_TYPE_FP32));
  }

  // Indicates whether the model ingests sliding windows or full windows.
  [[nodiscard]] bool Sliding() const override { return !full_windows_; };

  // 16 or 32
  [[nodiscard]] std::uint8_t Precision() const override { return static_cast<std::uint8_t>(sizeof(data_type) * 8); }

  // Copy the initial steps - 1 onto the device.
  void InitX(util::Tensor32 const &t) override {
    // See abstractModel<TData>::InitX logic

#if DISABLE_ATOMIC_EXCHANGE == 1
    DeviceSignalSwitchReset();
#endif

    // Might need to revise for ping pong model unit testing
  };

  void Input(float const *host_in, bool sequence_major) override {
    if (!stage_) {
      throw std::runtime_error("Only supported with staging");
    }

    host_input_ptr_ = (float *)host_in;
    InputImpl((float const **)&host_input_ptr_);
  };

  void Input(half const *host_in, bool sequence_major) override {
    if (!stage_) {
      throw std::runtime_error("Only supported with staging");
    }

    host_input_ptr_ = (float *)host_in;
    InputImpl((half const **)&host_input_ptr_);
  };

  void Input(float const **host_addr) override { InputImpl(host_addr); };

  void Input(half const **host_addr) override { InputImpl(host_addr); }

  // Check if the last async copy operation of the input has finished.
  bool InputReady() override { return true; };

  // Wait until the last async copy operation of the input has finished.
  void InputWait() override {}

  void PreProcess() override {
    InitParams();

    KernelLaunch();

    // Wait till data is loaded by the kernel
    // ssgemv::cpu_pingpong_handles::poll_load<uint>((uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
    // can just use poll_load since HOST_SPIN_0_UNTIL_SIGNAL_VALUE == HOST_SPIN_1_WHILE_SIGNAL_VALUE
    ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
        (uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
  }

  bool PreProcessReady() override { return true; }

  void LstmForward() override {
#if GDRCOPY_AVAILABLE
    if (stage_ && input_gdr_buffer_ptr_) {
      // the input to its device buffer first: the GDRCopy writes of one thread reach the device in order, so the kernel
      // sees the input once it sees the signal
      auto &buf = input_gdr_buffer_ptr_;
      int const ret = gdr_copy_to_mapping(
          buf->GetMemHandlePtr()->mh, buf->GetHostPtr(), host_input_ptr_, num_features_ * sizeof(TData));
      StateCheck(ret == 0);
    }
#endif
    // Signal the device that the computation can be started
#if DISABLE_ATOMIC_EXCHANGE == 1
    SignalReleaseDevice<true>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#else
    SignalReleaseDevice<false>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#endif
  }

  // Run the final aggregation.
  void Aggregate() override {}

  // Copy the inference results to the host. The `wait` arguments determines whether this method blocks until the
  // async copy operation has finished.
  void Output(float *host_out, bool wait, bool shift) override {
    // wait for GPU to signal
    ssgemv::cpu_pingpong_handles::poll_load<uint, true>((uint *)HostSignal(), HOST_SPIN_1_WHILE_SIGNAL_VALUE);

    *host_out = *pinned_out_.get();

    is_output_ready_ = true;
  };

  virtual void KernelLaunch() {
    // dummy launch
    ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
  }

  // Launch the full inference CUDA graph.
  void Launch(Graph &graph, cudaHostFn_t call_back, void *user_data) override {
#if GRAPH_CAPTURE_DISABLE == 1
    graph_ = nullptr;
#else
    graph_ = &graph;
    if (!is_capturing_) throw std::runtime_error("Launch not captured");
#endif
    if (stage_) {
      if (is_fp32_input_) {
        Infer(host_input_ptr_);
      } else {
        Infer(reinterpret_cast<half const *>(host_input_ptr_));
      }
    } else {
      if (is_fp32_input_) {
        Infer(host_input_addr_);
      } else {
        Infer(reinterpret_cast<half const **>(host_input_addr_));
      }
    }
    Output(host_output_ptr_, false, true);
  }

  // Check if the last async copy operation of the output has finished.
  bool OutputReady() override { return is_output_ready_; }

  // Wait until the last async copy operation of the output has finished.
  void OutputWait() override {}

  // Capture the full inference CUDA graph.
  Graph Capture(float const *input, float *host_out, bool sequence_major) override {
    SetCaptureStates(&input, host_out, true);
    return Graph();
  }

  // Capture the full inference CUDA graph with FP16 input.
  Graph Capture(half const *input, float *host_out, bool sequence_major) override {
    SetCaptureStates(&input, host_out, true);
    return Graph();
  }

  // Capture the full inference CUDA graph with indirect copy.
  Graph Capture(float const **input, float *host_out) override {
    SetCaptureStates(input, host_out, false);
    return Graph();
  }

  // Capture the full inference CUDA graph with indirect copy with FP16 input.
  Graph Capture(half const **input, float *host_out) override {
    SetCaptureStates(input, host_out, false);
    return Graph();
  }

 protected:
  CudaEnv &Env() { return cuda_env_; }

  // We have same method in the AbstractModel - should probably move to the ModelInstance
  static std::conditional_t<std::is_same_v<TData, float>, std::vector<float> const &, std::vector<TData>> Convert(
      std::vector<float> const &input) {
    if constexpr (std::is_same_v<TData, float>) {
      return input;
    } else {
      static_assert(std::is_same_v<TData, half>);
      return cex::FloatToHalf(input.data(), input.size());
    }
  }

  template <typename IType>
  void SetCaptureStates(IType const **input, float *host_out, bool stage) {
#if GRAPH_CAPTURE_DISABLE == 1
    // is using graph capture API
    is_capturing_ = false;
#else
    is_capturing_ = true;
#endif
    stage_ = stage;
    is_fp32_input_ = (sizeof(IType) == sizeof(float));

    if (stage) {
      host_input_ptr_ = (float *)*input;
    } else {
      // host_input_addr_used by Postprocess
      host_input_addr_ = reinterpret_cast<float const **>(input);
      host_input_ptr_ = nullptr;
      pinned_input_addr_.get()[0] = (void *)input[0];
    }
    host_output_ptr_ = host_out;
  }

  // For a model whose kernel reads the step's input through KernelInputPtr() (call in its constructor): with --gdrcopy
  // the input goes to a buffer in device memory before each step's signal instead of being read over PCIe.
  // Only with GDRCOPY_DEVICE_INPUT (above).
  void UseDeviceInput() {
#if GDRCOPY_DEVICE_INPUT
    if (Env().UseGdrcopy() && !input_gdr_buffer_ptr_)
      input_gdr_buffer_ptr_ = std::make_unique<gdrcopy_utils::GdrcopyBuffer<TData>>(num_features_);
#endif
  }

#if DISABLE_ATOMIC_EXCHANGE == 1
  void DeviceSignalSwitchReset() { device_signal_switch_ = SIGNAL_SWITCH_INIT_VALUE; }
  uint32_t &DeviceSignalSwitch() { return device_signal_switch_; }
#endif

 private:
  template <typename TInput>
  void InputImpl(TInput const **host_addr) {
    is_fp32_input_ = (sizeof(TInput) == sizeof(float));

    if (is_ping_pong_) StateCheck(stage_);

    host_input_ptr_ = (float *)host_addr[0];
    host_input_addr_ = (float const **)host_addr;

    StateCheck(!(stage_ && is_fp32_input_ && data_kind_ != ssgemv::DATA_TYPE_FP32));

    // params is bound to pinned_input_addr_
    //*pinned_input_addr_ = (void *) host_input_ptr_;
    //_mm_stream_si32
    // https://learn.microsoft.com/en-us/cpp/intrinsics/mm-stream-ss?view=msvc-170
    //_mm_stream_si64((long long *)*pinned_input_addr_, (long long)host_input_ptr_);
    //_mm_sfence();
    // We are setting the input pointer while the kernel is spinning
    // need to make sure there is no race condition

    if (!stage_) {
      static_assert(sizeof(void *) == 8);
      _mm_stream_si64((long long *)pinned_input_addr_.get(), reinterpret_cast<intptr_t>(host_input_ptr_));
      _mm_sfence();
    }
  }

  static constexpr ssgemv::Data_kind data_kind_ =
      (std::is_same_v<TData, float>) ? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

  CudaEnv &cuda_env_;

  std::uint32_t const num_features_;

#if AARCH64_GH == 1 && GH_PAGED_INPUT == 1
  cex::PagedPtr<data_type> pinned_x_;
#else
  cex::PinnedPtr<data_type> pinned_x_;
#endif
  float *host_input_ptr_{nullptr};

  ModelInterface::Graph *graph_{nullptr};
  // required if stage == 0
  cex::PinnedPtr<void *> pinned_input_addr_;
  // We store input pointer
  // in Input() or in Capture() such that Postprocess or LstmForward can use it
  float const **host_input_addr_{nullptr};

  // flag for ingesting full windows instead of individual samples of sliding windows
  bool const full_windows_{false};
  bool const single_step_{false};
  bool const is_ping_pong_{false};
  // We keep output pointer from Capture(), such that
  // we can use it without graphs
  float *host_output_ptr_{nullptr};
  bool is_fp32_input_{true};
  bool is_output_ready_{false};
  // we should also remember we are in the capturing state
  bool is_capturing_{false};
  // kernel implementation is different depending on the input staging
  bool stage_{false};

  // Aggregation result is written here
#if AARCH64_GH == 1
  cex::PagedPtr<float> pinned_out_;
  cex::PagedPtr<int> sig_device_;
#else
  cex::PinnedPtr<float> pinned_out_;
  cex::PinnedPtr<int> sig_device_;
#endif

  gdrcopy_utils::GdrcopyBufferPtr<uint> sig_device_gdr_buffer_ptr_;
  // UseDeviceInput() with --gdrcopy: the step's input in device memory, written by LstmForward on staged runs
  gdrcopy_utils::GdrcopyBufferPtr<TData> input_gdr_buffer_ptr_;

#if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_signal_switch_{SIGNAL_SWITCH_INIT_VALUE};
#endif
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_MODELPERSISTENTBASE_CUH_
