/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NVLSTMINF_MODELASSINGLESTEPGEMV_CUH
#define NVLSTMINF_MODELASSINGLESTEPGEMV_CUH

#include <type_traits>

#include <thread>
#include <atomic>

#include "ModelPersistentBase.cuh"
#include "ModelSingleStep.cuh"
#include "CudaEnv.cuh"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "cuda_helpers.cuh"
#include "ss_gemv_impl/kernel_params.h"
#include "ss_gemv_impl/gemv_lstm.h"

#define PRECOMPUTE_OVERLAP 0
#define PRECOMPUTE_SEPARATE_THREAD 0

namespace nvidia::quant::lowlat_infer {

// Supported data types are: float, half
template <typename TData = float>
class ModelAsSingleStepGemv : public ModelPersistentBase<TData> {
 public:
  using params_base_type = std::unique_ptr<ssgemv::Lstm_params_base>;
  using data_type = TData;
  using Base = ModelPersistentBase<TData>;
  using Base::Convert;
  using Base::Env;
  using Base::HostSignal;
  using Base::DeviceSignal;
  using Base::SignalReleaseDevice;

  using lstm_s_params_type = ssgemv::Lstm_s_bpt_params;
  using lstm_m_params_type = ssgemv::Lstm_m_tbc_t_params;
  using lstm_l_params_type = ssgemv::Lstm_l_dev_t_params;

  ModelAsSingleStepGemv(CudaEnv &cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                  ModelDef const &model_def,
                  ModelConfig const &config = {});

  ModelAsSingleStepGemv(CudaEnv &cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                  ModelDef const &model_def,
                  std::uint32_t batch_size,
                  ModelConfig const &config = {})
      : ModelAsSingleStepGemv(cuda_env, model_def, config) {
    StateCheck(batch_size == 1);
  }

  ~ModelAsSingleStepGemv();

  // Indicates whether the model ingests sliding windows or full windows. 
  [[nodiscard]] bool Sliding() const override { return model_singe_step_->Sliding(); }

  // Indicates whether this is a PingPong model.
  [[nodiscard]] bool IsPingPong() const override { return false; }

  // Copy the initial steps - 1 onto the device.
  void InitX(util::Tensor32 const &t) override { 
    // Make sure we lock CPU barrier
    ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)HostSignal(), 0u);
    // set GPU kernel to reset its buffer counter.
    Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE);
#if DISABLE_ATOMIC_EXCHANGE == 1
  Base::DeviceSignalSwitchReset();
  //*DeviceSignal() = 0u;
#endif
    model_singe_step_->InitX(t); 
    odd_step_ = false;
  };

  // Copy the inference results to the host. The `wait` arguments determines whether this method blocks until the
  // async copy operation has finished.
  void OutputImpl(float *host_out, bool wait, bool shift) { 
    Base::Output(host_out, wait, shift);
    if (shift) {
    odd_step_ = !odd_step_;
    }
  }

  void Output(float *host_out, bool wait, bool shift) override { 
    if (spawn_thread_) {
      // here we expect that previous Output call was completed
      output_params_.out.store(host_out, std::memory_order_release);
      output_params_.wait.store(wait, std::memory_order_release);
      output_params_.shift.store(shift, std::memory_order_release);
      thread_output_.store(true, std::memory_order_release);
    } else {
      OutputImpl(host_out, wait, shift);
    }
  }

  bool OutputReady() override { 
    if (spawn_thread_) {
      return !thread_output_.load(std::memory_order_acquire);
    } else {
      return Base::OutputReady();
    }
  }

  void KernelLaunch() override {

    auto custom_stream = cuda_env_.CustomStream();

    switch(num_features_) {
      case LSTM_S_NUM_INPUTS: ssgemv::run_lstm_s_bpt(*static_cast<lstm_s_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
      case LSTM_M_NUM_INPUTS: ssgemv::run_lstm_m_tbc_t(*static_cast<lstm_m_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
      case LSTM_L_NUM_INPUTS: ssgemv::run_lstm_l_dev_t(*static_cast<lstm_l_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
      default: break;
    }
  }

  using ModelInterface::Output;

  // Capture the full inference CUDA graph.
  ModelInterface::Graph Capture(float const *input, float *host_out, bool sequence_major) override { 
    Base::SetCaptureStates(&input, host_out, true);
    return model_singe_step_->Capture(input, pinned_ref_out_.get(), sequence_major); }

  // Capture the full inference CUDA graph with FP16 input.
  ModelInterface::Graph Capture(half const *input, float *host_out, bool sequence_major) override { 
    Base::SetCaptureStates(&input, host_out, true);
    return model_singe_step_->Capture(input, pinned_ref_out_.get(), sequence_major); }

  // Capture the full inference CUDA graph with indirect copy.
  ModelInterface::Graph Capture(float const **input, float *host_out) override {
    Base::SetCaptureStates(input, host_out, false);
    return model_singe_step_->Capture(input, pinned_ref_out_.get()); }

  // Capture the full inference CUDA graph with indirect copy with FP16 input.
  ModelInterface::Graph Capture(half const **input, float *host_out) override {
    Base::SetCaptureStates(input, host_out, false);
    return model_singe_step_->Capture(input, pinned_ref_out_.get()); }

  void PreProcess() override;

  bool PreProcessReady() override {
    if (spawn_thread_) {
      return !thread_preprocess_.load(std::memory_order_acquire);
    } else {
      return true;
    }
  }

  void PostProcess() override;

  bool PostProcessReady() override;

  void PostProcessWait() override;

  // Indicates that the last graph has been launched.
  bool Launched() override {
    throw std::runtime_error("Not Implemented.");
  }

  // Store the input without running the inference (for sliding windows)
  void StoreInput() override { model_singe_step_->StoreInput(); }

  ssgemv::Lstm_params_base* GetParamsPtr(){ return params_ptr_.get(); }

  void Finalize () override;

  private:

    void ThreadMain();

    void PreProcessFirstImpl();

    void PreProcessImpl();

    void PreCompute();

    template <typename Params>
    void InitParams (Params& params);

    // Binds params pointers to the internal array
    void InitParams () override;

    void LaunchGemvLstm();

    CudaEnv &Env() { return cuda_env_; }

    const ssgemv::Data_kind data_kind_ = (std::is_same_v<TData, float>)? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

    CudaEnv &cuda_env_;

    ModelConfig config_;

    std::uint32_t const num_layers_, num_steps_, num_units_, num_features_;

    // Aggregation result is written here
#if AARCH64_GH == 1
    cex::PagedPtr<float> pinned_ref_out_;
#else
    cex::PinnedPtr<float> pinned_ref_out_;
#endif

    std::unique_ptr<ModelSingleStep<TData>> model_singe_step_;

    // Pointers to the device memory with weight matrices applied to x
    std::vector<cex::DevicePtr<data_type>> dev_input_weights_;

    // Pointers to the device memory with biases
    std::vector<cex::DevicePtr<data_type>> dev_biases_;

    // Weights for the dot product
    // float for now
    cex::DevicePtr<float> dev_mul_;

    // bias for final addition
    float const bias_;

    // LSTM-C implementation uses it
    cex::DevicePtr<data_type> dev_input_buf_;

    params_base_type params_ptr_;

    // If we capture SingleStep graph, dev_cxs_in_/dev_cxs_out_ pointers
    // are not swapped in Aggregate() anymore, so we need to remember to
    // swap the pointers in SingleStepGemv explicitly.
    // indicates that the next step is odd
    bool odd_step_{false};

    //bool first_preprocess_{true};
    std::atomic_bool first_preprocess_{true};

    // DEBUG!!!
    //std::vector<cex::DevicePtr<data_type>> dev_cxs_in_;

    bool spawn_thread_{true};

    std::thread thread_;
    std::atomic_bool thread_exit_{false};
    std::atomic_bool thread_preprocess_{false};
    std::atomic_bool thread_output_{false};
#if PRECOMPUTE_SEPARATE_THREAD == 1
    std::atomic_bool thread_postprocess_{false};
#endif

    struct {std::atomic<float*> out; std::atomic_bool wait; std::atomic_bool shift;} output_params_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NVLSTMINF_MODELASSINGLESTEPGEMV_CUH
