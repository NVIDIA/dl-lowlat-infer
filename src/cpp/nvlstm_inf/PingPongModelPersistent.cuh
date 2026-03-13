/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_PINGPONGMODELPERSISTENT_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_PINGPONGMODELPERSISTENT_CUH_

#include <cstdint>
#include <random>

#include "ModelPersistentBase.cuh"
#include "CudaEnv.cuh"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "util.h"
#include "ss_gemv_impl/gemv_lstm.h"
// for scalar product
#include "cuda_copy.cuh"

namespace nvidia::quant::lowlat_infer {

// Supported data types are: float, half
template <typename TData = float>
class PingPongModelPersistent : public ModelPersistentBase<TData> {
 public:
    using data_type = TData;
    using Base = ModelPersistentBase<TData>;
    
    using Base::Env;
    using Base::SignalReleaseDevice;

    using lstm_s_params_type = ssgemv::Lstm_s_bpt_params;
    using lstm_m_params_type = ssgemv::Lstm_m_bpt_params;
    using lstm_l_params_type = ssgemv::Lstm_l_bpt_params;

    PingPongModelPersistent(CudaEnv &cuda_env, ModelDef const &model_def, std::uint32_t batch_size, ModelConfig const &config)
        : Base{cuda_env, model_def, 1, false, true}  // full_windows=false, single_step=true
        , cuda_env_{cuda_env}
        , num_steps_(model_def.NumSteps())
        , num_features_(model_def.NumFeatures()) {
      StateCheck(batch_size == 1);
      StateCheck(model_def.IsPingPong());

      cuda_env.SetPrecomputeContext();

      auto stream = cuda_env.Stream();

      switch(num_features_) {
        case LSTM_S_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_s_params_type>(); break;
        case LSTM_M_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_m_params_type>(); break;
        case LSTM_L_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_l_params_type>(); break;
        default: break;
      }

      params_ptr_->always_spin = 1;
      params_ptr_->max_smem_size = config.max_shared_mem_per_block;
      params_ptr_->sm_count = config.sm_count;
      params_ptr_->sm_arch = config.sm_arch;

      CudaCheck(cudaStreamSynchronize(stream));
    }

    using ModelInterface::Output;

    ~PingPongModelPersistent() {
      if (!spawn_thread_) {
        Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE);
        cuda_env_.SetModelContext();
        CudaCheck(cudaStreamSynchronize(cuda_env_.CustomStream()));
        is_kernel_launched_ = false;
        cuda_env_.SetPrecomputeContext();
      }
    }

    // Indicates whether this is a PingPong model.
    [[nodiscard]] bool IsPingPong() const override { return true; };

    template <typename Params>
    void InitParams (Params& params) {

      Base::InitParams();

      params.host_ld_complete = Base::HostSignal();
      params.gmem_ld_complete = Base::DeviceSignal();
      params.use_gdrcopy = Env().UseGdrcopy();
      // We always mountain a pinned buffer where we collect inputs for all time steps
      params.host_addr = nullptr;

      params.gmem_input = reinterpret_cast<void *>(Base::HostInputPtr());
    }

    void InitParams () override {
      switch(num_features_) {
        case LSTM_S_NUM_INPUTS: InitParams(*static_cast<lstm_s_params_type *>(params_ptr_.get())); break;
        case LSTM_M_NUM_INPUTS: InitParams(*static_cast<lstm_m_params_type *>(params_ptr_.get())); break;
        case LSTM_L_NUM_INPUTS: InitParams(*static_cast<lstm_l_params_type *>(params_ptr_.get())); break;
        default: break;
      }
    }

    void InitX(util::Tensor32 const &t) override {
      if (is_kernel_launched_) {
#if AS_STATES_LD_BAR_IN_NULL_MODEL == 0
        ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);
#else
        // Make sure we lock CPU barrier
        ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)  (uint *)Base::HostSignal(), 0u);
#endif
        // set GPU kernel to reset its buffer counter.
        Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE);
#if DISABLE_ATOMIC_EXCHANGE == 1
        Base::DeviceSignalSwitchReset();
#endif
      }
    }

    void KernelLaunch() override {
      // Get the stream for the inference green context
      cuda_env_.SetModelContext();
      auto custom_stream = cuda_env_.CustomStream();

      switch(num_features_) {
        case LSTM_S_NUM_INPUTS: ssgemv::run_lstm_s_ping_pong_bpt(*static_cast<lstm_s_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
        case LSTM_M_NUM_INPUTS: ssgemv::run_lstm_m_ping_pong_bpt(*static_cast<lstm_m_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
        case LSTM_L_NUM_INPUTS: ssgemv::run_lstm_l_ping_pong_bpt(*static_cast<lstm_l_params_type *>(params_ptr_.get()), data_kind_, custom_stream); break;
        default: break;
      }
      is_kernel_launched_ = true;
      cuda_env_.SetPrecomputeContext();
    }

    void StoreInput() override { }

    // Indicates that the last graph has been launched.
    bool Launched() override {
      throw std::runtime_error("Not Implemented.");
    }

    void PreProcessFirstImpl() {

      //Base::PreProcess() begin
      InitParams();

      KernelLaunch();

      // Wait till data is loaded by the kernel
      ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);
      
 #if AS_STATES_LD_BAR_IN_NULL_MODEL == 1
      // can start load the states from gmem
 #if DISABLE_ATOMIC_EXCHANGE == 1
     Base::template SignalReleaseDevice<true>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
 #else
     Base::template SignalReleaseDevice<false>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
 #endif
     // GPU loads the states into registers and signals CPU...
     // wait for GPU to signal
     // can just use poll_load since HOST_SPIN_0_UNTIL_SIGNAL_VALUE == HOST_SPIN_1_WHILE_SIGNAL_VALUE
     ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
 #endif
    }

    void PreProcess() override { 
      if (first_preprocess_) {

        if (spawn_thread_) {
          thread_ = std::thread{&PingPongModelPersistent<TData>::ThreadMain, this};
    
          auto core_id = cuda_env_.GetCpuCoreId();
          if (core_id >= 0) {
            util::SetAffinity(thread_.native_handle(), core_id);
          }
        }
        thread_preprocess_.store(true, std::memory_order_release);
      } else if (spawn_thread_) {
        thread_preprocess_.store(true, std::memory_order_release);
      }
      else {
        PreProcessImpl();
      }
    }

    bool PreProcessReady() override {
      if (spawn_thread_) {
        return !thread_preprocess_.load(std::memory_order_acquire);
      } else {
        return true;
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

    void Finalize () override {
      if (spawn_thread_) {
        // reset host signal
        ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)Base::HostSignal(), 0u);
        // shutdown the kernel

        Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE);
        // wait till the shutdown
        ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);

        cuda_env_.SetModelContext();
        CudaCheck(cudaStreamSynchronize(cuda_env_.CustomStream()));
        cuda_env_.SetPrecomputeContext();
        is_kernel_launched_ = false;
        thread_exit_.store(true, std::memory_order_release);
        thread_.join();
      }  
    }

  private:

  void  PreProcessImpl() {
    if (first_preprocess_) {
      PreProcessFirstImpl();
      first_preprocess_ = false;
      return;
    }
    // reset the signal, since it was altered by the output
    static_assert(HOST_SPIN_1_WHILE_SIGNAL_VALUE == HOST_SPIN_0_UNTIL_SIGNAL_VALUE);

#if AS_STATES_LD_BAR_IN_NULL_MODEL == 1
    // can start load the states from gmem
#if DISABLE_ATOMIC_EXCHANGE == 1
    Base::template SignalReleaseDevice<true>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#else
    Base::template SignalReleaseDevice<false>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#endif
    // GPU loads the states into registers and signals CPU...
#endif
    // wait for GPU to signal
    ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
  }

    void OutputImpl(float *host_out, bool wait, bool shift) {
      Base::Output(host_out, wait, shift);
    }

    void ThreadMain() {
      while (!thread_exit_.load(std::memory_order_acquire)) {
        if (thread_preprocess_.load(std::memory_order_acquire)) {
          PreProcessImpl();
          thread_preprocess_.store(false, std::memory_order_release);
        } else if (thread_output_.load(std::memory_order_acquire)) {
          OutputImpl(output_params_.out.load(std::memory_order_acquire),
                    output_params_.wait.load(std::memory_order_acquire),
                    output_params_.shift.load(std::memory_order_acquire));
          thread_output_.store(false, std::memory_order_release);
        }
      }
      Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE); 
    }

    bool spawn_thread_{true};

    std::atomic_bool first_preprocess_{true};

    std::thread thread_;
    std::atomic_bool thread_exit_{false};
    std::atomic_bool thread_preprocess_{false};
    std::atomic_bool thread_output_{false};
    struct {std::atomic<float*> out; std::atomic_bool wait; std::atomic_bool shift;} output_params_;

    CudaEnv &cuda_env_;

    std::uint32_t num_features_, num_steps_;
    const ssgemv::Data_kind data_kind_ = (std::is_same_v<TData, float>)? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

    std::unique_ptr<ssgemv::Lstm_params_base> params_ptr_;

    bool pre_compute_{true};

    bool is_kernel_launched_ {false};
  };

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_PINGPONGMODELPERSISTENT_CUH_
