/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <iostream>

#include "ModelAsSingleStepGemv.cuh"
#include "fp16_reg_util.cuh"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

/// The model's weights on the device in its kernel's layout, and its precompute (see the declaration).
template <typename TData>
ModelAsSingleStepGemv<TData>::ModelAsSingleStepGemv(
    CudaEnv &cuda_env, const ModelDef &model_def, const ModelConfig &config)
    : Base{cuda_env, model_def, 1, false, true}  // full_windows=false, single_step=true
      ,
      cuda_env_{cuda_env},
      bias_{model_def.HasBias() ? model_def.Bias() : 0.0f},
      num_layers_(model_def.NumLayers()),
      num_steps_(model_def.NumSteps()),
      num_units_(model_def.NumUnits()),
      num_features_(model_def.NumFeatures()) {
  static_assert(NUM_GATES == kNumGates);
  Base::UseDeviceInput();

  // DEBUG!!!
  {
    // dev_cxs_in_.resize(num_layers_);
    // for (int layer = 0; layer < num_layers_; layer++) {
    //   dev_cxs_in_[layer] = cex::Malloc<data_type>(num_units_);
    // }
  }

  switch (num_features_) {
    case LSTM_S_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_s_params_type>();
      break;
    case LSTM_M_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_m_params_type>();
      break;
    case LSTM_L_NUM_INPUTS:
      params_ptr_ = std::make_unique<lstm_l_params_type>();
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_features: " + std::to_string(num_features_) +
          ". Expected: " + std::to_string(LSTM_S_NUM_INPUTS) + ", " + std::to_string(LSTM_M_NUM_INPUTS) + ", or " +
          std::to_string(LSTM_L_NUM_INPUTS));
  }

  config_ = config;
  config_.pre_compute = false;

#if USE_SINGLE_STEP_RESIDENT
  if (config_.precompute_resident) {
    // the inference kernel waits on the precompute's ready word: the FP16 register kernel
    StateCheck(Fp16RegS() || Fp16RegM());
    // the compile-time overlap variant runs the cuBLAS precompute, which the resident path does not create
    StateCheck(PRECOMPUTE_OVERLAP == 0);
    // the kernel takes a change of the device signal word as the go and never resets it: the alternating go
    static_assert(DISABLE_ATOMIC_EXCHANGE == 1, "USE_SINGLE_STEP_RESIDENT sets the switch signal protocol");
    // the slots' R * h is gate-major, which the register kernel reads only with GEMM_H_VECTORIZED == 0
    static_assert(GEMM_H_VECTORIZED == 0, "the resident precompute writes R * h gate-major");
    cuda_env.SetPrecomputeContext();
    resident_stream_ = cuda_env.CreatePrecomputeStream();
    resident_ = std::make_unique<single_step::SingleStepResident>(
        model_def,
        single_step::SingleStepResidentConfig{
            .step_splits = config_.precompute_step_splits, .use_gdrcopy = cuda_env.UseGdrcopy()},
        resident_stream_.get());
  }
#else
  StateCheck(!config_.precompute_resident);  // main.cu rejects it in a build without USE_SINGLE_STEP_RESIDENT
#endif
  if (!Resident()) model_singe_step_ = std::make_unique<ModelSingleStep<TData>>(cuda_env, model_def, config_);

  cuda_env.SetPrecomputeContext();

  auto stream = cuda_env.Stream();

  // Populate descriptor objects
  auto const num_units_gates = num_units_ * kNumGates;

#if AARCH64_GH == 1
  pinned_ref_out_ = cex::MallocPaged<float>(1);
#else
  pinned_ref_out_ = cex::MallocHost<float>(1);
#endif

  dev_input_weights_.resize(num_layers_);
  dev_biases_.resize(num_layers_);

  // FP16 register-resident kernel for lstm_m: the words its CTAs exchange (the weights below in its tile layout).
  if (Fp16RegM()) dev_fp16_reg_xchg_ = fp16_reg_device_exchange(ssgemv::FP16_REG_LSTM_M_DIMS);

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    dev_input_buf_ = cex::Malloc<data_type>(2 * std::max(num_features_, num_units_));
  }

  std::vector<TData> tmp_h(kNumGates * std::max(num_units_, num_features_) * num_units_);

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    std::uint32_t num_inputs = (layer == 0) ? num_features_ : num_units_;

    // weights
    dev_biases_[layer] = cex::Malloc<data_type>(num_units_gates);

    auto &&matrix_W_conv = Convert(model_def.LstmW().at(layer).Data());
    if (Fp16RegS() || Fp16RegM()) {
      const ssgemv::Fp16_reg_model_dims dims{int(num_features_), int(num_units_), int(num_layers_)};
      dev_input_weights_[layer] = fp16_reg_device_weights(dims, int(layer), matrix_W_conv.data(), stream);
    } else {
      dev_input_weights_[layer] = cex::Malloc<data_type>(num_units_gates * num_inputs);
      ssgemv::weights_permute(tmp_h.data(), matrix_W_conv.data(), kNumGates, num_inputs, num_units_);
      cex::Copy(dev_input_weights_[layer].get(), tmp_h.data(), num_units_gates * num_inputs, stream);
    }

    // biases
    std::vector<float> biases_reduced(kNumGates * num_units_);
    for (std::size_t idx_gate = 0; idx_gate < kNumGates; ++idx_gate) {
      auto const gate = kGates.at(idx_gate);
      const float *biases1 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, false);
      const float *biases2 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, true);
      for (std::size_t k = 0; k < num_units_; ++k) {
        biases_reduced[idx_gate * num_units_ + k] = biases1[k] + biases2[k];
      }
    }
    auto &&bias_conv = Convert(biases_reduced);
    ssgemv::gate_transpose(tmp_h.data(), bias_conv.data(), kNumGates * num_units_, 1);
    cex::Copy(dev_biases_[layer].get(), tmp_h.data(), kNumGates * num_units_, stream);

    // dot product
    dev_mul_ = cex::Malloc<float>(num_units_);
    cex::Copy(dev_mul_, model_def.MatMul().Data().data(), num_units_, stream);
  }  // end layer loop

  GetParamsPtr()->always_spin = 1;
  GetParamsPtr()->max_smem_size = config_.max_shared_mem_per_block;
  GetParamsPtr()->sm_count = config_.sm_count;
  GetParamsPtr()->sm_arch = config_.sm_arch;

  CudaCheck(cudaStreamSynchronize(stream));
};

template <typename TData>
void ModelAsSingleStepGemv<TData>::Finalize() {
  if (spawn_thread_) {
    // reset host signal
    ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)Base::HostSignal(), 0u);
    // shutdown the kernel
    Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE);
    // wait till the shutdown
    ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);

    cuda_env_.SetModelContext();
    CudaCheck(cudaStreamSynchronize(cuda_env_.CustomStream()));
    cuda_env_.SetPrecomputeContext();

    thread_exit_.store(true, std::memory_order_release);
    thread_.join();
  }
#if USE_SINGLE_STEP_RESIDENT
  // the resident precompute too, before anything synchronizes the device; its memory is freed with the model, once
  // every instance's precompute has stopped
  if (resident_) resident_->Stop();
#endif
}

template <typename TData>
ModelAsSingleStepGemv<TData>::~ModelAsSingleStepGemv() {
  if (!spawn_thread_) {
    Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE);
    cuda_env_.SetModelContext();
    CudaCheck(cudaStreamSynchronize(cuda_env_.CustomStream()));
  }
#if USE_SINGLE_STEP_RESIDENT
  if (resident_) resident_->Stop();
#endif
  cuda_env_.SetPrecomputeContext();
  CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));
}

/// The kernel's parameters (see the declaration).
template <typename TData>
template <typename Params>
void ModelAsSingleStepGemv<TData>::InitParams(Params &params) {
  Base::InitParams();

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    params.gmem_weights[layer] = dev_input_weights_[layer].get();
    params.gmem_biases[layer] = dev_biases_[layer].get();
    if (!model_singe_step_) continue;  // the resident precompute's state slots, below

    const auto &dev_gemms = model_singe_step_->Gemms();
    std::size_t leading_dim = dev_gemms[layer].LeadingDim();

    params.gmem_gemm_h[layer] = (void *)(dev_gemms[layer].Get() + (num_steps_ - 1) * leading_dim);

    const auto &dev_cxs_in = model_singe_step_->CxIn();
    const auto &dev_cxs_out = model_singe_step_->CxOut();

    leading_dim = dev_cxs_in[layer].LeadingDim();
    params.gmem_c_in[layer] = (void *)(dev_cxs_in[layer].Get() + (num_steps_ - 1) * leading_dim);
    params.gmem_c_out[layer] = (void *)(dev_cxs_out[layer].Get() + (num_steps_ - 1) * leading_dim);
    // DEBUG
    // {
    //   cudaMemcpyAsync(dev_cxs_in_[layer].get(), (dev_cxs_in[layer].Get() + (num_steps_ - 1) * leading_dim),
    //                   num_units_*sizeof(TData), cudaMemcpyDefault, cuda_env_.Stream());
    //   params.gmem_c_in[layer] = (void *) dev_cxs_in_[layer].get();
    // }
    //     CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));
  }
#if USE_SINGLE_STEP_RESIDENT
  if (resident_) {
    // the resident precompute's state slots, by step parity, and its ready word
    for (std::size_t layer = 0; layer < num_layers_; ++layer) {
      params.gmem_gemm_h[layer] = resident_->GemmH(0, static_cast<int>(layer));
      params.gmem_gemm_h_odd[layer] = resident_->GemmH(1, static_cast<int>(layer));
      params.gmem_c_in[layer] = resident_->CellState(0, static_cast<int>(layer));
      params.gmem_c_out[layer] = resident_->CellState(1, static_cast<int>(layer));
    }
    params.gmem_states_ready = resident_->ReadyWord();
  }
#endif
  params.dot_prod_bias = bias_;
  params.gmem_dot_prod_weights = (void *)dev_mul_.get();
  params.host_ld_complete = Base::HostSignal();
  params.gmem_ld_complete = Base::DeviceSignal();
  params.use_gdrcopy = Env().UseGdrcopy();

  // Todo move to base
  params.host_addr = (Base::IsStaging()) ? nullptr : Base::PinedInputAddr();

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    params.gmem_h_out = dev_input_buf_.get();
  }

  params.gmem_xchg = dev_fp16_reg_xchg_.get();

  Base::SetKernelInput(params);
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::InitParams() {
  switch (num_features_) {
    case LSTM_S_NUM_INPUTS:
      InitParams(*static_cast<lstm_s_params_type *>(params_ptr_.get()));
      break;
    case LSTM_M_NUM_INPUTS:
      InitParams(*static_cast<lstm_m_params_type *>(params_ptr_.get()));
      break;
    case LSTM_L_NUM_INPUTS:
      InitParams(*static_cast<lstm_l_params_type *>(params_ptr_.get()));
      break;
    default:
      break;
  }
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::PreCompute() {
  if (Base::GraphPtr() == nullptr) {
    model_singe_step_->SetPinnedOut(pinned_ref_out_.get());
    model_singe_step_->PreProcess();
    if (Base::IsStaging()) {
      // Unlikely to happen without capturing the graph
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputPtr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const *>(Base::HostInputPtr()));
      }
    } else {
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputAddr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const **>(Base::HostInputAddr()));
      }
    }
#if PRECOMPUTE_OVERLAP == 0
    model_singe_step_->PostProcess();
#endif
  } else {
    model_singe_step_->PreProcess();
    model_singe_step_->PreProcessWait();
    model_singe_step_->Launch(*Base::GraphPtr());
#if PRECOMPUTE_OVERLAP == 0
    model_singe_step_->PostProcess();
#endif
  }
}

/// The first step's preprocess (see the declaration).
template <typename TData>
void ModelAsSingleStepGemv<TData>::PreProcessFirstImpl() {
  cuda_env_.SetPrecomputeContext();

  CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));

  // Base::PreProcess() begin
  InitParams();

  if (num_units_ == LSTM_M_NUM_UNITS && !Fp16RegM()) {
    // last check inside of a green context if initiated
    StateCheck(ssgemv::verify_lstm_m_cluster_size_supported(GetParamsPtr()->sm_arch, sizeof(TData)));
  }

  cuda_env_.SetModelContext();
  KernelLaunch();
  cuda_env_.SetPrecomputeContext();

#if USE_SINGLE_STEP_RESIDENT
  // the resident precompute stays from here on, as the inference kernel does; started after the inference kernel's
  // launch, whose first launch loads its module (lazy loading), which would wait for a kernel that never ends
  if (resident_) resident_->Start();
#endif

  // Wait till data is loaded by the kernel
  ssgemv::cpu_pingpong_handles::poll_atomic<uint>((uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, 0u);

  if (Resident()) {
    // the states come from the resident precompute on the device: only the output's signal to arm
    ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)Base::HostSignal(), HOST_SPIN_1_WHILE_SIGNAL_VALUE);
    return;
  }

  // DEBUG
  {
    // for (int layer = 0; layer < num_layers_; ++layer) {
    //   printf("Debug Update c state %d\n", layer);
    //   const auto& dev_cxs_in = (odd_step_ && Base::IsCapturing())? model_singe_step_->CxOut() :
    //   model_singe_step_->CxIn(); std::size_t leading_dim = dev_cxs_in[layer].LeadingDim();
    //   cudaMemcpyAsync(dev_cxs_in_[layer].get(), (dev_cxs_in[layer].Get() + (num_steps_ - 1) * leading_dim),
    //                   num_units_*sizeof(TData), cudaMemcpyDefault, cuda_env_.Stream());
    // }
    // CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));
  }

#if PRECOMPUTE_OVERLAP == 1
  PreCompute();
  model_singe_step_->PostProcessWait();
  CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));
#endif

#if AS_STATES_LD_BAR == 1
  // can start load the states from gmem
#if DISABLE_ATOMIC_EXCHANGE == 1
  Base::template SignalReleaseDevice<true>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#else
  Base::template SignalReleaseDevice<false>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#endif
  // GPU loads the states into registers and signals CPU...
  // wait for GPU to signal
  // can just use poll_load since HOST_SPIN_0_UNTIL_SIGNAL_VALUE == HOST_SPIN_1_WHILE_SIGNAL_VALUE
  ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
      (uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
#endif
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::PreProcessImpl() {
  if (first_preprocess_) {
    PreProcessFirstImpl();
    first_preprocess_ = false;
    return;
  }

#if USE_SINGLE_STEP_RESIDENT
  if (resident_) {
    // the next step's states: the resident precompute's last pass published them (untimed, as the release of the states
    // is otherwise; the pass may still be advancing the other windows), then only the output's signal to re-arm
    resident_->WaitPublished();
    ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)Base::HostSignal(), HOST_SPIN_1_WHILE_SIGNAL_VALUE);
    return;
  }
#endif

  CudaCheck(cudaStreamSynchronize(cuda_env_.Stream()));
#if PRECOMPUTE_OVERLAP == 1
  model_singe_step_->PostProcessWait();
#endif

  // reset the signal, since it was altered by the output
  static_assert(HOST_SPIN_1_WHILE_SIGNAL_VALUE == HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
#if AS_STATES_LD_BAR == 0
  ssgemv::cpu_pingpong_handles::signal_release<uint>((uint *)Base::HostSignal(), HOST_SPIN_1_WHILE_SIGNAL_VALUE);
#endif

#if PRECOMPUTE_OVERLAP == 1
  PreCompute();
#endif
#if AS_STATES_LD_BAR == 1
  // can start load the states from gmem
#if DISABLE_ATOMIC_EXCHANGE == 1
  Base::template SignalReleaseDevice<true>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#else
  Base::template SignalReleaseDevice<false>(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);
#endif
  // GPU loads the states into registers and signals CPU...
  // wait for GPU to signal
  ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
      (uint *)Base::HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
#endif

#if PRECOMPUTE_OVERLAP == 0
  StateCheck(Base::PostProcessReady());
#endif
}  // End of preprocess implementation

template <typename TData>
void ModelAsSingleStepGemv<TData>::PreProcess() {
  if (first_preprocess_) {
    if (spawn_thread_) {
      thread_ = std::thread{&ModelAsSingleStepGemv<TData>::ThreadMain, this};

      auto core_id = cuda_env_.GetCpuCoreId();
      if (core_id >= 0) {
        util::SetAffinity(thread_.native_handle(), core_id);
      }
    }
    thread_preprocess_.store(true, std::memory_order_release);
  } else if (spawn_thread_) {
    thread_preprocess_.store(true, std::memory_order_release);
  } else {
    PreProcessImpl();
  }
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::PostProcess() {
  if (Resident()) return;  // the resident precompute gets the input in PostLaunch()
#if PRECOMPUTE_OVERLAP == 0
#if PRECOMPUTE_SEPARATE_THREAD == 1
  if (spawn_thread_) {
    thread_postprocess_.store(true, std::memory_order_release);
  } else {
    PreCompute();
  }
#else
  PreCompute();
#endif
#else
  model_singe_step_->PostProcess();
#endif
};

template <typename TData>
void ModelAsSingleStepGemv<TData>::PostLaunch() {
#if USE_SINGLE_STEP_RESIDENT
  if (resident_) {
    // this step's input to the resident precompute, which publishes the states of the next step
    auto const input = Base::IsStaging() ? static_cast<void const *>(Base::HostInputPtr())
                                         : static_cast<void const *>(Base::HostInputAddr()[0]);
    if (Base::IsFp32Input()) {
      resident_->Step(resident_step_++, static_cast<float const *>(input));
    } else {
      resident_->Step(resident_step_++, static_cast<half const *>(input));
    }
  }
#endif
}

template <typename TData>
bool ModelAsSingleStepGemv<TData>::PostProcessReady() {
  return !model_singe_step_ || model_singe_step_->PostProcessReady();
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::PostProcessWait() {
  if (model_singe_step_) model_singe_step_->PostProcessWait();
}

template <typename TData>
void ModelAsSingleStepGemv<TData>::ThreadMain() {
  while (!thread_exit_.load(std::memory_order_acquire)) {
    if (thread_preprocess_.load(std::memory_order_acquire)) {
      PreProcessImpl();
      thread_preprocess_.store(false, std::memory_order_release);
    } else if (thread_output_.load(std::memory_order_acquire)) {
      OutputImpl(
          output_params_.out.load(std::memory_order_acquire),
          output_params_.wait.load(std::memory_order_acquire),
          output_params_.shift.load(std::memory_order_acquire));
      thread_output_.store(false, std::memory_order_release);
    }
#if PRECOMPUTE_SEPARATE_THREAD == 1
    else if (thread_postprocess_.load(std::memory_order_acquire)) {
      PreCompute();
      thread_postprocess_.store(false, std::memory_order_release);
    }
#endif
  }
  Base::template SignalReleaseDevice<false>(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE);
}

}  // namespace nvidia::quant::lowlat_infer

template class nvidia::quant::lowlat_infer::ModelAsSingleStepGemv<float>;
template class nvidia::quant::lowlat_infer::ModelAsSingleStepGemv<half>;
