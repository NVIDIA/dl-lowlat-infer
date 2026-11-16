/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NVLSTMINF_MODELSINGLESTEPGEMV_CUH
#define NVLSTMINF_MODELSINGLESTEPGEMV_CUH

#include <type_traits>
#include <utility>

#include "CudaEnv.cuh"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "ModelPersistentBase.cuh"
#include "ModelSingleStep.cuh"
#include "cuda_helpers.cuh"
#include "ss_gemv_impl/gemv_lstm.h"
#include "ss_gemv_impl/kernel_params.h"

namespace nvidia::quant::lowlat_infer {

// Supported data types are: float, half
template <typename TData = float>
class ModelSingleStepGemv : public ModelPersistentBase<TData> {
 public:
  using params_base_type = std::unique_ptr<ssgemv::Lstm_params_base>;
  using data_type = TData;
  using Base = ModelPersistentBase<TData>;
  using Base::Convert;
  using Base::DeviceSignal;
  using Base::Env;
  using Base::HostSignal;

  using lstm_s_params_type = ssgemv::Lstm_s_bpt_params;
  using lstm_m_params_type = ssgemv::Lstm_m_tbc_t_params;
  using lstm_l_params_type = ssgemv::Lstm_l_dev_t_params;

  /// FP16 lstm_l models run with the FP16 register-resident kernel (USE_FP16_REG_KERNELS) when its CTAs fit the device.
  static constexpr bool kFp16RegL = USE_FP16_REG_KERNELS && std::is_same_v<TData, half>;
  // --steps-per-graph: each kernel of a graph waits for the host's reset of the device signal word before it signals
  // ready (the switch protocol, which a build with USE_FP16_REG_L_GRAPH has).
  static_assert(!USE_FP16_REG_L_GRAPH || DISABLE_ATOMIC_EXCHANGE == 1, "USE_FP16_REG_L_GRAPH: the switch protocol");

  ModelSingleStepGemv(
      CudaEnv &cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
      ModelDef const &model_def,
      ModelConfig const &config = {});

  ModelSingleStepGemv(
      CudaEnv &cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
      ModelDef const &model_def,
      std::uint32_t batch_size,
      ModelConfig const &config = {})
      : ModelSingleStepGemv(cuda_env, model_def, config) {
    StateCheck(batch_size == 1);
  }

  ~ModelSingleStepGemv() override {
    try {
      DrainGraph();
    } catch (...) {
    }
  }

  void Finalize() override { DrainGraph(); }

  // Indicates whether the model ingests sliding windows or full windows.
  [[nodiscard]] bool Sliding() const override { return model_singe_step_->Sliding(); }

  // Indicates whether this is a PingPong model.
  [[nodiscard]] bool IsPingPong() const override { return false; }

  // 16 or 32
  //  [[nodiscard]] std::uint8_t Precision() const override { return static_cast<std::uint8_t>(sizeof(data_type) * 8); }

  // Copy the initial steps - 1 onto the device.
  void InitX(util::Tensor32 const &t) override {
    DrainGraph();
    model_singe_step_->InitX(t);
    odd_step_ = false;
  };

  // Copy the inference results to the host. The `wait` arguments determines whether this method blocks until the
  // async copy operation has finished.
  void Output(float *host_out, bool wait, bool shift) override {
    Base::Output(host_out, wait, shift);
    if (shift) {
      odd_step_ = !odd_step_;
    }
  }

  void KernelLaunch() override { KernelLaunch(cuda_env_.CustomStream()); }

  void KernelLaunch(cudaStream_t stream) {
    ResetSignalForLaunch();
    LaunchKernel(stream);
  }

  // The host's part of a launch per inference.
  void ResetSignalForLaunch() {
#if DISABLE_ATOMIC_EXCHANGE == 1
    // we need to reset the device signal switch to the initial value
    // since we launch the kernel every time we call PreProcess
    Base::DeviceSignalSwitchReset();
    Base::template SignalReleaseDevice<false>(0);
#endif
  }

  void LaunchKernel(cudaStream_t stream) {
    switch (num_features_) {
      case LSTM_S_NUM_INPUTS:
        ssgemv::run_lstm_s_bpt(*static_cast<lstm_s_params_type *>(params_ptr_.get()), data_kind_, stream);
        break;
      case LSTM_M_NUM_INPUTS:
        ssgemv::run_lstm_m_tbc_t(*static_cast<lstm_m_params_type *>(params_ptr_.get()), data_kind_, stream);
        break;
      case LSTM_L_NUM_INPUTS:
        if constexpr (kFp16RegL) {
          if (fp16_reg_l_) {
            ssgemv::run_lstm_l_fp16_reg_t(*static_cast<lstm_l_params_type *>(params_ptr_.get()), stream);
            break;
          }
        }
        ssgemv::run_lstm_l_dev_t(*static_cast<lstm_l_params_type *>(params_ptr_.get()), data_kind_, stream);
        break;
      default:
        break;
    }
  }

  using ModelInterface::Output;

  // Capture the full inference CUDA graph.
  ModelInterface::Graph Capture(float const *input, float *host_out, bool sequence_major) override {
    Base::SetCaptureStates(&input, host_out, true);
    return CaptureWithKernel([&] { return model_singe_step_->Capture(input, pinned_ref_out_.get(), sequence_major); });
  }

  // Capture the full inference CUDA graph with FP16 input.
  ModelInterface::Graph Capture(half const *input, float *host_out, bool sequence_major) override {
    Base::SetCaptureStates(&input, host_out, true);
    return CaptureWithKernel([&] { return model_singe_step_->Capture(input, pinned_ref_out_.get(), sequence_major); });
  }

  // Capture the full inference CUDA graph with indirect copy.
  ModelInterface::Graph Capture(float const **input, float *host_out) override {
    Base::SetCaptureStates(input, host_out, false);
    return CaptureWithKernel([&] { return model_singe_step_->Capture(input, pinned_ref_out_.get()); });
  }

  // Capture the full inference CUDA graph with indirect copy with FP16 input.
  ModelInterface::Graph Capture(half const **input, float *host_out) override {
    Base::SetCaptureStates(input, host_out, false);
    return CaptureWithKernel([&] { return model_singe_step_->Capture(input, pinned_ref_out_.get()); });
  }

  void PreProcess() override;

  void PostProcess() override;

  bool PostProcessReady() override;

  void PostProcessWait() override;

  // Indicates that the last graph has been launched.
  bool Launched() override { throw std::runtime_error("Not Implemented."); }

  // Store the input without running the inference (for sliding windows)
  void StoreInput() override { model_singe_step_->StoreInput(); }

  ssgemv::Lstm_params_base *GetParamsPtr() { return params_ptr_.get(); }

 private:
  void PreCompute();

  template <typename Params>
  void InitParams(Params &params);

  // Binds params pointers to the internal array
  void InitParams() override;

  void LaunchGemvLstm();

  CudaEnv &Env() { return cuda_env_; }

  const ssgemv::Data_kind data_kind_ = (std::is_same_v<TData, float>) ? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

  CudaEnv &cuda_env_;

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

  // lstm_l only: the kernel's hidden-state output buffer (params.gmem_h_out)
  cex::DevicePtr<data_type> dev_input_buf_;
  /// The FP16 register-resident kernel runs lstm_l (kFp16RegL, and its CTAs fit the device).
  bool fp16_reg_l_ = false;
  /// The words its CTAs exchange (params.gmem_xchg).
  cex::DevicePtr<unsigned long long> dev_fp16_reg_xchg_;

  params_base_type params_ptr_;

  // If we capture SingleStep graph, dev_cxs_in_/dev_cxs_out_ pointers
  // are not swapped in Aggregate() anymore, so we need to remember to
  // swap the pointers in SingleStepGemv explicitly.
  // indicates that the next step is odd
  bool odd_step_{false};

  // The captured precompute graph runs on the GPU after the step's kernel, launched with it on one stream (PreProcess).
  bool graph_launched_{false};

  /// --lstm-l-in-graph (USE_FP16_REG_L_GRAPH): the FP16 register lstm_l kernel is captured at the root of each step's
  /// precompute graph, which also takes layer 0's W * x from it; --steps-per-graph: steps per graph (1 or even).
  bool const in_graph_;
  unsigned const steps_per_graph_;
  /// in_graph_: the graphs by starting parity (owned by the caller of Capture), and
  /// the kernel's layer-0 W * x for the precompute (params.gmem_x_dot). nullptr: not captured yet.
  cudaGraphExec_t graph_with_kernel_[2] = {nullptr, nullptr};
  cex::DevicePtr<data_type> dev_fp16_reg_x_dot_;
  /// The steps of the last graph launched done or under way (0: the next step launches a graph).
  unsigned steps_in_graph_ = 0;

  template <typename CaptureFn>
  ModelInterface::Graph CaptureWithKernel(CaptureFn &&capture) {
    if (!in_graph_) return capture();
#if USE_FP16_REG_L_GRAPH
    // The kernel's parameters of both parities from the buffers as they are before the capture: capturing a step swaps
    // the precompute's cell-state buffers (Aggregate), so that the second graph's would otherwise be the other
    // parity's.
    auto &params = *static_cast<lstm_l_params_type *>(params_ptr_.get());
    lstm_l_params_type params_by_parity[2];
    bool const odd_step = odd_step_;
    for (int p = 0; p < 2; p++) {
      odd_step_ = p != 0;
      InitParams();
      params_by_parity[p] = params;
      params_by_parity[p].gmem_x_dot = dev_fp16_reg_x_dot_.get();  // the same input (the step's) and weights
    }
    odd_step_ = odd_step;
    model_singe_step_->SetLayer0InputProduct(dev_fp16_reg_x_dot_.get());
    model_singe_step_->SetStepsPerGraph(steps_per_graph_);
    model_singe_step_->SetCapturePrologue([this, &params, params_by_parity](cudaStream_t stream, bool odd) {
      params = params_by_parity[odd ? 1 : 0];
      LaunchKernel(stream);
    });
    auto graph = capture();
    params.gmem_x_dot = nullptr;  // a kernel launched on its own leaves it
    model_singe_step_->SetCapturePrologue(nullptr);
    model_singe_step_->SetStepsPerGraph(1);
    model_singe_step_->SetLayer0InputProduct(nullptr);
    StateCheck(graph.size() == 2);
    for (int p = 0; p < 2; p++) graph_with_kernel_[p] = graph.at(p).get();
    return graph;
#else
    throw std::logic_error("--lstm-l-in-graph needs USE_FP16_REG_L_GRAPH");
#endif
  }

  // The kernels of the last graph not yet given a step (the benchmark stopped within it): each one gets a go and its
  // output is dropped, so that the graph completes (before InitX's or the teardown's synchronization).
  void DrainGraph() {
    while (steps_in_graph_ != 0) {
      ResetSignalForLaunch();
      ssgemv::cpu_pingpong_handles::poll_atomic<uint>(
          (uint *)HostSignal(), HOST_SPIN_0_UNTIL_SIGNAL_VALUE, HOST_SPIN_1_WHILE_SIGNAL_VALUE);
      Base::LstmForward();
      float dropped = 0.0f;
      Base::Output(&dropped, true, false);
      steps_in_graph_ = (steps_in_graph_ + 1) % steps_per_graph_;
    }
  }
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NVLSTMINF_MODELSINGLESTEPGEMV_CUH
