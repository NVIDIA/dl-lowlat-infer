/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NVLSTMINF_MODELSINGLESTEPBASE_CUH_
#define NVLSTMINF_MODELSINGLESTEPBASE_CUH_

#include <array>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "CudaEnv.cuh"
#include "ModelDef.h"
#include "ModelInterface.h"
#include "cuda_helpers.cuh"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

// Supported data types are: float, half
template <typename TData = float>
class ModelSingleStepBase : public ModelInterface {
 public:
  using data_type = TData;
  using ModelInterface::Graph;

  struct Config final {
    bool streaming;
    // everything on the main stream, so a captured graph has no fork/join nodes
    bool single_stream = false;
  };

  ModelSingleStepBase(CudaEnv &cuda_env, ModelDef const &model_def, std::uint32_t batch_size, const Config &cfg);

  ~ModelSingleStepBase() override = default;

  // Indicates whether the model ingests sliding windows or full windows.
  [[nodiscard]] bool Sliding() const override { return true; };

  // Indicates whether this is a PingPong model.
  [[nodiscard]] bool IsPingPong() const override { return false; };

  // 16 or 32
  [[nodiscard]] std::uint8_t Precision() const override { return static_cast<std::uint8_t>(sizeof(data_type) * 8); }

  // Copy the initial steps - 1 onto the device.
  void InitX(util::Tensor32 const &t) override;

  // Copy new input from host to device dev_x_.
  void Input(float const *host_in, bool sequence_major) override;

  // Copy new input from host to device dev_x_. This call is only support for FP16 models.
  void Input(half const *host_in, bool sequence_major) override;

  // Copy new input from host to device dev_x_ via pointer indirection.
  void Input(float const **host_addr) override;

  // Copy new input from host to device dev_x_ via pointer indirection.
  // This call is only support for FP16 models.
  void Input(half const **host_addr) override;

  // Check if the last async copy operation of the input has finished.
  bool InputReady() override;

  // Wait until the last async copy operation of the input has finished.
  void InputWait() override;

  // Run the final aggregation on dev_y_ and store results in dev_out_.
  void Aggregate() override;

  // Copy the inference results to the host. The `wait` arguments determines whether this method blocks until the
  // async copy operation has finished.
  void Output(float *host_out, bool wait, bool shift) override;

  using ModelInterface::Output;

  // Check if the last async copy operation of the output has finished.
  bool OutputReady() override;

  // Wait until the last async copy operation of the output has finished.
  void OutputWait() override;

  // Capture the full inference CUDA graph.
  Graph Capture(float const *input, float *host_out, bool sequence_major) override;

  // Capture the full inference CUDA graph with FP16 input.
  Graph Capture(half const *input, float *host_out, bool sequence_major) override;

  // Capture the full inference CUDA graph with indirect copy.
  Graph Capture(float const **input, float *host_out) override;

  // Capture the full inference CUDA graph with indirect copy with FP16 input.
  Graph Capture(half const **input, float *host_out) override;

  // Launch the full inference CUDA graph.
  void Launch(Graph &graph, cudaHostFn_t call_back, void *user_data) override;

  using ModelInterface::Launch;

  // Indicates that the last graph has been launched.
  bool Launched() override;

  // Store the input without running the inference (for sliding windows)
  void StoreInput() override;

  void SetPinnedOut(float *host_out) {
    if (BatchSize() == 1) {
      pinned_out_ = host_out;
    }
  }

  void ResetPinnedOut() { pinned_out_ = nullptr; }

#if USE_FP16_REG_L_GRAPH
  // For a kernel captured in front of each step (ModelSingleStepGemv's --lstm-l-in-graph), for the next Capture:
  // Work captured at the start of each step (on the capture stream, given the step's parity), before its inference:
  // the graph's single root. nullptr: none.
  void SetCapturePrologue(std::function<void(cudaStream_t, bool)> prologue) { capture_prologue_ = std::move(prologue); }
  // Capture n (1 or even) consecutive steps, each the prologue and the inference, in each graph (one per starting
  // parity).
  void SetStepsPerGraph(unsigned n) { steps_per_graph_ = n; }
  // Layer 0's input product W_x * x of the step (per unit its 4 gates), written by the prologue's work: layer 0 reads
  // it instead of computing it, and the step's input is not copied (layer 0 is its only reader). nullptr: computed.
  void SetLayer0InputProduct(data_type const *dev) {
    StateCheck(dev == nullptr || skip_replicate_);
    layer0_x_dot_ = dev;
  }
  // Launch graphs[parity] of a pair Capture returned with steps_per_graph steps each, as Launch.
  void LaunchExec(cudaGraphExec_t const *graphs, unsigned steps_per_graph);
#endif

 protected:
  data_type *GetDevXLast() { return dev_x_last_; }
  /// SetLayer0InputProduct (nullptr: computed).
  data_type const *Layer0InputProduct() const {
#if USE_FP16_REG_L_GRAPH
    return layer0_x_dot_;
#else
    return nullptr;
#endif
  }
  bool SkipReplicate() { return skip_replicate_; }
  bool SingleStream() const { return single_stream_; }
  CudaEnv &Env() { return cuda_env_; }

  cex::DevicePtr<data_type> &DevXExpanded() { return dev_x_expanded_; }
  cex::DevicePtr<data_type> &DevY() { return dev_y_; }
  data_type const *DevMul() const { return dev_mul_.get(); }
  float *OutputTarget() { return pinned_out_ ? pinned_out_ : dev_out_.get(); }
  float Bias() const { return bias_; }

  // Indicates whether sliding window optimizations are enabled.
  [[nodiscard]] bool SingleStep() const { return true; }

  // Indicates whether conversion to TF32 is allowed.
  [[nodiscard]] bool AllowTf32() const { return false; }

  // Flip the input buffer dev_x_. Call this between inferences.
  virtual void ShiftX();

  static std::conditional_t<std::is_same_v<TData, float>, std::vector<float> const &, std::vector<TData>> Convert(
      std::vector<float> const &input);

  // The position of the last batch in dev_x_ and dev_y_
  data_type *dev_x_last_ = nullptr;
  data_type *dev_y_last_ = nullptr;

 private:
  void ExpandInput();
  float const *StageHostPointer(float const *host_in, cudaStream_t stream);

  template <typename TInput>
  void InputGeneric(TInput const **host_addr);

  // The high-priority stream for input and output copies, or the main stream with single_stream
  cudaStream_t ForkHighPriority(cudaEvent_t after);
  void JoinHighPriority(cudaEvent_t after);

  // Capture the full inference CUDA graph with indirect copy.
  Graph Capture(std::function<void()> const &ops);

  void ClearOutput(cudaStream_t stream);

  CudaEnv &cuda_env_;

  // Input buffers
  cex::DevicePtr<data_type> dev_x_;
  cex::DevicePtr<data_type> dev_x_expanded_;
  cex::DevicePtr<float> dev_x_staging_;

  // device memory pointers for LSTM output
  cex::DevicePtr<data_type> dev_y_;

  // vector for final scalar aggregation
  cex::DevicePtr<data_type> dev_mul_;
  // bias for final addition
  float const bias_;
  // vector for final output
  cex::DevicePtr<float> dev_out_;
  float *pinned_out_ = nullptr;

  // event for synchronizing input
  cex::EventPtr event_in_start_, event_in_stop_;
  // event for synchronizing output
  cex::EventPtr event_out_start_, event_out_stop_;
  // event for starting aggregation
  cex::EventPtr event_agg_start_;

  // indicates that the next step is odd
  bool odd_step_{false};

  bool skip_replicate_{true};

  bool const single_stream_;

#if USE_FP16_REG_L_GRAPH
  std::function<void(cudaStream_t, bool)> capture_prologue_;
  unsigned steps_per_graph_ = 1;
  data_type const *layer0_x_dot_ = nullptr;
#endif
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NVLSTMINF_MODELSINGLESTEPBASE_CUH_
