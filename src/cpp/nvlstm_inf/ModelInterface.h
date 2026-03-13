/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_MODELINTERFACE_H_
#define NLSTMINF_SRC_CPP_NVLSTMINF_MODELINTERFACE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "util.h"

// Forward declarations for CUDA.
struct CUgraphExec_st;
using cudaGraphExec_t = struct CUgraphExec_st *;
struct __half;
using half = struct __half;

using cudaHostFn_t = void (*)(void *user_data);

namespace nvidia::quant::lowlat_infer {

class ModelInterface {
 public:
  using Graph = std::vector<std::unique_ptr<CUgraphExec_st, std::function<void(cudaGraphExec_t)>>>;

  ModelInterface(std::uint32_t batch_size, std::uint32_t num_steps, std::uint32_t num_features, std::uint32_t num_units)
      : batch_size_{batch_size}, num_steps_{num_steps}, num_features_{num_features}, num_units_{num_units} {}

  virtual ~ModelInterface() = default;

  // Indicates whether the model ingests sliding windows or full windows.
  [[nodiscard]] virtual bool Sliding() const = 0;

  // Indicates whether this is a PingPong model.
  [[nodiscard]] virtual bool IsPingPong() const = 0;

  // 16 or 32
  [[nodiscard]] virtual std::uint8_t Precision() const = 0;

  // Copy the initial time-steps - 1 onto the device for models ingesting sliding windows.
  virtual void InitX(util::Tensor32 const &t) = 0;

  // Resets the internal state
  void Reset() {
    InitX(util::Tensor32{util::Vector32{}, util::ShapeVector{0, 0}});
  }

  // Copy new input from host to device dev_x_. For the non-Sliding() case, sequence_major indicates that the
  // input data was transposed to sequence major layout.
  virtual void Input(float const *host_in, bool sequence_major) = 0;

  // Copy new input from host to device dev_x_. For the non-Sliding() case, sequence_major indicates that the
  // input data was transposed to sequence major layout. This call is only support for FP16 models.
  virtual void Input(half const *host_in, bool sequence_major) = 0;

  // Copy new input from host to device dev_x_ via pointer indirection.
  virtual void Input(float const **host_addr) = 0;

  // Copy new input from host to device dev_x_ via pointer indirection.
  // This call is only support for FP16 models.
  virtual void Input(half const **host_addr) = 0;

  // Check if the last async copy operation of the input has finished.
  virtual bool InputReady() = 0;

  // Wait until the last async copy operation of the input has finished.
  virtual void InputWait() = 0;

  // Run the Lstm inference on input dev_x_ and store results in dev_y_.
  virtual void LstmForward() = 0;

  // Run the final aggregation on dev_y_ and store results in dev_out_.
  virtual void Aggregate() = 0;

  // Run the full inference.
  void Infer() {
    LstmForward();
    Aggregate();
  }

  // Input data copying and inference in a single call.
  void Infer(float const *input, bool sequence_major = false) {
    Input(input, sequence_major);
    Infer();
  }

  // Input data (FP16) copying and inference in a single call.
  void Infer(half const *input, bool sequence_major = false) {
    Input(input, sequence_major);
    Infer();
  }

  // Input data copying and inference in a single call.
  void Infer(float const **input) {
    Input(input);
    Infer();
  }

  // Input data (FP16) copying and inference in a single call.
  void Infer(half const **input) {
    Input(input);
    Infer();
  }

  // Copy the inference results to the host. The `wait` arguments determines whether this method blocks until the
  // async copy operation has finished. The `shift` argument controls input shifting.
  virtual void Output(float *host_out, bool wait, bool shift) = 0;

  // Copy the inference results to the host and flip.
  void Output(float *host_out, bool wait) { return Output(host_out, wait, true); }

  // Copy the inference results to the host in a blocking call and flip.
  void Output(float *host_out) { return Output(host_out, true); }

  // Check if the last async copy operation of the output has finished.
  virtual bool OutputReady() = 0;

  // Wait until the last async copy operation of the output has finished.
  virtual void OutputWait() = 0;

  // Capture the full inference CUDA graph.
  virtual Graph Capture(float const *input, float *host_out, bool sequence_major) = 0;

  // Capture the full inference CUDA graph with FP16 input.
  virtual Graph Capture(half const *input, float *host_out, bool sequence_major) = 0;

  // Capture the full inference CUDA graph with indirect copy.
  virtual Graph Capture(float const **input, float *host_out) = 0;

  // Capture the full inference CUDA graph with indirect copy with FP16 input.
  virtual Graph Capture(half const **input, float *host_out) = 0;

  // Launch the full inference CUDA graph without a callback.
  void Launch(Graph &graph) {
    Launch(graph, nullptr, nullptr);
  }

  virtual void Finalize() { };

  // Launch the full inference CUDA graph with a callback.
  virtual void Launch(Graph &graph, cudaHostFn_t call_back, void *user_data) = 0;

  // Indicates that the last graph has been launched.
  virtual bool Launched() = 0;

  // Store the input without running the inference (for sliding windows)
  virtual void StoreInput() = 0;

  // Called before ``Input()``
  virtual void PreProcess() {}

  // Check whether the pre-processing has finished.
  virtual bool PreProcessReady() { return true; }

  // Wait until the pre-processing has finished.
  virtual void PreProcessWait() {
    while (!PreProcessReady()) {
    }
  }

  // Called after ``Input()``
  virtual void PostProcess() {}

  // Check whether the post-processing has finished.
  virtual bool PostProcessReady() { return true; }

  // Wait until the post-processing has finished.
  virtual void PostProcessWait() {
    while (!PostProcessReady()) {
    }
  }

  [[nodiscard]] auto BatchSize() const { return batch_size_; }
  [[nodiscard]] auto NumSteps() const { return num_steps_; }
  [[nodiscard]] auto NumFeatures() const { return num_features_; }
  [[nodiscard]] auto NumUnits() const { return num_units_; }
  [[nodiscard]] auto InputSize() const { return NumFeatures() * (Sliding() ? 1 : NumSteps()); }

 private:
  // input sizes
  std::uint32_t const batch_size_;
  std::uint32_t const num_steps_;
  std::uint32_t const num_features_;
  std::size_t const num_units_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_MODELINTERFACE_H_
