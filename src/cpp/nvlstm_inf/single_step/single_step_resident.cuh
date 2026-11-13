/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Resident precompute (README.md): one kernel per model instance, launched once, that advances every open window of the
// sliding-window LSTM by one step per input and publishes the states the last step of the oldest window needs, for
// whatever does that step (an inference kernel, or the host).
//
// Per pass the host writes the input into the input slot and the go words; the kernel copies the input to the device,
// runs the pass, and for a pass that publishes writes, per layer, the recurrent term R * h (4 gates, gate-major) and
// the cell state of the window that has seen num_steps - 1 inputs into one of two state slots, then the ready word.
// The ready word only grows: Init publishes the states of step 0 at a new odd value R0 (Init returns before the
// consumer is told of it, which then reads R0 from the word), the states of step k at R0 + k. The consumer of step k
// waits until the ready word is at least R0 + k and reads slot k % 2.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>

namespace nvidia::quant::lowlat_infer {

class ModelDef;

namespace single_step {

struct SingleStepResidentConfig {
  // CTAs over the window steps for every tile of units (1: one CTA per 8 units).
  int step_splits = 1;
  // The go words and the input slot in device memory, written by the host through a GDRCopy mapping, instead of in
  // pinned host memory (the kernel then polls L2, not PCIe). Needs a build with GDRCopy.
  bool use_gdrcopy = false;
  // Weight slabs resident in shared memory as far as they fit (layer 0's input weights first, then R of every layer,
  // then the input weights of layers 1..), instead of streamed through the TMA pipeline with the states (or, layer 0's,
  // read by the critical warp from global memory).
  bool resident_weights = true;
};

class SingleStepResident {
 public:
  // Allocates and prepares everything; the kernel runs on `stream`, which is its alone while it runs (a non-blocking
  // stream in the precompute's partition of the device).
  SingleStepResident(ModelDef const& model_def, SingleStepResidentConfig const& config, cudaStream_t stream);
  ~SingleStepResident();  // stops the kernel

  // Launches the kernel, which then stays and waits for its passes, until destruction. Nothing may synchronize the
  // whole device while it runs (cudaDeviceSynchronize, cudaFree): start it where the persistent inference kernel
  // starts.
  void Start();

  // Stops the kernel (after its last pass). Frees nothing: with several instances in one partition, freeing memory
  // synchronizes the partition, which must wait until every instance's kernel has stopped (destruction).
  void Stop();

  SingleStepResident(SingleStepResident const&) = delete;
  SingleStepResident& operator=(SingleStepResident const&) = delete;

  // Clears every window and the slots (the ready word to 0), then runs one pass per row of `inputs` (up to num_steps -
  // 1 rows of num_features, the first inputs of the stream), the last one publishing the states for step 0; with no
  // rows step 0 gets the zero state (as ModelSingleStep after ClearHiddenState).  Waits.
  // Before Start() it launches the kernel for these passes and stops it again.
  void Init(float const* inputs, std::uint32_t rows);

  // The pass with the input of step k (num_features values): publishes the states for step k + 1 (ready R0 + k + 1).
  // Does not wait.
  void Step(std::uint32_t k, half const* input);
  void Step(std::uint32_t k, float const* input);

  // Waits until the last pass is done.
  void Wait() const;
  // Waits until the last pass has published its slot (the ready word written): the consumer's states are in; the pass
  // may still be advancing the other windows.
  void WaitPublished() const;

  // For the consumer: the state slots, and the ready word (device memory).
  [[nodiscard]] half* GemmH(int slot, int layer) const;      // 4 * num_units values, gate-major (i, f, c, o)
  [[nodiscard]] half* CellState(int slot, int layer) const;  // num_units values
  [[nodiscard]] unsigned* ReadyWord() const;
  [[nodiscard]] int Ctas() const;
  /// The kernel's CTAs for a model of num_units units and num_steps window steps (Ctas() before construction): one SM
  /// each, all resident at once (cooperative launch), so all instances' kernels together must fit the SMs of the
  /// precompute's partition.
  [[nodiscard]] static int CtasFor(int num_units, int num_steps, int step_splits);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace single_step
}  // namespace nvidia::quant::lowlat_infer
