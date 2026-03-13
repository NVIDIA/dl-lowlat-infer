/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_MODELDEF_H_
#define NLSTMINF_SRC_CPP_NVLSTMINF_MODELDEF_H_

#include <cstdint>
#include <vector>

#include "util.h"

namespace nvidia::quant::lowlat_infer {

class ModelDef {
 public:
  using LstmTensors = std::vector<util::Tensor32>;

  explicit ModelDef(char const *model_dir, bool skip_weights = false);

  ModelDef(std::uint32_t num_steps,
           std::uint32_t num_features,
           LstmTensors &&lstm_W,
           LstmTensors &&lstm_R,
           LstmTensors &&lstm_B,
           util::Tensor32 &&mat_mul,
           float bias);

  [[nodiscard]] auto NumSteps() const { return num_steps_; }
  void NumSteps(std::uint32_t num_steps) { num_steps_ = num_steps; }
  [[nodiscard]] auto NumFeatures() const { return num_features_; }
  [[nodiscard]] auto NumLayers() const { return static_cast<std::uint32_t>(lstm_W_.size()); }
  [[nodiscard]] std::uint32_t NumUnits() const;
  [[nodiscard]] std::size_t NumLstmWeights() const;

  // Offsets according to https://github.com/onnx/onnx/blob/main/docs/Operators.md#lstm in order [iofc]
  static std::size_t GateIndex(char gate);
  [[nodiscard]] std::size_t OffsetW(char gate, bool input_layer) const;
  [[nodiscard]] std::size_t OffsetR(char gate) const;
  [[nodiscard]] std::size_t OffsetB(char gate, bool hidden) const;

  // one tensor per layer with shape 1 x (num_units * 4) x (num_features if layer == 0 else num_units)
  [[nodiscard]] auto &LstmW() const { return lstm_W_; }
  // one tensor per layer with shape 1 x (num_units * 4) x num_units
  [[nodiscard]] auto &LstmR() const { return lstm_R_; }
  // one tensor per layer with shape 1 x (num_units * 8)
  [[nodiscard]] auto &LstmB() const { return lstm_B_; }

  [[nodiscard]] auto &LstmW() { return lstm_W_; }
  [[nodiscard]] auto &LstmR() { return lstm_R_; }
  [[nodiscard]] auto &LstmB() { return lstm_B_; }

  // shape num_units x 1
  [[nodiscard]] auto &MatMul() const { return mat_mul_; }
  [[nodiscard]] auto Bias() const { return bias_; }
  [[nodiscard]] bool HasBias() const;
  [[nodiscard]] bool IsPingPong() const { return NumLayers() == 0 && !HasBias(); }

  static auto constexpr kCostActivation = 2;

  [[nodiscard]] std::size_t Flops(std::uint32_t cost_sigmoid = kCostActivation,
                                  std::uint32_t cost_tanh = kCostActivation) const;

  [[nodiscard]] double SerialSolNanos(double flops_per_nano_gemm,
                                      double flops_per_nano_other,
                                      std::uint32_t cost_sigmoid = 2,
                                      std::uint32_t cost_tanh = 2) const;

  [[nodiscard]] double FlopsPerNanoSol(double flops_per_nano_gemm,
                                       double flops_per_nano_other,
                                       std::uint32_t cost_sigmoid = kCostActivation,
                                       std::uint32_t cost_tanh = kCostActivation) const {
    return static_cast<double>(Flops(cost_sigmoid, cost_sigmoid)) /
           SerialSolNanos(flops_per_nano_gemm, flops_per_nano_other, cost_sigmoid, cost_tanh);
  }

 private:
  void Check();

  std::uint32_t num_steps_;
  std::uint32_t num_features_;

  LstmTensors lstm_W_;
  LstmTensors lstm_R_;
  LstmTensors lstm_B_;

  util::Tensor32 mat_mul_;
  float bias_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_MODELDEF_H_
