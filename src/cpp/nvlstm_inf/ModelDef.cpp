/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "ModelDef.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "util.h"

using namespace nvidia::quant::lowlat_infer;

namespace {

util::Tensor32 LoadLstm(
    std::filesystem::path const &model_path, std::string const &prefix, std::size_t level, std::string const &suffix) {
  auto filename = prefix + std::to_string(level) + suffix;
  return util::Tensor32::Load(model_path / filename);
}

}  // namespace

ModelDef::ModelDef(const char *model_dir, bool skip_weights) {
  std::filesystem::path model_path{model_dir};

  // Always load dimensions (needed by ping pong model)
  auto input_size = util::LoadVector<std::int32_t>(model_path / "input_size.npy");
  num_steps_ = input_size.at(0);
  num_features_ = input_size.at(1);

  if (!skip_weights) {
    // Only load weights for non-ping pong models
    std::size_t num_layers = 0;
    char constexpr prefix[] = "lstm_";

    for (auto &entry : std::filesystem::directory_iterator{model_path}) {
      auto filename = entry.path().filename().string();
      if (filename.rfind(prefix, 0) == 0) {
        auto const level_char = filename.at(std::size(prefix) - 1);
        num_layers = std::max(num_layers, static_cast<std::size_t>(level_char - '0') + 1);
      }
    }

    for (std::size_t i = 0; i < num_layers; ++i) {
      lstm_W_.emplace_back(LoadLstm(model_path, prefix, i, "_W.npy"));
      lstm_R_.emplace_back(LoadLstm(model_path, prefix, i, "_R.npy"));
      lstm_B_.emplace_back(LoadLstm(model_path, prefix, i, "_B.npy"));
    }

    mat_mul_ = util::Tensor32::Load(model_path / "mul.npy");

    auto const add_path = model_path / "add.npy";
    if (std::filesystem::exists(add_path)) {
      bias_ = util::LoadVector<float>(add_path).at(0);
    } else {
      bias_ = NAN;
    }
  } else {
    // Ping pong model: skip all weight files
    // PingPong model performs simple sum reduction without weights or bias
    bias_ = NAN;
  }

  Check();
}

ModelDef::ModelDef(
    std::uint32_t num_steps,
    std::uint32_t num_features,
    ModelDef::LstmTensors &&lstm_W,
    ModelDef::LstmTensors &&lstm_R,
    ModelDef::LstmTensors &&lstm_B,
    util::Tensor32 &&mat_mul,
    float bias)
    : num_steps_{num_steps},
      num_features_{num_features},
      lstm_W_{std::move(lstm_W)},
      lstm_R_{std::move(lstm_R)},
      lstm_B_{std::move(lstm_B)},
      mat_mul_{std::move(mat_mul)},
      bias_{bias} {
  Check();
}

bool ModelDef::HasBias() const { return !std::isnan(bias_); }

std::uint32_t ModelDef::NumUnits() const {
  if (NumLayers() == 0) return NumFeatures() * NumSteps();

  return static_cast<std::uint32_t>(lstm_R_.at(0).Shape().at(2));
}

std::size_t ModelDef::NumLstmWeights() const {
  std::size_t total{0};
  for (std::size_t i = 0; i < NumLayers(); ++i) {
    total += LstmW().at(i).Data().size() + LstmR().at(i).Data().size() + LstmB().at(i).Data().size();
  }
  return total;
}

std::size_t ModelDef::GateIndex(char gate) {
  switch (gate) {
    case 'i':
      return 0;
    case 'o':
      return 1;
    case 'f':
      return 2;
    case 'c':
      return 3;
    default:
      std::string msg{"invalid character: "};
      msg.append(1, gate);
      throw std::invalid_argument(msg);
  }
}
std::size_t ModelDef::OffsetW(char gate, bool input_layer) const {
  auto const size = (input_layer ? NumFeatures() : NumUnits()) * NumUnits();
  return GateIndex(gate) * size;
}
std::size_t ModelDef::OffsetR(char gate) const {
  auto const size = NumUnits() * NumUnits();
  return GateIndex(gate) * size;
}
std::size_t ModelDef::OffsetB(char gate, bool hidden) const {
  return NumUnits() * (GateIndex(gate) + (hidden ? 4 : 0));
}

void ModelDef::Check() {
  auto const num_layers = NumLayers();
  StateCheck(lstm_W_.size() == num_layers);
  StateCheck(lstm_R_.size() == num_layers);
  StateCheck(lstm_B_.size() == num_layers);

  // Skip validation for ping pong models (when weights weren't loaded)
  if (IsPingPong()) {
    return;
  }

  auto const num_units = NumUnits();
  auto const num_gates = static_cast<std::size_t>(num_units) * 4;
  auto const num_biases = static_cast<std::size_t>(num_units) * 8;
  auto layer_input_size = NumFeatures();

  for (std::uint32_t i = 0; i < num_layers; ++i) {
    auto &shape_W = lstm_W_.at(i).Shape();
    StateCheck(shape_W.at(1) == num_gates);
    StateCheck(shape_W.at(2) == layer_input_size);
    auto &shape_R = lstm_R_.at(i).Shape();
    StateCheck(shape_R.at(1) == num_gates);
    StateCheck(shape_R.at(2) == num_units);
    auto &shape_B = lstm_B_.at(i).Shape();
    StateCheck(shape_B.at(0) == 1);
    StateCheck(shape_B.at(1) == num_biases);

    // The input size for the next layer is the number of units in the current layer.
    layer_input_size = num_units;
  }

  StateCheck(mat_mul_.Shape().at(0) == (num_layers > 0 ? num_units : NumSteps() * NumFeatures()));
}

std::size_t ModelDef::Flops(std::uint32_t cost_sigmoid, std::uint32_t cost_tanh) const {
  // Ping pong models have no meaningful FLOPS calculation
  if (IsPingPong()) {
    return 0;
  }

  std::size_t result{0};
  for (std::size_t l = 0; l < NumLayers(); ++l) {
    // matrix - vector multiplications
    auto &shape_W = LstmW().at(l).Shape();
    assert(shape_W.at(0) == 1);
    auto const num_input = shape_W.at(2);
    auto const num_linear = shape_W.at(1);
    result += 2 * num_input * num_linear;
    auto &shape_R = LstmR().at(l).Shape();
    assert(shape_R.at(0) == 1);
    auto const num_hidden = shape_R.at(2);
    assert(4 * num_hidden == num_linear);
    result += 2 * num_hidden * shape_R.at(1);
    // vector - vector addition
    auto &shape_B = LstmB().at(l).Shape();
    assert(shape_B.at(0) == 1);
    result += shape_B.at(1);
    // sigmoid and tanh: assume 2 flops each
    result += (3 * cost_sigmoid + cost_tanh) * num_hidden;
    // update of the cell state: 2 * Hadamard product + vector addition
    result += 3 * num_hidden;
    // update of the hidden state: tanh + Hadamard product
    result += (cost_tanh + 1) * num_hidden;
  }

  result *= NumSteps();

  // Final scalar product and bias
  auto &shape_mul = MatMul().Shape();
  result += 2 * shape_mul.at(0);
  assert(shape_mul.at(1) == 1);

  return result;
}

double ModelDef::SerialSolNanos(
    double const flops_per_nano_gemm,
    double const flops_per_nano_other,
    std::uint32_t cost_sigmoid,
    std::uint32_t cost_tanh) const {
  double result{0};

  for (std::uint32_t l = 0; l < NumLayers(); ++l) {
    // vector - vector addition for bias
    auto &shape_B = LstmB().at(l).Shape();
    assert(shape_B.at(0) == 1);
    auto const num_bias = shape_B.at(1) / 2;

    // matrix - vector multiplications
    auto &shape_W = LstmW().at(l).Shape();
    assert(shape_W.at(0) == 1);
    auto const num_input = shape_W.at(2);
    auto const num_linear = shape_W.at(1);
    result += static_cast<double>(2 * num_input * num_linear) / flops_per_nano_gemm;
    assert(num_linear == num_bias);

    auto &shape_R = LstmR().at(l).Shape();
    assert(shape_R.at(0) == 1);
    auto const num_hidden = shape_R.at(2);
    assert(4 * num_hidden == num_linear);
    result += static_cast<double>(2 * num_hidden * shape_R.at(1) + num_bias) / flops_per_nano_other;
    // sigmoid and tanh: assume 2 flops each
    result += static_cast<double>((3 * cost_sigmoid + cost_tanh) * num_hidden) / flops_per_nano_other;
    // update of the cell state: 2 * Hadamard product + vector addition
    result += static_cast<double>(3 * num_hidden) / flops_per_nano_other;
    // update of the hidden state: tanh + Hadamard product
    result += static_cast<double>((cost_tanh + 1) * num_hidden) / flops_per_nano_other;
  }

  result *= NumSteps();

  // Final scalar product and bias
  auto &shape_mul = MatMul().Shape();
  result += static_cast<double>(2 * shape_mul.at(0)) / flops_per_nano_other;
  assert(shape_mul.at(1) == 1);

  return result;
}
