/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NVLSTMINF_CPP_BENCHMARK_CUH_
#define NVLSTMINF_CPP_BENCHMARK_CUH_

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "CudaEnv.cuh"
#include "InferenceResult.h"
#include "ModelInstance.cuh"
#include "util.h"

namespace nvidia::quant::lowlat_infer {

class Benchmark {
 public:
  using Clock = InferenceResult::Clock;

  enum class SamplerType {
    Random,
    Ring,
  };

  struct SamplerConfig final {
    std::uint64_t base_seed;
    SamplerType sampler_type = SamplerType::Random;
  };

  Benchmark(
      std::filesystem::path const &path_data,
      std::filesystem::path const &path_model,
      SamplerConfig sampler_config,
      std::uint32_t num_instances,
      std::uint32_t batch_size = 1,
      ModelConfig const &config = {},
      bool stage = false,
      bool reap = false,
      util::CpuSet cpu_set = {},
      std::chrono::nanoseconds grace_period = {},
      std::optional<Clock::time_point> ts_base = std::nullopt,
      std::optional<bool> use_omp = std::nullopt,
      std::uint32_t clones = 1,
      GreenContextsVector green_contexts = GreenContextsVector(),
      bool use_precompute_green_context = false);

  void Run(std::uint32_t num_seconds, std::uint32_t log_seconds = 0);

  void Save(std::filesystem::path const &base_path);

  void Reset() {
    for (auto &model : models_) {
      model->Reset();
    }
  }

  void Finalize() {
    for (auto &model : models_) {
      model->Finalize();
    }
  }

  void Reserve(std::size_t size) {
    for (auto &model : models_) {
      model->Result().Reserve(size);
    }
  }

  bool UseOmp() { return models_[0]->UseOmp(); }

  [[nodiscard]] std::size_t NumInstances() const { return models_.size(); }

  [[nodiscard]] std::size_t Iterations() const;

  using ResultVector = std::vector<std::shared_ptr<InferenceResult const>>;

  [[nodiscard]] InferenceResult const &Result(std::size_t index_model) const {
    return models_.at(index_model)->Result();
  }

  [[nodiscard]] ResultVector Results() const;

  [[nodiscard]] auto TsBase() const { return ts_base_; }

  static void Save(
      std::filesystem::path const &base_path, ResultVector const &results, Clock::time_point const &ts_base);

 private:
  void StartReaper();

  std::unique_ptr<ModelInstance> InitModel(
      SamplerConfig sampler_config,
      std::uint32_t batch_size,
      ModelConfig const &config,
      bool stage,
      bool reap,
      std::optional<bool> use_omp,
      cex::PinnedTensor16 const &data16,
      cex::PinnedTensor32 const &data32,
      const ModelDef &model_def,
      std::uint32_t clones,
      size_t i);

  GreenContextsVector green_contexts_;
  std::vector<std::unique_ptr<CudaEnv>> cuda_envs_{};
  std::vector<std::unique_ptr<ModelInstance>> models_;
  Clock::time_point const ts_base_;

  bool const reap_;
  std::thread reaper_{};
  util::CpuSet const cpu_set_;
  std::atomic_bool running_{false};

  std::chrono::nanoseconds const grace_period_;

  bool use_green_contexts_{false};
  // use different context from inference for the precompute
  bool use_precompute_green_context_{false};
  bool use_threads_in_models_{false};
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NVLSTMINF_CPP_BENCHMARK_CUH_
