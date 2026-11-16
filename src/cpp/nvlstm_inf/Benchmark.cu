/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <iostream>
#include <numeric>
#include <utility>

#ifdef USE_NVTX
#include <nvtx3/nvtx3.hpp>
#include <unistd.h>
#endif

#include "Benchmark.cuh"
#include "DataIterator.h"
#include "ModelDef.h"

namespace nvidia::quant::lowlat_infer {

Benchmark::Benchmark(
    std::filesystem::path const &path_data,
    std::filesystem::path const &path_model,
    SamplerConfig sampler_config,
    std::uint32_t num_instances,
    std::uint32_t batch_size,
    ModelConfig const &config,
    bool stage,
    bool reap,
    util::CpuSet cpu_set,
    std::chrono::nanoseconds grace_period,
    std::optional<Clock::time_point> ts_base,
    std::optional<bool> use_omp,
    std::uint32_t clones,
    GreenContextsVector green_contexts,
    bool use_precompute_green_context)
    : ts_base_{ts_base ? *ts_base : Clock::now()},
      reap_{reap},
      cpu_set_{std::move(cpu_set)},
      grace_period_{grace_period},
      use_precompute_green_context_{use_precompute_green_context} {
  ModelDef model_def{path_model.c_str(), config.ping_pong_model_persistent};

  cex::PinnedTensor16 data16{};
  cex::PinnedTensor32 data32{};
  {  // scope for temporary data
    auto const data = util::Tensor32::Load(path_data);
    if (config.Precision() == 32) {
      data32 = data.Reallocate<cex::PinnedAllocator<float>>();
    } else {
      // Input data is converted to FP16 on the host.
      data16 = data.WithData(cex::FloatToHalf<cex::PinnedAllocator<half>>(data.Data().data(), data.Data().size()));
    }
  }

  green_contexts_ = green_contexts;

  use_threads_in_models_ = (green_contexts_.size() >= 1) && (cpu_set_.size() >= (reap ? 2 : 1) + num_instances);

  models_.resize(num_instances);
  for (std::size_t i = 0; i < num_instances; ++i) {
    models_[i] =
        InitModel(sampler_config, batch_size, config, stage, reap, use_omp, data16, data32, model_def, clones, i);
    models_[i]->SetModelId(i);
    if (i == 0) {
      use_omp = models_[0]->UseOmp();
    }
  }
}

std::unique_ptr<ModelInstance> Benchmark::InitModel(
    SamplerConfig sampler_config,
    uint32_t batch_size,
    const ModelConfig &config,
    bool stage,
    bool reap,
    std::optional<bool> use_omp,
    const cex::PinnedTensor16 &data16,
    const cex::PinnedTensor32 &data32,
    const ModelDef &model_def,
    std::uint32_t clones,
    size_t i) {
  StateCheck(clones >= 1);
  ModelInstance::ModelVector models{clones};

  for (std::uint32_t j = 0; j < clones; ++j) {
    if (green_contexts_.size() >= 1) {
      int core_id = (use_threads_in_models_) ? cpu_set_[(reap ? 2 : 1) + i] : -1;
      if (use_precompute_green_context_) {
        cuda_envs_.push_back(
            std::make_unique<CudaEnv>(
                false, true, config.use_gdrcopy, green_contexts_[i], green_contexts_.back(), core_id));
      } else {
        cuda_envs_.push_back(
            std::make_unique<CudaEnv>(false, true, config.use_gdrcopy, green_contexts_[i], nullptr, core_id));
      }
    } else {
      cuda_envs_.push_back(std::make_unique<CudaEnv>(false, true, config.use_gdrcopy));
    }
    models[j] = ModelInstance::CreateModel(model_def, *cuda_envs_.back(), batch_size, config);
  }
  std::unique_ptr<util::DataSampler> seq;
  std::size_t n_samples = (config.Precision() == 32) ? data32.Shape().at(0) : data16.Shape().at(0);
  if (sampler_config.sampler_type == SamplerType::Random) {
    seq = std::make_unique<util::RandomSampler>(sampler_config.base_seed, i, n_samples);
  } else {
    seq = std::make_unique<util::RingSampler>(sampler_config.base_seed, i, n_samples);
  }
  ModelInstance::AnyDataIterator data_iterator;
  if (config.Precision() == 32) {
    StateCheck(models.at(0)->Precision() == 32);
    data_iterator = std::make_unique<ModelInstance::PinnedDataIterator32>(data32, std::move(seq));
  } else {
    StateCheck(models.at(0)->Precision() == 16);
    data_iterator = std::make_unique<ModelInstance::PinnedDataIterator16>(data16, std::move(seq));
  }
  auto constexpr capacity = InferenceResult::kDefaultCapacity;
  return std::make_unique<ModelInstance>(std::move(models), std::move(data_iterator), capacity, stage, use_omp, !reap);
}

void Benchmark::Run(std::uint32_t const num_seconds, std::uint32_t const log_seconds) {
  auto idx_cpu = 0u;
  if (reap_) {
    StartReaper();
    if (idx_cpu < cpu_set_.size()) {
      util::SetAffinity(reaper_.native_handle(), cpu_set_[idx_cpu++]);
    }
  }

  util::CpuSet cpu_set_orig{};
  if (idx_cpu < cpu_set_.size()) {
    if (UseOmp() && !use_threads_in_models_) {
      // start threads before setting affinity
#pragma omp parallel default(none)
      Clock::now();  // dummy call
    }
    cpu_set_orig = util::GetAffinity(0);
    util::SetAffinity(0, cpu_set_[idx_cpu++]);
  }

#ifdef USE_NVTX
  nvtxNameOsThreadA(gettid(), "launcher");
#endif

  auto const ts_start = Clock::now();
  auto const duration = std::chrono::seconds{num_seconds};
  auto const ts_stop = ts_start + duration;
  auto const log_intv = std::chrono::seconds{log_seconds};
  auto ts_log = ts_start + log_intv;
  auto const log_enabled = log_seconds > 0;

  if (log_enabled) {
    std::cout << "# iterations: 0 (0%)";
    std::cout.flush();
  }

  bool running = true;
  for (auto &model : models_) {
    model->Start();
  }

  auto const models_begin = models_.begin();
  auto const models_end = models_.end();
  auto const grace_period_disabled = grace_period_ == Clock::duration::zero();
  auto ts_last_launch = Clock::time_point{Clock::duration::zero()};
  auto model_to_launch = models_begin;

  while (running) {
    running = false;
    auto const now = Clock::now();
    auto const finish = now > ts_stop;
    for (auto it = models_begin; it != models_end; ++it) {
      auto &model = *it;
      if (likely(model->Running())) {
        running = true;
        auto const launched =
            model->Loop(grace_period_disabled || (model_to_launch == it && grace_period_ <= now - ts_last_launch));
        if (launched) {
          ts_last_launch = now;
          ++model_to_launch;
          if (model_to_launch == models_end) model_to_launch = models_begin;
        }
        if (unlikely(finish)) model->Stop();
      }
    }
    if (log_enabled && ((!finish && now > ts_log) || !running)) {
      std::cout << "\r# iterations: " << Iterations();
      if (!running) {
        std::cout << " (100%)" << std::endl;
      } else {
        auto const progress = 100 * (now - ts_start) / duration;
        std::cout << " (" << progress << "%)";
        std::cout.flush();
      }
      ts_log = now + log_intv;
    }
  }

  if (!cpu_set_orig.empty()) {
    // reset affinity
    util::SetAffinity(0, cpu_set_orig);
  }

  if (reap_) {
    running_.store(false, std::memory_order_release);
    if (reaper_.joinable()) reaper_.join();
  }
}

std::size_t Benchmark::Iterations() const {
  return std::accumulate(models_.begin(), models_.end(), std::size_t{}, [](auto const &acc, auto const &res) {
    return acc + res->Result().Size();
  });
}

Benchmark::ResultVector Benchmark::Results() const {
  ResultVector results{};
  results.reserve(models_.size());
  for (auto const &model : models_) {
    results.push_back(model->ResultShared());
  }
  return results;
}

void Benchmark::Save(const std::filesystem::path &base_path) { Save(base_path, Results(), ts_base_); }

void Benchmark::Save(
    const std::filesystem::path &base_path, const Benchmark::ResultVector &results, Clock::time_point const &ts_base) {
  auto const num_instances = results.size();
  for (std::size_t i = 0; i < num_instances; ++i) {
    std::filesystem::path filepath{base_path};
    filepath += std::to_string(i + 1) + ".npy";
    results[i]->Save(filepath.c_str(), ts_base);
  }
}

void Benchmark::StartReaper() {
  running_.store(false, std::memory_order_release);
  reaper_ = std::thread{[this]() {
#ifdef USE_NVTX
    nvtxNameOsThreadA(gettid(), "reaper");
#endif
    running_.store(true, std::memory_order_release);
    while (running_.load(std::memory_order_acquire)) {
      for (auto &model : models_) {
        model->Reap();
      }
    }
  }};

  while (!running_.load(std::memory_order_acquire)) {
    // wait for reaper thread to start
  }
}

}  // namespace nvidia::quant::lowlat_infer
