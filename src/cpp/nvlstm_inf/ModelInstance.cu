/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <algorithm>
#include <cstring>
#include <omp.h>
#include <stdexcept>

#ifdef USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include "ModelAsSingleStepGemv.cuh"
#include "ModelDef.h"
#include "ModelInstance.cuh"
#include "ModelSingleStep.cuh"
#include "ModelSingleStepGemv.cuh"
#include "PingPongModelPersistent.cuh"
#include "ss_gemv_impl/kernel_traits.h"
#include "util.h"

#ifdef USE_NVTX
namespace {
nvtx3::registered_string const kGraphLaunchRange{"graph_launch"};
}
#endif

namespace nvidia::quant::lowlat_infer {

ModelInstance::ModelInstance(
    ModelVector&& models,
    AnyDataIterator&& data,
    std::size_t init_capacity,
    bool stage,
    std::optional<bool> use_omp,
    bool reap)
    : models_{std::move(models)},
      num_models_(models_.size()),
      model_{models_.at(0)},
      data_{std::move(data)},
      void_iterator_{CreateVoidIterator(data_)},
      result_{std::make_shared<InferenceResult>(init_capacity)},
      stage_{stage},
      omp_num_threads_{omp_get_max_threads()},  // controlled by environment variable OMP_NUM_THREADS
      is_half_{model_->Precision() == 16},
      holds_half_{std::holds_alternative<PinnedDataIteratorPtr16>(data_)},
      transpose_{stage_ && !model_->Sliding() && !model_->IsPingPong()},
      input_{num_models_},
      output_{num_models_},
      staging_32_{num_models_},
      staging_16_{num_models_},
      graph_{num_models_},
      ts_start_{num_models_},
      ts_stop_{num_models_},
#ifdef USE_NVTX
      nvtx_ranges_{num_models_},
#endif
      reap_{reap},
      states_(num_models_, State::kInitialized) {
  StateCheck(!holds_half_ || is_half_);
  StateCheck(holds_half_ || std::holds_alternative<PinnedDataIteratorPtr32>(data_));
  StateCheck(!holds_half_ || std::get<PinnedDataIteratorPtr16>(data_));
  StateCheck(holds_half_ || std::get<PinnedDataIteratorPtr32>(data_));
  if (num_models_ > 1) {
    for (auto& model : models_) {
      StateCheck(model->Sliding() == model_->Sliding());
      StateCheck(model->IsPingPong() == model_->IsPingPong());
      StateCheck(model->Precision() == model_->Precision());
      StateCheck(model->BatchSize() == model_->BatchSize());
      StateCheck(model->NumSteps() == model_->NumSteps());
      StateCheck(model->NumFeatures() == model_->NumFeatures());
      StateCheck(model->NumUnits() == model_->NumUnits());
    }
  }
  if (!model_->Sliding()) {
    auto const num_excess_samples = std::visit([](auto const& d) { return d->NumExcessSamples(); }, data_);
    StateCheck(num_excess_samples >= model_->NumSteps() - 1);
  }
  auto const batch_size = model_->BatchSize();
  for (std::size_t i = 0; i < num_models_; ++i) {
    input_[i] = cex::MallocHost<void const*>(batch_size);
    output_[i] = cex::MallocHost<float>(batch_size);

    auto& graph = graph_[i];
    auto& model = models_[i];
    auto* output = output_[i].get();
    model->Reset();
    if (stage_) {
      if (is_half_) {
        staging_16_[i] = cex::MallocHost<half>(batch_size * model_->InputSize());
        graph = model->Capture(staging_16_[i].get(), output, transpose_);
      } else {
        staging_32_[i] = cex::MallocHost<float>(batch_size * model_->InputSize());
        graph = model->Capture(staging_32_[i].get(), output, transpose_);
      }
    } else if (holds_half_) {
      graph = model->Capture(reinterpret_cast<half const**>(input_[i].get()), output);
    } else {
      graph = model->Capture(reinterpret_cast<float const**>(input_[i].get()), output);
    }
    ts_start_[i].resize(batch_size);
  }
  use_omp_ = use_omp.has_value() ? *use_omp : TryOpenMp();
  StateCheck(!use_omp_ || stage);
}

void ModelInstance::Initialize() {
  for (auto& model : models_) {
    model->Reset();
  }

  for (auto& state : states_) {
    state = State::kInitialized;
  }
}

void ModelInstance::Finalize() {
  for (auto& model : models_) {
    model->Finalize();
  }
}

bool ModelInstance::Inference() {
  auto const idx_model = next_to_launch_;

  auto& model_to_launch = models_[idx_model];
  auto& model_state = states_[idx_model];

  StateCheck(model_state != State::kInference);

  if (unlikely(model_state == State::kStopped)) {
    return true;
  }

  // Process entry state
  if ((model_state == State::kPostProcess && model_to_launch->PostProcessReady()) && model_state != State::kStopped ||
      unlikely(model_state == State::kInitialized)) {
    model_to_launch->PreProcess();
    model_state = State::kPreProcess;
  } else if (model_state == State::kPostProcess) {
    // still post-processing
    return false;
  }

  if (model_state == State::kPreProcess) {
    if (model_to_launch->PreProcessReady()) {
      model_state = State::kInference;
    } else {
      // still pre-processing
      return false;
    }
  }

  StateCheck(model_state == State::kInference);

#ifdef USE_NVTX
  nvtx_ranges_[idx_model] = std::make_shared<nvtx3::unique_range>(kGraphLaunchRange);
#endif
  auto in = input_[idx_model].get();
  auto& ts_start = ts_start_[idx_model];
  auto const batch_size = model_->BatchSize();
  for (uint32_t i = 0; i < batch_size; ++i) {
    in[i] = void_iterator_();
    ts_start[i] = Clock::now();
  }
  if (stage_) Stage();
  model_to_launch->Launch(graph_[idx_model]);
  --ready_to_launch_;
  next_to_launch_ = NextModelIndex(idx_model);
  spsc_.NotifyConsumed();
  model_to_launch->PostLaunch();
  result_->Reserve(batch_size);

  // Distribute inputs to all sliding window models
  if (num_models_ > 1 && model_->Sliding()) {
    for (std::size_t i = 0; i < num_models_; ++i) {
      if (i != idx_model) {
        auto& model = models_[i];
        if (holds_half_) {
          model->Input(reinterpret_cast<half const**>(in));
        } else {
          model->Input(reinterpret_cast<float const**>(in));
        }
        model->StoreInput();
      }
    }
  }

  return true;
}

void ModelInstance::Start() {
  if (!running_) {
    running_ = true;
    ready_to_launch_ = num_models_;
    next_to_launch_ = 0;
    next_to_reap_ = 0;
    next_to_consume_ = 0;
  } else {
    throw std::runtime_error("Already started.");
  }
}

bool ModelInstance::Loop(bool may_launch) {
  if (unlikely(!running_)) return false;

  if (reap_) Reap();
  if (spsc_.TryConsume()) {
    auto const batch_size = model_->BatchSize();
    auto const idx_model = next_to_consume_;
    if (batch_size == 1) {
      result_->Add(output_[idx_model][0], ts_start_[idx_model][0], ts_stop_[idx_model]);
    } else {
      result_->AddBatch(batch_size, output_[idx_model].get(), ts_start_[idx_model].data(), ts_stop_[idx_model]);
    }
    StateCheck(states_[idx_model] == State::kInference);
    StateCheck(models_[idx_model]->OutputReady());
    models_[idx_model]->PostProcess();
    states_[idx_model] = State::kPostProcess;
    next_to_consume_ = NextModelIndex(idx_model);
    ++ready_to_launch_;
  }

  if (ready_to_launch_ > 0) {
    if (unlikely(stopping_)) {
      if (ready_to_launch_ == num_models_) {
        // All models have stopped inference -- graceful shutdown
        bool allStopped = true;
        for (std::size_t idx = 0; idx < num_models_; ++idx) {
          if (states_[idx] == State::kPostProcess) {
            if (models_[idx]->PostProcessReady()) {
              states_[idx] = State::kStopped;
            } else {
              allStopped = false;
            }
          } else {
            allStopped = false;
          }
          if ((states_[idx] == State::kPreProcess || states_[idx] == State::kInference) && may_launch && !allStopped) {
            return Inference();
          }
        }
        if (allStopped) {
          stopping_ = false;
          running_ = false;
        }
      }
    } else if (may_launch) {
      return Inference();
    }
  }

  return false;
}

void ModelInstance::Reap() {
  auto const idx_model = next_to_reap_;
  if (spsc_.CanProduce() && models_[idx_model]->OutputReady() && spsc_.TryProduce()) {
    ts_stop_[idx_model] = Clock::now();
#ifdef USE_NVTX
    nvtx_ranges_[idx_model].reset();
#endif
    next_to_reap_ = NextModelIndex(idx_model);
    spsc_.NotifyProduced();
  }
}

void ModelInstance::Stop() {
  if (running_ && !stopping_) {
    stopping_ = true;
  }
}

namespace {

template <typename TData>
std::unique_ptr<ModelInterface> MakeModel(
    ModelDef const& model_def, CudaEnv& cuda_env, std::uint32_t batch_size, ModelConfig const& config) {
  StateCheck(batch_size == 1);  // single-step always requires batch_size == 1

  if (model_def.IsPingPong()) {
    StateCheck(config.algorithm == ModelConfig::Algo::PingPongPersistent);
    return std::make_unique<PingPongModelPersistent<TData>>(cuda_env, model_def, batch_size, config);
  }

  if (config.algorithm == ModelConfig::Algo::SingleStep || config.algorithm == ModelConfig::Algo::SingleStepStreaming) {
    return std::make_unique<ModelSingleStep<TData>>(cuda_env, model_def, config);
  }

  if (config.algorithm == ModelConfig::Algo::SingleStepGemv) {
    if (config.as_supported) {
      return std::make_unique<ModelAsSingleStepGemv<TData>>(cuda_env, model_def, config);
    }
    return std::make_unique<ModelSingleStepGemv<TData>>(cuda_env, model_def, config);
  }

  throw std::runtime_error("Invalid algorithm for single-step mode");
}

}  // namespace

std::unique_ptr<ModelInterface> ModelInstance::CreateModel(
    ModelDef const& model_def, CudaEnv& cuda_env, std::uint32_t batch_size, ModelConfig const& config) {
  switch (config.Type()) {
    case ssgemv::DATA_TYPE_FP32:
      return MakeModel<float>(model_def, cuda_env, batch_size, config);
    case ssgemv::DATA_TYPE_FP16:
      return MakeModel<half>(model_def, cuda_env, batch_size, config);
    case ssgemv::DATA_TYPE_BF16:
      throw std::invalid_argument("BF16 data type not yet supported");
    default:
      throw std::invalid_argument("unsupported data type");
  }
}

std::unique_ptr<ModelInterface> ModelInstance::CreateModel(
    std::filesystem::path const& path, CudaEnv& cuda_env, std::uint32_t batch_size, ModelConfig const& config) {
  return CreateModel(ModelDef{path.c_str()}, cuda_env, batch_size, config);
}

void ModelInstance::Reset() {
  StateCheck(!running_);
  result_->Reset();
  std::visit([](auto& d) { d->Reset(); }, data_);
  Initialize();
}

// clang-format off
#define ForAllBatches(operation) \
  for (std::uint32_t i = 0; i < batch_size; ++i) operation

#define MaybeOmp1(operation)                                                                                 \
  if (use_omp_) {                                                                                            \
    _Pragma(                                                                                                 \
        "omp parallel for default(none) num_threads(omp_num_threads_) shared(batch_size, idx_model, input, num_steps, num_features)") \
    ForAllBatches(operation)                                                                                 \
  } else {                                                                                                   \
    ForAllBatches(operation)                                                                                 \
  }

#define MaybeOmp2(operation)                                                                                 \
  if (use_omp_) {                                                                                            \
    _Pragma(                                                                                                 \
        "omp parallel for default(none) num_threads(omp_num_threads_) shared(batch_size, idx_model, input, input_size, input_size_bytes)") \
    ForAllBatches(operation)                                                                                 \
  } else {                                                                                                   \
    ForAllBatches(operation)                                                                                 \
  }

// clang-format on

void ModelInstance::Stage() {
  auto const batch_size = model_->BatchSize();
  auto const idx_model = next_to_launch_;
  auto const& input = input_[idx_model];

  if (transpose_ && batch_size > 1) {
    auto const num_steps = model_->NumSteps();
    auto const num_features = model_->NumFeatures();
    if (is_half_) {
      if (holds_half_) {
        MaybeOmp1({
          auto const dst = staging_16_[idx_model].get() + i * num_features;
          util::CopyTranspose(dst, static_cast<half const*>(input[i]), batch_size, num_steps, num_features);
        })
      } else {
        MaybeOmp1({
          auto const dst = staging_16_[idx_model].get() + i * num_features;
          cex::CopyTransposeConvert(dst, static_cast<float const*>(input[i]), batch_size, num_steps, num_features);
        })
      }
    } else {
      MaybeOmp1({
        auto const dst = staging_32_[idx_model].get() + i * num_features;
        util::CopyTranspose(dst, static_cast<float const*>(input[i]), batch_size, num_steps, num_features);
      })
    }
  } else {
    auto const input_size = model_->InputSize();
    if (is_half_) {
      auto const input_size_bytes = input_size * sizeof(half);
      if (holds_half_) {
        MaybeOmp2({
          auto const dst = staging_16_[idx_model].get() + i * input_size;
          std::memcpy(dst, input[i], input_size_bytes);
        })
      } else {
        MaybeOmp2({
          auto const dst = staging_16_[idx_model].get() + i * input_size;
          cex::FloatToHalf(dst, static_cast<float const*>(input[i]), input_size);
        })
      }
    } else {
      auto const input_size_bytes = input_size * sizeof(float);
      MaybeOmp2({ std::memcpy(staging_32_[idx_model].get() + i * input_size, input[i], input_size_bytes); })
    }
  }
}

bool ModelInstance::TryOpenMp() {
  if (stage_) {
    // prepare data for micro benchmark without advancing the data iterator
    auto const batch_size = model_->BatchSize();
    auto const num_samples = std::visit([](auto& d) { return d->NumSamples(); }, data_);
    auto const num_features = std::visit([](auto& d) { return d->NumFeatures(); }, data_);
    auto const data_ptr = std::visit([](auto& d) { return static_cast<void const*>(d->Data().data()); }, data_);
    auto const& input = input_[next_to_launch_];
    for (std::size_t i = 0; i < batch_size; ++i) {
      auto const offset = (i % num_samples) * num_features;
      if (holds_half_) {
        input[i] = static_cast<half const*>(data_ptr) + offset;
      } else {
        input[i] = static_cast<float const*>(data_ptr) + offset;
      }
    }

    // estimate a reasonable number of iterations
    std::size_t constexpr iter_warm_up_max{1 << 5};
    std::size_t constexpr iter_measure_max{1 << 10};
    auto constexpr duration_warm_up_max = std::chrono::milliseconds(50);
    auto constexpr duration_measure_max = std::chrono::milliseconds(500);
    use_omp_ = false;
    auto staging_func = [this]() { Stage(); };
    auto const duration_single = util::TimeIt(1, staging_func);
    auto const iter_warm_up =
        std::max(std::min(static_cast<std::size_t>(duration_warm_up_max / duration_single), iter_warm_up_max), 1ul);
    auto const iter_measure =
        std::max(std::min(static_cast<std::size_t>(duration_measure_max / duration_single), iter_measure_max), 1ul);

    util::TimeIt(iter_warm_up, staging_func);
    auto const duration_omp_false = util::TimeIt(iter_measure, staging_func);
    use_omp_ = true;
    util::TimeIt(iter_warm_up, staging_func);
    auto const duration_omp_true = util::TimeIt(iter_measure, staging_func);

    // use benchmark results for setting the OpenMP flag
    return duration_omp_true * 100 < duration_omp_false * 95;
  } else {
    return false;
  }
}
ModelInstance::VoidIterator ModelInstance::CreateVoidIterator(ModelInstance::AnyDataIterator& data) {
  return std::visit(
      [](auto& dataIterator) {
        auto ptr = dataIterator.get();
        return VoidIterator{[ptr]() { return static_cast<void const*>(ptr->Next()); }};
      },
      data);
}

}  // namespace nvidia::quant::lowlat_infer
