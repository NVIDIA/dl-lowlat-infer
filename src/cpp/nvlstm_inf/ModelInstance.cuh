/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef NLSTMINF_SRC_CPP_NVLSTMINF_MODELINSTANCE_CUH_
#define NLSTMINF_SRC_CPP_NVLSTMINF_MODELINSTANCE_CUH_

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "CudaEnv.cuh"
#include "DataIterator.h"
#include "InferenceResult.h"
#include "ModelConfig.cuh"
#include "ModelDef.h"
#include "ModelInterface.h"
#include "cuda_helpers.cuh"

namespace nvidia::quant::lowlat_infer {

class ModelInstance {
 public:

  enum class State {
    kStopped,
    kPreProcess,
    kInference,
    kPostProcess,
    kInitialized,
  };

  using PinnedDataIterator16 = DataIterator<cex::PinnedVector16>;
  using PinnedDataIterator32 = DataIterator<cex::PinnedVector32>;
  using PinnedDataIteratorPtr16 = std::unique_ptr<PinnedDataIterator16>;
  using PinnedDataIteratorPtr32 = std::unique_ptr<PinnedDataIterator32>;
  using AnyDataIterator = std::variant<PinnedDataIteratorPtr16, PinnedDataIteratorPtr32>;
  using ModelVector = std::vector<std::unique_ptr<ModelInterface>>;

  ModelInstance(ModelVector&& models,
                AnyDataIterator&& data,
                std::size_t init_capacity = InferenceResult::kDefaultCapacity,
                bool stage = false,
                std::optional<bool> use_omp = std::nullopt,
                bool reap = true);

  ModelInstance(std::unique_ptr<ModelInterface>&& model,
                AnyDataIterator&& data,
                std::size_t init_capacity = InferenceResult::kDefaultCapacity,
                bool stage = false,
                std::optional<bool> use_omp = std::nullopt,
                bool reap = true)
      : ModelInstance{WrapModel(std::move(model)), std::move(data), init_capacity, stage, use_omp, reap} {}

  void Start();

  // The return value indicates whether a new inference was launched.
  bool Loop(bool may_launch = true);

  // Automatically called in Loop() if reap_ is set. May be called additionally from a different thread.
  void Reap();

  void Stop();

  void Reset();

  void Finalize();

  [[nodiscard]] bool Running() const { return running_; }
  [[nodiscard]] bool Stopping() const { return stopping_; }
  [[nodiscard]] InferenceResult& Result() const { return *result_; }
  [[nodiscard]] InferenceResult& Result() { return *result_; }
  [[nodiscard]] std::shared_ptr<InferenceResult> ResultShared() { return result_; }
  [[nodiscard]] std::shared_ptr<InferenceResult const> ResultShared() const { return result_; }
  [[nodiscard]] bool UseOmp() const { return use_omp_; }

  static std::unique_ptr<ModelInterface> CreateModel(ModelDef const& model_def,
                                                     CudaEnv& cuda_env,
                                                     std::uint32_t batch_size,
                                                     ModelConfig const& config);

  static std::unique_ptr<ModelInterface> CreateModel(std::filesystem::path const& path,
                                                     CudaEnv& cuda_env,
                                                     std::uint32_t batch_size = 1,
                                                     ModelConfig const& config = {});

  void SetModelId(std::uint32_t id) {model_id_ = id;};
  std::uint32_t GetModelId() { return model_id_;};

 private:
  using Clock = InferenceResult::Clock;
  using VoidIterator = std::function<void const*()>;

  static VoidIterator CreateVoidIterator(AnyDataIterator& data);

  static ModelVector WrapModel(std::unique_ptr<ModelInterface>&& model) {
    ModelVector result(1);
    result[0] = std::move(model);
    return result;
  }

  void Initialize();
  bool Inference();
  void Stage();
  bool TryOpenMp();

  [[nodiscard]] inline std::uint32_t NextModelIndex(std::uint32_t idx) const {
    return (idx + 1) % num_models_;
  }

  ModelVector models_;
  std::size_t const num_models_;
  std::unique_ptr<ModelInterface>& model_;
  AnyDataIterator data_;
  std::function<void const*()> const void_iterator_;
  std::shared_ptr<InferenceResult> result_;
  bool const stage_;
  int const omp_num_threads_;
  bool use_omp_;
  bool const is_half_, holds_half_;
  bool const transpose_;

  std::vector<cex::PinnedPtr<void const*>> input_;
  std::vector<cex::PinnedPtr<float>> output_;
  std::vector<cex::PinnedPtr<float>> staging_32_;
  std::vector<cex::PinnedPtr<half>> staging_16_;

  std::vector<ModelInterface::Graph> graph_;
  std::vector<std::vector<Clock::time_point>> ts_start_;
  std::vector<Clock::time_point> ts_stop_;
#ifdef USE_NVTX
  std::vector<std::shared_ptr<void>> nvtx_ranges_;
#endif

  bool running_{false};
  bool stopping_{false};
  std::uint32_t ready_to_launch_{0};
  std::uint32_t next_to_launch_{}, next_to_reap_{}, next_to_consume_{};

  bool const reap_;
  util::SingleProducerSingleConsumer spsc_{0, 0};

  std::vector<State> states_;

  std::uint32_t model_id_;
};

}  // namespace nvidia::quant::lowlat_infer

#endif  // NLSTMINF_SRC_CPP_NVLSTMINF_MODELINSTANCE_CUH_
