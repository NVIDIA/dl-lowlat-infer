/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <algorithm>
#include <cassert>
#include <exception>
#include <type_traits>

#include "ModelSingleStepBase.cuh"
#include "cuda_copy.cuh"

using namespace nvidia::quant::lowlat_infer;


template <typename TData>
ModelSingleStepBase<TData>::ModelSingleStepBase(
    CudaEnv &cuda_env,  // NOLINT(cppcoreguidelines-pro-type-member-init)
    ModelDef const &model_def,
    std::uint32_t batch_size,
    const Config &cfg)
    : ModelInterface{batch_size, cfg.streaming ? 1u : model_def.NumSteps(), model_def.NumFeatures(), model_def.NumUnits()},
      cuda_env_{cuda_env},
      skip_scalar_product_{cfg.skip_scalar_product},
      bias_{model_def.HasBias() ? model_def.Bias() : 0.0f} {
  static_assert(std::is_same_v<data_type, half> || std::is_same_v<data_type, float>);

  StateCheck(batch_size == 1);

  // Populate descriptor objects
  auto const seq_length = NumSteps();
  auto const input_size = NumFeatures();
  dev_x_ = cex::Malloc<data_type>(input_size);
        
  dev_x_expanded_ = cex::Malloc<data_type>(seq_length * batch_size * input_size);

  auto const num_units = NumUnits();
  auto const output_tensor_size = seq_length * batch_size * num_units;
  dev_y_ = cex::Malloc<data_type>(output_tensor_size);
  dev_y_last_ = dev_y_.get() + (seq_length - 1) * batch_size * num_units;

  auto stream = Env().Stream();

  auto &&mat_mul = Convert(model_def.MatMul().Data());
  dev_mul_ = cex::Malloc<data_type>(mat_mul.size());
  cex::Copy(dev_mul_, mat_mul.data(), mat_mul.size(), stream);
  CublasCheck(cublasSetPointerMode(cuda_env.Cublas(), CUBLAS_POINTER_MODE_DEVICE));
  dev_out_ = cex::Malloc<float>(batch_size);

  event_in_start_ = cex::CreateEvent();
  event_in_stop_ = cex::CreateEvent();
  event_out_start_ = cex::CreateEvent();
  event_out_stop_ = cex::CreateEvent();

  Env().CreateStreams(1);
}

template <typename TData>
void ModelSingleStepBase<TData>::Aggregate() {
  assert(dev_y_last_ != nullptr);
  auto const tgt = pinned_out_ ? pinned_out_ : dev_out_.get();
  if (!skip_scalar_product_) {
    copy::ScalarProduct(tgt, dev_y_last_, dev_mul_.get(), bias_, NumUnits(), Env().Stream());
  }
}

template <typename TData>
void ModelSingleStepBase<TData>::InitX(util::Tensor32 const &t) {

  auto const &data = t.Data();
  auto const data_size = data.size();
  auto const num_steps = NumSteps();
  auto const num_features = NumFeatures();

  if (data_size > 0) {
    auto const &shape = t.Shape();
    StateCheck(shape.size() == 2);
    StateCheck(shape[0] == num_steps - 1);
    StateCheck(shape[1] == num_features);
  }

  auto stream = Env().Stream();
  auto &x = dev_x_;

  odd_step_ = false;
  dev_x_last_ = x.get();
  if (data_size > 0) {
    auto data_dev = cex::Malloc<float>(data_size);
    cex::Copy(data_dev, data.data(), data_size, stream);
    auto const begin = data_dev.get();
    auto const end = begin + t.Data().size();
    for (auto step = begin; step != end; step += num_features) {
      PreProcess();
      Infer(step);
      ShiftX();
      Env().JoinStreams();
      PostProcess();
    }
  }

  CudaCheck(cudaStreamSynchronize(stream));
  assert(dev_x_last_ != nullptr);
}

template <typename TData>
void ModelSingleStepBase<TData>::ShiftX() {
    odd_step_ = !odd_step_;
}

template <typename TData>
float const *ModelSingleStepBase<TData>::StageHostPointer(const float *host_in, cudaStream_t stream) {
  cudaPointerAttributes pointer_attributes{};
  CudaCheck(cudaPointerGetAttributes(&pointer_attributes, host_in));
  if (pointer_attributes.type == cudaMemoryTypeHost) {
    auto const num_elements = BatchSize() * NumSteps() * NumFeatures();
    cex::Copy(dev_x_staging_, host_in, num_elements, stream);
    return dev_x_staging_.get();
  }
  return host_in;
}

template <typename TData>
void ModelSingleStepBase<TData>::Input(float const *host_in, bool sequence_major) {
  auto stream_hp = Env().ForkHighPriority(event_in_start_.get());
  assert(!sequence_major);
  StateCheck(dev_x_last_ != nullptr);
  auto const input_size = BatchSize() * NumFeatures();
  copy::Convert(dev_x_last_, host_in, input_size, stream_hp);

  Env().JoinHighPriority(event_in_stop_.get());
  
  ExpandInput();
}

template <typename TData>
void ModelSingleStepBase<TData>::Input(half const *host_in, bool sequence_major) {
  if constexpr (std::is_same_v<data_type, half>) {
    auto stream_hp = Env().ForkHighPriority(event_in_start_.get());
    assert(!sequence_major);
    StateCheck(dev_x_last_ != nullptr);
    auto const input_size = BatchSize() * NumFeatures();
    cex::Copy(dev_x_last_, host_in, input_size, stream_hp);
    Env().JoinHighPriority(event_in_stop_.get());
    ExpandInput();
  } else {
    throw std::runtime_error("Input of type half is only supported for FP16 models.");
  }
}

template <typename TData>
void ModelSingleStepBase<TData>::Input(float const **host_addr) {
  InputGeneric(host_addr);
}

template <typename TData>
void ModelSingleStepBase<TData>::Input(half const **host_addr) {
  if constexpr (std::is_same_v<data_type, half>) {
    InputGeneric(host_addr);
  } else {
    throw std::runtime_error("Input of type half is only supported for FP16 models.");
  }
}

template <typename TData>
template <typename TInput>
void ModelSingleStepBase<TData>::InputGeneric(TInput const **host_addr) {
  auto stream_hp = Env().ForkHighPriority(event_in_start_.get());
  StateCheck(dev_x_last_ != nullptr);
  copy::Convert(dev_x_last_, host_addr, BatchSize(), NumFeatures(), stream_hp);

  Env().JoinHighPriority(event_in_stop_.get());
  ExpandInput();
}

template <typename TData>
void ModelSingleStepBase<TData>::Output(float *host_out, bool wait, bool shift) {
  auto stream_hp = Env().ForkHighPriority(event_out_start_.get());
  if (!pinned_out_) {
    copy::Convert(host_out, dev_out_.get(), BatchSize(), stream_hp);
  }
  if (shift) ShiftX();
  Env().JoinHighPriority(event_out_stop_.get());
  Env().JoinStreams();
  if (wait) OutputWait();
}

template <typename TData>
bool ModelSingleStepBase<TData>::InputReady() {
  return cex::EventReady(event_in_stop_.get());
}

template <typename TData>
void ModelSingleStepBase<TData>::InputWait() {
  CudaCheck(cudaEventSynchronize(event_in_stop_.get()));
}

template <typename TData>
bool ModelSingleStepBase<TData>::OutputReady() {
  return cex::EventReady(event_out_stop_.get());
}

template <typename TData>
void ModelSingleStepBase<TData>::OutputWait() {
  CudaCheck(cudaEventSynchronize(event_out_stop_.get()));
}

template <typename TData>
void ModelSingleStepBase<TData>::ExpandInput() {
 if (!skip_replicate_) {
  copy::Replicate(dev_x_expanded_.get(), dev_x_last_, NumFeatures(), NumSteps(), cuda_env_.Stream());
 }
}

template <typename TData>
void ModelSingleStepBase<TData>::StoreInput() {
  StateCheck(!SingleStep());
}

template <typename TData>
std::conditional_t<std::is_same_v<TData, float>, std::vector<float> const &, std::vector<TData>>
ModelSingleStepBase<TData>::Convert(const std::vector<float> &input) {
  if constexpr (std::is_same_v<TData, float>) {
    return input;
  } else {
    static_assert(std::is_same_v<TData, half>);
    return cex::FloatToHalf(input.data(), input.size());
  }
}
template <typename TData>
typename ModelSingleStepBase<TData>::Graph ModelSingleStepBase<TData>::Capture(float const *input,
                                                                   float *host_out,
                                                                   bool sequence_major) {
  return Capture([this, input, host_out, sequence_major]() {
    SetPinnedOut(host_out);
    Infer(input, sequence_major);
    auto constexpr wait = false;
    auto constexpr shift = true;
    Output(host_out, wait, shift);
    ResetPinnedOut();
  });
}

template <typename TData>
typename ModelSingleStepBase<TData>::Graph ModelSingleStepBase<TData>::Capture(half const *input,
                                                                   float *host_out,
                                                                   bool sequence_major) {
  return Capture([this, input, host_out, sequence_major]() {
    SetPinnedOut(host_out);
    Infer(input, sequence_major);
    auto constexpr wait = false;
    auto constexpr shift = true;
    Output(host_out, wait, shift);
    ResetPinnedOut();
  });
}

template <typename TData>
ModelInterface::Graph ModelSingleStepBase<TData>::Capture(const float **input, float *host_out) {
  return Capture([this, input, host_out]() {
    SetPinnedOut(host_out);
    Infer(input);
    auto constexpr wait = false;
    auto constexpr shift = true;
    Output(host_out, wait, shift);
    ResetPinnedOut();
  });
}

template <typename TData>
ModelInterface::Graph ModelSingleStepBase<TData>::Capture(const half **input, float *host_out) {
  return Capture([this, input, host_out]() {
    SetPinnedOut(host_out);
    Infer(input);
    auto constexpr wait = false;
    auto constexpr shift = true;
    Output(host_out, wait, shift);
    ResetPinnedOut();
  });
}

template <typename TData>
ModelInterface::Graph ModelSingleStepBase<TData>::Capture(std::function<void()> const &ops) {
  auto stream = cuda_env_.Stream();
  auto const num_graphs = SingleStep() ? 2u : 1u;
  Graph result{num_graphs};
  result.reserve(num_graphs);
  auto const odd_step_start = odd_step_;
  for (auto i = 0u; i < num_graphs; ++i) {
    auto const idx_graph = static_cast<std::uint32_t>(odd_step_);
    cudaGraph_t graph;
    CudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
      ops();
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      std::rethrow_exception(std::current_exception());
    }
    CudaCheck(cudaStreamEndCapture(stream, &graph));
    std::size_t num_root_nodes{1};
    std::vector<cudaGraphNode_t> root_nodes(num_root_nodes);
    CudaCheck(cudaGraphGetRootNodes(graph, root_nodes.data(), &num_root_nodes));
    StateCheck(num_root_nodes == 1);
    cudaGraphNode_t event_in_start_node;
    CudaCheck(cudaGraphAddEventRecordNode(&event_in_start_node, graph, nullptr, 0, event_in_start_.get()));
    CudaCheck(cudaGraphAddDependencies(graph, &event_in_start_node, root_nodes.data(), NULL, 1));
    cudaGraphNode_t event_in_stop_node;
    CudaCheck(cudaGraphAddEventRecordNode(
        &event_in_stop_node, graph, root_nodes.data(), num_root_nodes, event_in_stop_.get()));
    std::size_t num_all_nodes{0};
    CudaCheck(cudaGraphGetNodes(graph, nullptr, &num_all_nodes));
    std::vector<cudaGraphNode_t> all_nodes(num_all_nodes);
    CudaCheck(cudaGraphGetNodes(graph, all_nodes.data(), &num_all_nodes));
    std::vector<cudaGraphNode_t> leaf_nodes{};
    for (std::size_t n = 0; n < num_all_nodes; ++n) {
      auto node = all_nodes[n];
      std::size_t num_dependent{};
      CudaCheck(cudaGraphNodeGetDependentNodes(node, nullptr, NULL, &num_dependent));
      if (num_dependent == 0 && node != event_in_stop_node) {
        leaf_nodes.push_back(node);
      }
    }
    cudaGraphNode_t event_out_stop_node;
    CudaCheck(cudaGraphAddEventRecordNode(
        &event_out_stop_node, graph, leaf_nodes.data(), leaf_nodes.size(), event_out_stop_.get()));
    cudaGraphExec_t graph_exec;
    CudaCheck(cudaGraphInstantiate(&graph_exec, graph));
    CudaCheck(cudaGraphUpload(graph_exec, stream));
    CudaCheck(cudaGraphDestroy(graph));
    result.at(idx_graph) =
        Graph::value_type{graph_exec, [](cudaGraphExec_t ge) { CudaCheck(cudaGraphExecDestroy(ge)); }};
  }
  StateCheck(odd_step_start == odd_step_);
  return result;
}

template <typename TData>
void ModelSingleStepBase<TData>::Launch(ModelInterface::Graph &graph, cudaHostFn_t call_back, void *user_data) {
  auto stream = cuda_env_.Stream();
  CudaCheck(cudaGraphLaunch(graph.at(odd_step_).get(), stream));
  if (call_back != nullptr) {
    CudaCheck(cudaLaunchHostFunc(stream, call_back, user_data));
  }
  odd_step_ = !odd_step_;
}

template <typename TData>
bool ModelSingleStepBase<TData>::Launched() {
  return cex::EventReady(event_in_start_.get());
}

template <typename TData>
void ModelSingleStepBase<TData>::ClearOutput(cudaStream_t stream) {
  copy::Memset(dev_out_.get(), bias_, BatchSize(), stream);
}

template class nvidia::quant::lowlat_infer::ModelSingleStepBase<float>;
template class nvidia::quant::lowlat_infer::ModelSingleStepBase<half>;
