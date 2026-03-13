/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "ModelSingleStepGemv.cuh"
#include "util.h"

#include <iostream>

namespace nvidia::quant::lowlat_infer {

template <typename TData>
ModelSingleStepGemv<TData>::ModelSingleStepGemv(CudaEnv &cuda_env, const ModelDef &model_def, const ModelConfig &config)
    : Base{cuda_env, model_def, 1, false, true}  // full_windows=false, single_step=true
    , cuda_env_{cuda_env}
    , bias_{model_def.HasBias() ? model_def.Bias() : 0.0f}
    , num_layers_(model_def.NumLayers())
    , num_steps_(model_def.NumSteps())
    , num_units_(model_def.NumUnits())
    , num_features_(model_def.NumFeatures()) {

  static_assert(NUM_GATES == kNumGates);

  switch(num_features_) {
    case LSTM_S_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_s_params_type>(); break;
    case LSTM_M_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_m_params_type>(); break;
    case LSTM_L_NUM_INPUTS: params_ptr_ = std::make_unique<lstm_l_params_type>(); break;
    default:
    throw std::runtime_error("Unsupported num_features: " + std::to_string(num_features_) +
                            ". Expected: " + std::to_string(LSTM_S_NUM_INPUTS) + ", " +
                            std::to_string(LSTM_M_NUM_INPUTS) + ", or " +
                            std::to_string(LSTM_L_NUM_INPUTS));
  }

  StateCheck(config.skip_scalar_product_in_precompute);
  model_singe_step_ = std::make_unique<ModelSingleStep<TData>>(cuda_env, model_def, config);

  auto custom_stream = cuda_env.CustomStream();

  // Populate descriptor objects
  auto const num_units_gates = num_units_ * kNumGates;

#if AARCH64_GH == 1
  pinned_ref_out_ = cex::MallocPaged<float>(1);
#else
  pinned_ref_out_ = cex::MallocHost<float>(1);
#endif

  dev_input_weights_.resize(num_layers_);
  dev_biases_.resize(num_layers_);

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    dev_input_buf_ = cex::Malloc<data_type>(std::max(num_features_, num_units_));
  }

  std::vector<TData> tmp_h(kNumGates * std::max(num_units_, num_features_) * num_units_);

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    std::uint32_t num_inputs = (layer == 0)? num_features_ : num_units_;

    // weights
    dev_input_weights_[layer] = cex::Malloc<data_type>(num_units_gates * num_inputs);
    dev_biases_[layer] = cex::Malloc<data_type>(num_units_gates);

    auto &&matrix_W_conv = Convert(model_def.LstmW().at(layer).Data());
    ssgemv::weights_permute(tmp_h.data(), matrix_W_conv.data(), kNumGates, num_inputs, num_units_);
    cex::Copy(dev_input_weights_[layer].get(), tmp_h.data(), num_units_gates * num_inputs, custom_stream);

    // biases
    std::vector<float> biases_reduced(kNumGates * num_units_);
    for (std::size_t idx_gate = 0; idx_gate < kNumGates; ++idx_gate) {
      auto const gate = kGates.at(idx_gate);
      const float *biases1 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, false);
      const float *biases2 = model_def.LstmB().at(layer).Data().data() + model_def.OffsetB(gate, true);
      for (std::size_t k = 0; k < num_units_; ++k) {
        biases_reduced[idx_gate * num_units_ + k] = biases1[k] + biases2[k];
      }
    }
    auto &&bias_conv = Convert(biases_reduced);
    ssgemv::gate_transpose(tmp_h.data(), bias_conv.data(), kNumGates * num_units_, 1);
    cex::Copy(dev_biases_[layer].get(), tmp_h.data(), kNumGates * num_units_, custom_stream);

    // dot product
    dev_mul_ = cex::Malloc<float>(num_units_);
    cex::Copy(dev_mul_, model_def.MatMul().Data().data(), num_units_, custom_stream);
  } // end layer loop

  GetParamsPtr()->max_smem_size = config.max_shared_mem_per_block;
  GetParamsPtr()->sm_count = config.sm_count;
  GetParamsPtr()->sm_arch = config.sm_arch;
};

template <typename TData>
template <typename Params>
void ModelSingleStepGemv<TData>::InitParams (Params& params) {
  Base::InitParams();

  for (std::size_t layer = 0; layer < num_layers_; ++layer) {
    params.gmem_weights[layer] = dev_input_weights_[layer].get();
    params.gmem_biases[layer] = dev_biases_[layer].get();

    const auto& dev_gemms = model_singe_step_->Gemms();
    std::size_t leading_dim = dev_gemms[layer].LeadingDim();

    params.gmem_gemm_h[layer] = (void *)(dev_gemms[layer].Get() + (num_steps_ - 1) * leading_dim);

    const auto& dev_cxs_in = (odd_step_ && Base::IsCapturing())? model_singe_step_->CxOut() : model_singe_step_->CxIn();
    leading_dim = dev_cxs_in[layer].LeadingDim();
    params.gmem_c_in[layer] = (void *)(dev_cxs_in[layer].Get() + (num_steps_ - 1) * leading_dim);
  }
  params.dot_prod_bias = bias_;
  params.gmem_dot_prod_weights = (void *)dev_mul_.get();
  params.host_ld_complete = Base::HostSignal();

  params.gmem_ld_complete = Base::DeviceSignal();
  params.use_gdrcopy = Env().UseGdrcopy();


  // Todo move to base
  params.host_addr = (Base::IsStaging())? nullptr : Base::PinedInputAddr();

  if (num_features_ == LSTM_L_NUM_INPUTS) {
    params.gmem_h_out = dev_input_buf_.get();
  }

  params.gmem_input = reinterpret_cast<void *>(Base::HostInputPtr());
}

template <typename TData>
void ModelSingleStepGemv<TData>::InitParams () {
  switch(num_features_) {
    case LSTM_S_NUM_INPUTS: InitParams(*static_cast<lstm_s_params_type *>(params_ptr_.get())); break;
    case LSTM_M_NUM_INPUTS: InitParams(*static_cast<lstm_m_params_type *>(params_ptr_.get())); break;
    case LSTM_L_NUM_INPUTS: InitParams(*static_cast<lstm_l_params_type *>(params_ptr_.get())); break;
    default: break;
  }
}

template <typename TData>
void ModelSingleStepGemv<TData>::PreCompute() {
  
  if (Base::GraphPtr() == nullptr) 
  {
    model_singe_step_->SetPinnedOut(pinned_ref_out_.get());
    model_singe_step_->PreProcess();
    if (Base::IsStaging()) {
      // Unliqely to happen without capturing the graph
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputPtr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const *>(Base::HostInputPtr()));
      }
    } else {
      if (Base::IsFp32Input()) {
        model_singe_step_->Infer(Base::HostInputAddr());
      } else {
        model_singe_step_->Infer(reinterpret_cast<half const **>(Base::HostInputAddr()));
      }
    }
    model_singe_step_->Output(pinned_ref_out_.get());
    model_singe_step_->PostProcess(); 
  } else {
    model_singe_step_->PreProcess();
    model_singe_step_->PreProcessWait();
    model_singe_step_->Launch(*Base::GraphPtr());
    model_singe_step_->InputWait();
    model_singe_step_->OutputWait();
    model_singe_step_->PostProcess(); 

  }
};

template <typename TData>
void ModelSingleStepGemv<TData>::PreProcess() {
  Base::PreProcess();
}

template <typename TData>
void ModelSingleStepGemv<TData>::PostProcess() {

  PreCompute();  

};

template <typename TData>
bool ModelSingleStepGemv<TData>::PostProcessReady() {
  return model_singe_step_->PostProcessReady();
}

template <typename TData>
void ModelSingleStepGemv<TData>::PostProcessWait() {
  model_singe_step_->PostProcessWait();
}

}  // namespace nvidia::quant::lowlat_infer

template class nvidia::quant::lowlat_infer::ModelSingleStepGemv<float>;
template class nvidia::quant::lowlat_infer::ModelSingleStepGemv<half>;
