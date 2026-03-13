/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cub/cub.cuh>

#include "nvlstm_inf/ss_gemv_impl/gemv_lstm_utils.cuh"
#include "nvlstm_inf/ss_gemv_impl/kernel_params.h"

namespace nvidia::quant::lowlat_infer::single_step {

__forceinline__ __device__ float sigma_g(float in) { return __fdividef(1.f, (1.f + __expf(-in))); }
__forceinline__ __device__ float sigma_c(float in) { return tanhf(in); }

template <typename TData, int ELEMS_>
struct __align__(8) RType {
  enum { ELEMS = ELEMS_ };
  TData val[ELEMS];
};

struct ReduceLayer0 {
  template <template <class, int> class R, class TData, int ELEMS>
  __device__ __forceinline__ R<TData, ELEMS> operator()(const R<TData, ELEMS>& a, const R<TData, ELEMS>& b) const {
    R<TData, ELEMS> res;
/// TODO - support hfma2
#pragma unroll
    for (int ii = 0; ii < ELEMS; ii++) {
      res.val[ii] = a.val[ii] + b.val[ii];
    }
    return res;
  }
};

struct Layer0_eltwise_params {
  void* gmem_input_weights;
  void* gmem_gemm_h;
  void* gmem_gemm_x;
  int gemm_leading_dim;
  void* gmem_bias;
  void* gmem_cx_in;
  void* gmem_cx_out;
  void* gmem_hx_out;
  void* gmem_y;
  unsigned int hidden_leading_dim;
  unsigned int hidden_size;
  unsigned int seq_size;
  bool write_y;
};

// replicate input vectors in flight
// making the first gemm be rather gemv
// TODO transpose input weights in constructor
template <typename TData, int ELEMENTS_PER_THREAD, int BLOCK_THREADS>
__global__ void single_step_layer_0(
    TData const* __restrict__ dev_input_weights,  // 4 * num_units x num_inputs
    int input_weights_leading_dim,
    TData const* __restrict__ dev_x,
    int num_features,
    TData* __restrict__ y,
    TData const* __restrict__ gemm_h,
    unsigned int const gemm_leading_dim,
    TData const* __restrict__ bias,
    TData const* __restrict__ cx_in,
    TData* __restrict__ cx_out,
    TData* __restrict__ hx_out,
    const unsigned int hidden_leading_dim,
    const unsigned int hidden_size,
    const unsigned int seq_size,
    bool write_y) {
  // GEMM_H_VECTORIZED == 0 implementation
  constexpr int GEMM_X_ELEMENTS_PER_THREAD = ELEMENTS_PER_THREAD;
  using Reduce_type = RType<TData, 4 * GEMM_X_ELEMENTS_PER_THREAD>;
  using BlockReduce = cub::BlockReduce<Reduce_type, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ TData dot_prod[4 * GEMM_X_ELEMENTS_PER_THREAD];
  if (threadIdx.x == 0) {
    for (int jj = 0; jj < 4 * GEMM_X_ELEMENTS_PER_THREAD; jj++) {
      dot_prod[jj] = TData(0);
    }
  }

  // each block processes slice of size GEMM_X_ELEMENTS_PER_THREAD of num_units outputs
  const TData* gemm_input_weights_gate_0 =
      dev_input_weights + input_weights_leading_dim * blockIdx.x * GEMM_X_ELEMENTS_PER_THREAD;
  const TData* gemm_input_weights_gate_1 =
      dev_input_weights + input_weights_leading_dim * (blockIdx.x * GEMM_X_ELEMENTS_PER_THREAD + 1 * hidden_size);
  const TData* gemm_input_weights_gate_2 =
      dev_input_weights + input_weights_leading_dim * (blockIdx.x * GEMM_X_ELEMENTS_PER_THREAD + 2 * hidden_size);
  const TData* gemm_input_weights_gate_3 =
      dev_input_weights + input_weights_leading_dim * (blockIdx.x * GEMM_X_ELEMENTS_PER_THREAD + 3 * hidden_size);

  Reduce_type thread_data;
  int num_features_up = ssgemv::div_up<BLOCK_THREADS>(num_features) * BLOCK_THREADS;
  for (int idx = threadIdx.x; idx < num_features_up; idx += BLOCK_THREADS) {
    // load features
    TData x = (idx < num_features) ? dev_x[idx] : TData(0);
// load four gates
#pragma unroll
    for (int jj = 0; jj < GEMM_X_ELEMENTS_PER_THREAD; jj++) {
      auto y_offset = GEMM_X_ELEMENTS_PER_THREAD * blockIdx.x + jj;
      auto yy = jj * input_weights_leading_dim;
      thread_data.val[4 * jj] =
          (y_offset < hidden_size && idx < num_features) ? (gemm_input_weights_gate_0 + yy)[idx] : TData(0);
      thread_data.val[4 * jj + 1] =
          (y_offset < hidden_size && idx < num_features) ? (gemm_input_weights_gate_1 + yy)[idx] : TData(0);
      thread_data.val[4 * jj + 2] =
          (y_offset < hidden_size && idx < num_features) ? (gemm_input_weights_gate_2 + yy)[idx] : TData(0);
      thread_data.val[4 * jj + 3] =
          (y_offset < hidden_size && idx < num_features) ? (gemm_input_weights_gate_3 + yy)[idx] : TData(0);
    }
#pragma unroll
    for (int jj = 0; jj < 4 * GEMM_X_ELEMENTS_PER_THREAD; jj++) {
      thread_data.val[jj] *= x;
    }
    Reduce_type res = BlockReduce(temp_storage).Reduce(thread_data, ReduceLayer0{});

    if (threadIdx.x == 0) {
#pragma unroll
      for (int jj = 0; jj < 4 * GEMM_X_ELEMENTS_PER_THREAD; jj++) {
        dot_prod[jj] += res.val[jj];
      }
    }
    __syncthreads();
  }

  // Elementwise
  // each warp handles a range in timesteps

  const int group_id = threadIdx.x / GEMM_X_ELEMENTS_PER_THREAD;
  const int group_lane_id = threadIdx.x % GEMM_X_ELEMENTS_PER_THREAD;

  auto const idx_unit = blockIdx.x * GEMM_X_ELEMENTS_PER_THREAD + group_lane_id;
  auto const idx_in = idx_unit;
  auto const idx_forget = idx_unit + hidden_size;
  auto const idx_c_tilde = idx_unit + 2 * hidden_size;
  auto const idx_out = idx_unit + 3 * hidden_size;

  if (idx_unit >= hidden_size) return;

  auto const bias_in = bias[idx_in];
  auto const bias_forget = bias[idx_forget];
  auto const bias_c_tilde = bias[idx_c_tilde];
  auto const bias_out = bias[idx_out];

  // Each warp handles seq_size / GEMM_X_ELEMENTS_PER_THREAD steps
  int seq_per_group = ssgemv::div_up<GEMM_X_ELEMENTS_PER_THREAD>(seq_size);

  for (int idx_step = group_id * seq_per_group; idx_step < min((group_id + 1) * seq_per_group, seq_size); idx_step++) {
    if (idx_unit >= hidden_size) continue;

    auto acc_in{bias_in};
    auto acc_forget{bias_forget};
    auto acc_c_tilde{bias_c_tilde};
    auto acc_out{bias_out};

    auto smem_x_step = dot_prod + 4 * group_lane_id;

    auto const gemm_h_step = gemm_h + idx_step * gemm_leading_dim;
    acc_in += gemm_h_step[idx_in] + smem_x_step[0];
    acc_forget += gemm_h_step[idx_forget] + smem_x_step[1];
    acc_c_tilde += gemm_h_step[idx_c_tilde] + smem_x_step[2];
    acc_out += gemm_h_step[idx_out] + smem_x_step[3];

    float const gate_in{sigma_g(acc_in)};
    float const gate_forget{sigma_g(acc_forget)};
    float const gate_c_tilde{sigma_c(acc_c_tilde)};
    float const gate_out{sigma_g(acc_out)};

    auto const offset_step_idx = idx_step * hidden_leading_dim + idx_unit;
    float const cx_old{cx_in[offset_step_idx]};
    auto const cx = gate_forget * cx_old + gate_in * gate_c_tilde;
    auto const hx = static_cast<TData>(gate_out * sigma_c(cx));
    cx_out[offset_step_idx] = static_cast<TData>(cx);
    hx_out[offset_step_idx] = hx;
    if (write_y && idx_step == seq_size - 1) {
      y[idx_step * hidden_size + idx_unit] = hx;
    }
  }
}

template <typename TData>
void launch_single_step_layer_0(
    TData const* __restrict__ dev_input_weights,  // 4 * num_units x num_inputs
    int input_weights_leading_dim,
    TData const* __restrict__ dev_x,
    int num_features,
    TData* __restrict__ y,
    TData const* __restrict__ gemm_h,
    unsigned int const gemm_leading_dim,
    TData const* __restrict__ bias,
    TData const* __restrict__ cx_in,
    TData* __restrict__ cx_out,
    TData* __restrict__ hx_out,
    const unsigned int hidden_leading_dim,
    const unsigned int hidden_size,
    const unsigned int seq_size,
    bool write_y,
    cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 256;
  constexpr int ELEMENTS_PER_THREAD = 8;

  int grid = cuda::ceil_div(hidden_size, ELEMENTS_PER_THREAD);

  single_step_layer_0<TData, ELEMENTS_PER_THREAD, BLOCK_THREADS><<<grid, BLOCK_THREADS, 0, stream>>>(
      dev_input_weights,
      input_weights_leading_dim,
      dev_x,
      num_features,
      y,
      gemm_h,
      gemm_leading_dim,
      bias,
      cx_in,
      cx_out,
      hx_out,
      hidden_leading_dim,
      hidden_size,
      seq_size,
      write_y);
}

struct Eltwise_params {
  void* gmem_gemm_h;
  void* gmem_gemm_x;
  int gemm_leading_dim;
  void* gmem_bias;
  void* gmem_cx_in;
  void* gmem_cx_out;
  void* gmem_hx_out;
  void* gmem_y;
  unsigned int hidden_leading_dim;
  unsigned int hidden_size;
  unsigned int seq_size;
  bool write_y;
};

template <int BLOCK_THREADS, ssgemv::Data_kind DATA_KIND>
__launch_bounds__(BLOCK_THREADS, 1) __global__ void run_eltwise(Eltwise_params params) {
  using data_type = ssgemv::Data_type<DATA_KIND>;
  using elem4_type = typename data_type::Elem4_type;
  using elem_type = typename data_type::Elem_type;

  auto gmem_bias = reinterpret_cast<elem4_type*>(params.gmem_bias);

  auto idx_step = blockIdx.x;

  auto gmem_gemm_h = reinterpret_cast<elem4_type*>(params.gmem_gemm_h) + idx_step * params.gemm_leading_dim / 4;
  auto gmem_gemm_x = reinterpret_cast<elem4_type*>(params.gmem_gemm_x) + idx_step * params.gemm_leading_dim / 4;
  auto gmem_cx_in = reinterpret_cast<elem_type*>(params.gmem_cx_in) + idx_step * params.hidden_leading_dim;

  auto gmem_cx_out = reinterpret_cast<elem_type*>(params.gmem_cx_out) + idx_step * params.hidden_leading_dim;
  auto gmem_hx_out = reinterpret_cast<elem_type*>(params.gmem_hx_out) + idx_step * params.hidden_leading_dim;
  auto gmem_y = reinterpret_cast<elem_type*>(params.gmem_y) + idx_step * params.hidden_size;

  int idx = blockIdx.y * BLOCK_THREADS + threadIdx.x;

  if (idx >= params.hidden_size) return;

  float hx, cx;
  auto acc = gmem_bias[idx];
  acc = ssgemv::add4<DATA_KIND>(acc, gmem_gemm_h[idx]);
  acc = ssgemv::add4<DATA_KIND>(acc, gmem_gemm_x[idx]);
  ssgemv::lstm_eltwise<DATA_KIND>(hx, cx, acc, gmem_cx_in[idx]);

  auto hhx = ssgemv::from_float<DATA_KIND>(hx);
  gmem_cx_out[idx] = ssgemv::from_float<DATA_KIND>(cx);
  gmem_hx_out[idx] = hhx;

  if (params.write_y && idx_step == params.seq_size - 1) {
    gmem_y[idx] = hhx;
  }
}

template <typename TData>
void launch_eltwise(
    TData* y,
    TData* gemm_h,
    TData* gemm_x,
    unsigned int const gemm_leading_dim,
    TData* bias,
    TData* cx_in,
    TData* cx_out,
    TData* hx_out,
    unsigned int hidden_leading_dim,
    unsigned int hidden_size,
    unsigned int seq_size,
    bool write_y,
    cudaStream_t stream) {
  constexpr ssgemv::Data_kind DATA_KIND = std::is_same_v<TData, float> ? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

  Eltwise_params params = {
      .gmem_gemm_h = gemm_h,
      .gmem_gemm_x = gemm_x,
      .gemm_leading_dim = gemm_leading_dim,
      .gmem_bias = bias,
      .gmem_cx_in = cx_in,
      .gmem_cx_out = cx_out,
      .gmem_hx_out = hx_out,
      .gmem_y = y,
      .hidden_leading_dim = hidden_leading_dim,
      .hidden_size = hidden_size,
      .seq_size = seq_size,
      .write_y = write_y,
  };

  if (hidden_size <= LSTM_S_NUM_UNITS) {
    constexpr int BLOCK_THREADS = LSTM_S_NUM_UNITS;
    dim3 grid_dim = dim3(seq_size, cuda::ceil_div(hidden_size, BLOCK_THREADS));
    run_eltwise<BLOCK_THREADS, DATA_KIND><<<grid_dim, BLOCK_THREADS, 0, stream>>>(params);
  } else if (hidden_size <= LSTM_M_NUM_UNITS) {
    constexpr int BLOCK_THREADS = LSTM_M_NUM_UNITS;
    dim3 grid_dim = dim3(seq_size, cuda::ceil_div(hidden_size, BLOCK_THREADS));
    run_eltwise<BLOCK_THREADS, DATA_KIND><<<grid_dim, BLOCK_THREADS, 0, stream>>>(params);
  } else {
    constexpr int BLOCK_THREADS = LSTM_L_NUM_UNITS;
    dim3 grid_dim = dim3(seq_size, cuda::ceil_div(hidden_size, BLOCK_THREADS));
    run_eltwise<BLOCK_THREADS, DATA_KIND><<<grid_dim, BLOCK_THREADS, 0, stream>>>(params);
  }
}

template <unsigned int kMaxHiddenStateSize, typename TData>
__launch_bounds__(kMaxHiddenStateSize, 1) __global__ void rnn_lstm(
    TData* __restrict__ y,
    TData const* __restrict__ gemm_h,
    TData const* __restrict__ gemm_x,
    unsigned int const gemm_leading_dim,
    TData const* __restrict__ bias,
    TData const* __restrict__ cx_in,
    TData* __restrict__ cx_out,
    TData* __restrict__ hx_out,
    const unsigned int hidden_leading_dim,
    const unsigned int hidden_size,
    const unsigned int seq_size,
    bool const write_y) {
  assert(blockDim.x <= kMaxHiddenStateSize);
  assert(blockDim.x == hidden_size);
  auto const idx_unit = threadIdx.x;
  auto const idx_in = idx_unit;
  auto const idx_forget = idx_unit + hidden_size;
  auto const idx_c_tilde = idx_unit + 2 * hidden_size;
  auto const idx_out = idx_unit + 3 * hidden_size;

  auto const bias_in = bias[idx_in];
  auto const bias_forget = bias[idx_forget];
  auto const bias_c_tilde = bias[idx_c_tilde];
  auto const bias_out = bias[idx_out];

  for (unsigned int idx_step = blockIdx.x; idx_step < seq_size; idx_step += gridDim.x) {
    auto acc_in{bias_in};
    auto acc_forget{bias_forget};
    auto acc_c_tilde{bias_c_tilde};
    auto acc_out{bias_out};

    auto const gemm_h_step = gemm_h + idx_step * gemm_leading_dim;
    auto const gemm_x_step = gemm_x + idx_step * gemm_leading_dim;
    acc_in += gemm_h_step[idx_in] + gemm_x_step[idx_in];
    acc_forget += gemm_h_step[idx_forget] + gemm_x_step[idx_forget];
    acc_c_tilde += gemm_h_step[idx_c_tilde] + gemm_x_step[idx_c_tilde];
    acc_out += gemm_h_step[idx_out] + gemm_x_step[idx_out];

    float const gate_in{sigma_g(acc_in)};
    float const gate_forget{sigma_g(acc_forget)};
    float const gate_c_tilde{sigma_c(acc_c_tilde)};
    float const gate_out{sigma_g(acc_out)};

    auto const offset_step_idx = idx_step * hidden_leading_dim + idx_unit;
    float const cx_old{cx_in[offset_step_idx]};
    auto const cx = gate_forget * cx_old + gate_in * gate_c_tilde;
    auto const hx = static_cast<TData>(gate_out * sigma_c(cx));
    cx_out[offset_step_idx] = static_cast<TData>(cx);
    hx_out[offset_step_idx] = hx;
    if (write_y && idx_step == seq_size - 1) {
      y[idx_step * hidden_size + idx_unit] = hx;
    }
  }
}

template <typename TData>
void launch_rnn_lstm(
    unsigned int num_units,
    unsigned int num_steps,
    TData* y,
    TData* gemm_h,
    TData* gemm_x,
    unsigned int const gemm_leading_dim,
    TData* bias,
    TData* cx_in,
    TData* cx_out,
    TData* hx_out,
    unsigned int hidden_leading_dim,
    unsigned int hidden_size,
    unsigned int seq_size,
    bool write_y,
    cudaStream_t stream) {
  dim3 const dim_block{num_units};
  dim3 const dim_grid{num_steps};

  if (num_units <= LSTM_S_NUM_UNITS) {
    rnn_lstm<LSTM_S_NUM_UNITS><<<dim_grid, dim_block, 0, stream>>>(
        y,
        gemm_h,
        gemm_x,
        gemm_leading_dim,
        bias,
        cx_in,
        cx_out,
        hx_out,
        hidden_leading_dim,
        hidden_size,
        seq_size,
        write_y);
  } else if (num_units <= LSTM_M_NUM_UNITS) {
    rnn_lstm<LSTM_M_NUM_UNITS><<<dim_grid, dim_block, 0, stream>>>(
        y,
        gemm_h,
        gemm_x,
        gemm_leading_dim,
        bias,
        cx_in,
        cx_out,
        hx_out,
        hidden_leading_dim,
        hidden_size,
        seq_size,
        write_y);
  } else if (num_units <= LSTM_L_NUM_UNITS) {
    rnn_lstm<LSTM_L_NUM_UNITS><<<dim_grid, dim_block, 0, stream>>>(
        y,
        gemm_h,
        gemm_x,
        gemm_leading_dim,
        bias,
        cx_in,
        cx_out,
        hx_out,
        hidden_leading_dim,
        hidden_size,
        seq_size,
        write_y);
  } else {
    StateCheck(false);
  }
}

template <typename TData>
nvidia::quant::lowlat_infer::cex::DevicePtr<void*> AllocateBatch(cudaStream_t stream, std::initializer_list<TData*> elems) {
  std::vector<TData*, nvidia::quant::lowlat_infer::cex::PinnedAllocator<TData*>> vec(elems);
  auto result = nvidia::quant::lowlat_infer::cex::Malloc<void*>(vec.size());
  nvidia::quant::lowlat_infer::cex::Copy(result, reinterpret_cast<void**>(vec.data()), vec.size(), stream);
  return result;
}

template <typename TData>
cudaDataType_t constexpr getCudaDataType() {
  if (std::is_same_v<TData, half>) {
    return CUDA_R_16F;
   } else {
    return CUDA_R_32F;
   }
}

template <typename TData>
cublasComputeType_t constexpr getComputeType(bool allowTf32) {
  if (std::is_same_v<TData, half>) {
    return CUBLAS_COMPUTE_16F;
  } else if (allowTf32) {
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  } else {
    return CUBLAS_COMPUTE_32F;
  }
}

}  // namespace single_step