/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/cmath>

#include "nvlstm_inf/ss_gemv_impl/gemv_lstm_utils.cuh"
#include "nvlstm_inf/ss_gemv_impl/kernel_params.h"

namespace nvidia::quant::lowlat_infer::single_step {

__forceinline__ __device__ float sigma_g(float in) { return __fdividef(1.f, (1.f + __expf(-in))); }
__forceinline__ __device__ float sigma_c(float in) { return tanhf(in); }

// Gate order of the weights, biases and GEMM outputs (i, f, c, o, read from the iofc order of the .npy files through
// ModelDef's offsets): row gate * num_units + unit.
enum Gate : int { GATE_IN = 0, GATE_FORGET = 1, GATE_C_TILDE = 2, GATE_OUT = 3 };

// Width of one vectorised load in single_step_layer_0.
constexpr int VEC_BYTES = static_cast<int>(sizeof(uint4));

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

// Layer 0 with the replicated input (every step has the same input vector): gates = W_x * x + gemm_h[step] + bias,
// then the LSTM eltwise for every step.  W_x * x is one GEMV shared by all steps.  A block owns UNITS_PER_BLOCK units,
// i.e. their 4 * UNITS_PER_BLOCK gate rows; each warp computes ROWS_PER_WARP of those rows with 16-byte loads (USE_VEC)
// and FP32 accumulation, reduced with warp shuffles, and then all threads run the eltwise over the (step, unit) pairs.
// X_DOT (USE_FP16_REG_L_GRAPH): W_x * x is computed already (x_dot, per unit its 4 gates, by the FP16 register lstm_l
// kernel before this one in the graph), and the block reads it instead of the GEMV.
template <typename TData, int UNITS_PER_BLOCK, int BLOCK_THREADS, bool USE_VEC, bool X_DOT = false>
__global__ void __launch_bounds__(BLOCK_THREADS) single_step_layer_0(
    TData const* __restrict__ dev_input_weights,  // 4 * num_units x num_inputs
    int input_weights_leading_dim,
    TData const* __restrict__ dev_x,
    int num_features,
    TData const* __restrict__ x_dot,  // X_DOT only
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
  constexpr int WARP_SIZE = 32;
  constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
  constexpr int NUM_ROWS = NUM_GATES * UNITS_PER_BLOCK;
  static_assert(NUM_ROWS % NUM_WARPS == 0, "every warp takes the same number of gate rows");
  constexpr int ROWS_PER_WARP = NUM_ROWS / NUM_WARPS;
  constexpr int ELEMS = USE_VEC ? VEC_BYTES / static_cast<int>(sizeof(TData)) : 1;

  // W_x * x + bias of the block's gate rows; local row = gate * UNITS_PER_BLOCK + unit.
  __shared__ float gate_x[NUM_ROWS];

  int const warp = threadIdx.x / WARP_SIZE;
  int const lane = threadIdx.x % WARP_SIZE;
  int const unit0 = blockIdx.x * UNITS_PER_BLOCK;

  if constexpr (X_DOT) {
    for (int local_row = threadIdx.x; local_row < NUM_ROWS; local_row += BLOCK_THREADS) {
      int const gate = local_row / UNITS_PER_BLOCK;
      int const unit = unit0 + local_row % UNITS_PER_BLOCK;
      gate_x[local_row] =
          (unit < static_cast<int>(hidden_size))
              ? static_cast<float>(x_dot[unit * NUM_GATES + gate]) + static_cast<float>(bias[gate * hidden_size + unit])
              : 0.f;
    }
  } else {
    TData const* w_row[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      int const local_row = warp * ROWS_PER_WARP + r;
      int const gate = local_row / UNITS_PER_BLOCK;
      int const unit = min(unit0 + local_row % UNITS_PER_BLOCK, static_cast<int>(hidden_size) - 1);
      w_row[r] = dev_input_weights + static_cast<std::size_t>(gate * hidden_size + unit) * input_weights_leading_dim;
      acc[r] = 0.f;
    }
    for (int k0 = lane * ELEMS; k0 < num_features; k0 += WARP_SIZE * ELEMS) {
      alignas(16) TData xv[ELEMS];  // 16-byte stores below (USE_VEC)
      alignas(16) TData wv[ROWS_PER_WARP][ELEMS];
      if (USE_VEC && k0 + ELEMS <= num_features) {
        *reinterpret_cast<uint4*>(xv) = *reinterpret_cast<uint4 const*>(dev_x + k0);
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
          *reinterpret_cast<uint4*>(wv[r]) = *reinterpret_cast<uint4 const*>(w_row[r] + k0);
        }
      } else {
        // Tail of the row (or no vector loads): element by element, nothing read past num_features.
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
          bool const in = k0 + e < num_features;
          xv[e] = in ? dev_x[k0 + e] : TData(0);
#pragma unroll
          for (int r = 0; r < ROWS_PER_WARP; ++r) {
            wv[r][e] = in ? w_row[r][k0 + e] : TData(0);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) {
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
          acc[r] += static_cast<float>(wv[r][e]) * static_cast<float>(xv[e]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
#pragma unroll
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc[r] += __shfl_xor_sync(FULL_WARP_MASK, acc[r], offset);
      }
    }
    if (lane < ROWS_PER_WARP) {
      // Lane r publishes row r (a constant-index select keeps acc[] in registers).
      float value = acc[0];
#pragma unroll
      for (int r = 1; r < ROWS_PER_WARP; ++r) {
        value = (lane == r) ? acc[r] : value;
      }
      int const local_row = warp * ROWS_PER_WARP + lane;
      int const gate = local_row / UNITS_PER_BLOCK;
      int const unit = unit0 + local_row % UNITS_PER_BLOCK;
      gate_x[local_row] =
          (unit < static_cast<int>(hidden_size)) ? value + static_cast<float>(bias[gate * hidden_size + unit]) : 0.f;
    }
  }
  __syncthreads();

  // Elementwise over (step, unit): consecutive threads take consecutive units of a step.
  for (int idx = threadIdx.x; idx < UNITS_PER_BLOCK * static_cast<int>(seq_size); idx += BLOCK_THREADS) {
    int const local_unit = idx % UNITS_PER_BLOCK;
    int const idx_step = idx / UNITS_PER_BLOCK;
    int const idx_unit = unit0 + local_unit;
    if (idx_unit >= static_cast<int>(hidden_size)) {
      continue;
    }
    TData const* gemm_h_step = gemm_h + static_cast<std::size_t>(idx_step) * gemm_leading_dim;
    auto gate_input = [&](Gate gate) {
      return static_cast<float>(gemm_h_step[gate * hidden_size + idx_unit]) +
             gate_x[gate * UNITS_PER_BLOCK + local_unit];
    };
    float const gate_in{sigma_g(gate_input(GATE_IN))};
    float const gate_forget{sigma_g(gate_input(GATE_FORGET))};
    float const gate_c_tilde{sigma_c(gate_input(GATE_C_TILDE))};
    float const gate_out{sigma_g(gate_input(GATE_OUT))};

    auto const offset_step_idx = static_cast<std::size_t>(idx_step) * hidden_leading_dim + idx_unit;
    float const cx_old{static_cast<float>(cx_in[offset_step_idx])};
    auto const cx = gate_forget * cx_old + gate_in * gate_c_tilde;
    auto const hx = static_cast<TData>(gate_out * sigma_c(cx));
    cx_out[offset_step_idx] = static_cast<TData>(cx);
    hx_out[offset_step_idx] = hx;
    if (write_y && idx_step == static_cast<int>(seq_size) - 1) {
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
    TData const* __restrict__ x_dot,  // W_x * x of the step (USE_FP16_REG_L_GRAPH), nullptr: the GEMV
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
  constexpr int UNITS_PER_BLOCK = 8;

  int const grid = cuda::ceil_div(hidden_size, UNITS_PER_BLOCK);
  // Vectorised loads need VEC_BYTES-aligned rows and input (Malloc2D pads rows to 128 bytes).
  bool const use_vectorized_loads = reinterpret_cast<std::uintptr_t>(dev_input_weights) % VEC_BYTES == 0 &&
                                    (input_weights_leading_dim * sizeof(TData)) % VEC_BYTES == 0 &&
                                    reinterpret_cast<std::uintptr_t>(dev_x) % VEC_BYTES == 0;
  auto kernel = use_vectorized_loads ? single_step_layer_0<TData, UNITS_PER_BLOCK, BLOCK_THREADS, true>
                                     : single_step_layer_0<TData, UNITS_PER_BLOCK, BLOCK_THREADS, false>;
#if USE_FP16_REG_L_GRAPH
  // from the FP16 register lstm_l kernel: FP16 only
  if constexpr (std::is_same_v<TData, half>) {
    if (x_dot != nullptr) kernel = single_step_layer_0<TData, UNITS_PER_BLOCK, BLOCK_THREADS, false, true>;
  } else {
    StateCheck(x_dot == nullptr);
  }
#else
  StateCheck(x_dot == nullptr);
#endif
  kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
      dev_input_weights,
      input_weights_leading_dim,
      dev_x,
      num_features,
      x_dot,
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
  constexpr ssgemv::Data_kind DATA_KIND =
      std::is_same_v<TData, float> ? ssgemv::DATA_TYPE_FP32 : ssgemv::DATA_TYPE_FP16;

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
    float* __restrict__ output,
    TData const* __restrict__ output_weights,
    float output_bias,
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
      using BlockReduce = cub::BlockReduce<float, kMaxHiddenStateSize>;
      __shared__ typename BlockReduce::TempStorage reduction_storage;
      auto const sum =
          BlockReduce(reduction_storage).Sum(static_cast<float>(hx) * static_cast<float>(output_weights[idx_unit]));
      if (idx_unit == 0) {
        output[0] = sum + output_bias;
      }
    }
  }
}

// rnn_lstm for FP16 with half2 accesses, without write_y: a thread takes two adjacent units of one window, a CTA
// 2 * RNN_LSTM2_THREADS units, so that a step has (units / 2 / RNN_LSTM2_THREADS) x windows CTAs instead of one per
// window. The arithmetic per element is rnn_lstm's (the same h and c). With write_y (the last layer, which also reduces
// the output over the last window's units) rnn_lstm runs: one CTA would take that window's units here.
constexpr int RNN_LSTM2_THREADS = 128;

__launch_bounds__(RNN_LSTM2_THREADS) __global__ void rnn_lstm_half2(
    __half const* __restrict__ gemm_h,
    __half const* __restrict__ gemm_x,
    unsigned int const gemm_leading_dim,
    __half const* __restrict__ bias,
    __half const* __restrict__ cx_in,
    __half* __restrict__ cx_out,
    __half* __restrict__ hx_out,
    const unsigned int hidden_leading_dim,
    const unsigned int hidden_size) {
  unsigned int const idx_step = blockIdx.y;
  unsigned int const pairs = hidden_size / 2;  // also the half2 elements between two gates' rows
  unsigned int const p = blockIdx.x * RNN_LSTM2_THREADS + threadIdx.x;
  if (p >= pairs) return;
  auto const gh = reinterpret_cast<__half2 const*>(gemm_h + idx_step * gemm_leading_dim);
  auto const gx = reinterpret_cast<__half2 const*>(gemm_x + idx_step * gemm_leading_dim);
  auto const b2 = reinterpret_cast<__half2 const*>(bias);
  auto const offset_step = idx_step * hidden_leading_dim;

  // bias + (gemm_h + gemm_x) in half, as rnn_lstm
  auto gate = [&](unsigned int g) { return __hadd2(b2[g * pairs + p], __hadd2(gh[g * pairs + p], gx[g * pairs + p])); };
  __half2 const a_in = gate(0), a_forget = gate(1), a_c_tilde = gate(2), a_out = gate(3);
  __half2 const c_old = reinterpret_cast<__half2 const*>(cx_in + offset_step)[p];
  float cx[2], hx[2];
#pragma unroll
  for (int e = 0; e < 2; e++) {
    auto lane = [&](__half2 v) { return e == 0 ? __low2float(v) : __high2float(v); };
    float const gate_in{sigma_g(lane(a_in))};
    float const gate_forget{sigma_g(lane(a_forget))};
    float const gate_c_tilde{sigma_c(lane(a_c_tilde))};
    float const gate_out{sigma_g(lane(a_out))};
    cx[e] = gate_forget * lane(c_old) + gate_in * gate_c_tilde;
    hx[e] = gate_out * sigma_c(cx[e]);
  }
  reinterpret_cast<__half2*>(cx_out + offset_step)[p] = __floats2half2_rn(cx[0], cx[1]);
  reinterpret_cast<__half2*>(hx_out + offset_step)[p] = __floats2half2_rn(hx[0], hx[1]);
}

template <typename TData>
void launch_rnn_lstm(
    unsigned int num_units,
    unsigned int num_steps,
    TData* y,
    float* output,
    TData const* output_weights,
    float output_bias,
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

  if constexpr (std::is_same_v<TData, __half>) {
    // half2 accesses: an even number of units, and the rows (and gate blocks) at 4-byte boundaries
    auto aligned = [](void const* ptr) { return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(__half2) == 0; };
    if (!write_y && hidden_size % 2 == 0 && gemm_leading_dim % 2 == 0 && hidden_leading_dim % 2 == 0 &&
        aligned(gemm_h) && aligned(gemm_x) && aligned(bias) && aligned(cx_in) && aligned(cx_out) && aligned(hx_out)) {
      dim3 const grid{cuda::ceil_div(hidden_size / 2, static_cast<unsigned int>(RNN_LSTM2_THREADS)), seq_size};
      rnn_lstm_half2<<<grid, RNN_LSTM2_THREADS, 0, stream>>>(
          gemm_h, gemm_x, gemm_leading_dim, bias, cx_in, cx_out, hx_out, hidden_leading_dim, hidden_size);
      return;
    }
  }

  if (num_units <= LSTM_S_NUM_UNITS) {
    rnn_lstm<LSTM_S_NUM_UNITS><<<dim_grid, dim_block, 0, stream>>>(
        y,
        output,
        output_weights,
        output_bias,
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
        output,
        output_weights,
        output_bias,
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
        output,
        output_weights,
        output_bias,
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
  } else if (num_units <= 1024) {
    // larger than this build's lstm_l: one thread per unit up to 1024
    rnn_lstm<1024><<<dim_grid, dim_block, 0, stream>>>(
        y,
        output,
        output_weights,
        output_bias,
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
nvidia::quant::lowlat_infer::cex::DevicePtr<void*> AllocateBatch(
    cudaStream_t stream, std::initializer_list<TData*> elems) {
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

}  // namespace nvidia::quant::lowlat_infer::single_step
