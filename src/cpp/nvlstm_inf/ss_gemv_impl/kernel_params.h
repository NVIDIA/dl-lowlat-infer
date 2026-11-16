/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <stdint.h>

#include "fp16_reg/gemv_lstm_fp16_reg_geometry.h"
#include "kernel_traits.h"

// Model sizes and kernel configurations other than the built-in ones below: a header of #defines given at build time
// (CMake MODEL_PARAMS_HEADER), setting a model's four LSTM_<S|M|L>_NUM_* sizes together and any FP16_REG_* knob.
#ifdef MODEL_PARAMS_HEADER
#include MODEL_PARAMS_HEADER
// clang-format off
#if (defined(LSTM_S_NUM_LAYERS) + defined(LSTM_S_NUM_INPUTS) + defined(LSTM_S_NUM_UNITS) + defined(LSTM_S_NUM_STEPS)) % 4
#error "MODEL_PARAMS_HEADER: set LSTM_S_NUM_LAYERS, _INPUTS, _UNITS and _STEPS together"
#endif
#if (defined(LSTM_M_NUM_LAYERS) + defined(LSTM_M_NUM_INPUTS) + defined(LSTM_M_NUM_UNITS) + defined(LSTM_M_NUM_STEPS)) % 4
#error "MODEL_PARAMS_HEADER: set LSTM_M_NUM_LAYERS, _INPUTS, _UNITS and _STEPS together"
#endif
#if (defined(LSTM_L_NUM_LAYERS) + defined(LSTM_L_NUM_INPUTS) + defined(LSTM_L_NUM_UNITS) + defined(LSTM_L_NUM_STEPS)) % 4
#error "MODEL_PARAMS_HEADER: set LSTM_L_NUM_LAYERS, _INPUTS, _UNITS and _STEPS together"
#endif
// clang-format on
#endif

// on aarch64 we have to set it manually
#if defined(__x86_64__) && not defined(__pcie__)
#define __pcie__
#endif

// PCIE workaround
#if not defined(DISABLE_ATOMIC_EXCHANGE)

#if defined(__pcie__)
#define DISABLE_ATOMIC_EXCHANGE 1
// we choose between 0 and 1 for the initial value of the signal switch
#define SIGNAL_SWITCH_INIT_VALUE 1
#else
#define DISABLE_ATOMIC_EXCHANGE 0
#endif
#endif

namespace nvidia::quant::lowlat_infer::ssgemv {

#define DO_DEBUG 0

// disable spin loops
#define NV_PROFILE 0

// enable outputs
#define DEBUG_OUTPUTS 0

#define NUM_GATES 4

// Problem sizes (built-in defaults; MODEL_PARAMS_HEADER may set others).
#ifndef LSTM_S_NUM_LAYERS
#define LSTM_S_NUM_LAYERS 2
#define LSTM_S_NUM_INPUTS 128
#define LSTM_S_NUM_UNITS 96
#define LSTM_S_NUM_STEPS 64
#endif

#ifndef LSTM_M_NUM_LAYERS
#define LSTM_M_NUM_LAYERS 3
#define LSTM_M_NUM_INPUTS 192
#define LSTM_M_NUM_UNITS 160
#define LSTM_M_NUM_STEPS 96
#endif

#ifndef LSTM_L_NUM_LAYERS
#define LSTM_L_NUM_LAYERS 4
#define LSTM_L_NUM_INPUTS 512
#define LSTM_L_NUM_UNITS 736
#define LSTM_L_NUM_STEPS 128
#endif

#define GEMM_H_VECTORIZED 0

// Built with the FP16 register-resident kernels (fp16_reg/README.md; CMake USE_FP16_REG_KERNELS, on by default), FP16
// lstm_s and lstm_m (always spinning; lstm_m if its CTAs fit) and lstm_l (if its CTAs fit the device) run with them
// instead of the older single-block (gemv_lstm_bp_impl.cu), cluster and device-wide kernels. 0 when not defined: the
// CMake target defines it, as only it compiles the kernels' source.
#ifndef USE_FP16_REG_KERNELS
#define USE_FP16_REG_KERNELS 0
#endif

// Their configuration of lstm_s: one CTA with all units, resident (ModelAsSingleStepGemv runs no other), lanes per
// unit, all weights in registers, FP16 gate activations.
#ifndef FP16_REG_S_LANES
#define FP16_REG_S_LANES 4
#endif
/// The FP16 register-resident kernel's configuration of lstm_s (above).
inline constexpr nvidia::quant::lowlat_infer::ssgemv::Fp16_reg_kernel_config FP16_REG_KERNEL_LSTM_S{
    .tile =
        {.units_per_cta = LSTM_S_NUM_UNITS,
         .lanes_per_unit = FP16_REG_S_LANES,
         .layer0_inputs_in_smem = 0,
         .upper_inputs_in_smem = 0,
         .fast_eltwise = true},
    .cluster_ctas = 0};

// Their configuration of lstm_m: resident (ModelAsSingleStepGemv) on several CTAs, units per CTA (the last CTA possibly
// partial), lanes per unit; by default 8 CTAs of 20 units in one hardware cluster, the hidden state through distributed
// shared memory. FP16_REG_M_DSMEM=0: tagged words in global memory instead, which are not limited to one cluster and so
// allow more than 8 CTAs (fewer units each); for some larger models that is the faster configuration.
#ifndef FP16_REG_M_UNITS_PER_CTA
#define FP16_REG_M_UNITS_PER_CTA 20
#endif
#ifndef FP16_REG_M_LANES
#define FP16_REG_M_LANES 16
#endif
#ifndef FP16_REG_M_DSMEM
#define FP16_REG_M_DSMEM 1
#endif
/// The FP16 register-resident kernel's configuration of lstm_m (above).
inline constexpr nvidia::quant::lowlat_infer::ssgemv::Fp16_reg_kernel_config FP16_REG_KERNEL_LSTM_M{
    .tile =
        {.units_per_cta = FP16_REG_M_UNITS_PER_CTA,
         .lanes_per_unit = FP16_REG_M_LANES,
         .layer0_inputs_in_smem = 0,
         .upper_inputs_in_smem = 0,
         .fast_eltwise = true},
    .cluster_ctas = 0,
    .dsmem_exchange = FP16_REG_M_DSMEM != 0};

// Their configuration of lstm_l, launched per inference (ModelSingleStepGemv): CTAs of 8 units, the hidden state as
// tagged words with one copy per 4 CTAs.
#ifndef FP16_REG_L_UNITS_PER_CTA
#define FP16_REG_L_UNITS_PER_CTA 8
#endif
#ifndef FP16_REG_L_LANES
#define FP16_REG_L_LANES 32
#endif
#ifndef FP16_REG_L_LAYER0_SMEM
#define FP16_REG_L_LAYER0_SMEM 0
#endif
#ifndef FP16_REG_L_CLUSTER_CTAS
#define FP16_REG_L_CLUSTER_CTAS 4
#endif
/// The FP16 register-resident kernel's configuration of lstm_l (above).
inline constexpr nvidia::quant::lowlat_infer::ssgemv::Fp16_reg_kernel_config FP16_REG_KERNEL_LSTM_L{
    .tile =
        {.units_per_cta = FP16_REG_L_UNITS_PER_CTA,
         .lanes_per_unit = FP16_REG_L_LANES,
         .layer0_inputs_in_smem = FP16_REG_L_LAYER0_SMEM,
         .upper_inputs_in_smem = 0,
         .fast_eltwise = true},
    .cluster_ctas = FP16_REG_L_CLUSTER_CTAS};

#define AS_STATES_LD_BAR 1
// The ping-pong model's per-step host protocol: 1 = the real kernels' two round trips (the second one, which keeps
// their state loads out of the timed step, has nothing to load here); 0 = one host signal per step, the host resets
// its signal word itself before the next step.
#define AS_STATES_LD_BAR_IN_PING_PONG_MODEL 0

// constants below should not be 0s

#define FLOAT_MAX_AS_INT 0x7f7fffffU
// See CUDART_NAN_F in math_constants.h
#define FLOAT_NAN_AS_INT 0x7fffffffU
#define HOST_BAR_VAL FLOAT_NAN_AS_INT

// Float NaN, spin while
#define HOST_SPIN_1_WHILE_SIGNAL_VALUE HOST_BAR_VAL

#define DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE 0x11111111U
#define DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE 0x22222222U

// Macro condition such that we can use cudaMemset
// for setting DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE

// INT -1, spin until
#define DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE 0xFFFFFFFFU
// Float NaN, spin while
#define HOST_SPIN_0_UNTIL_SIGNAL_VALUE HOST_BAR_VAL

static_assert(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE != 0);
static_assert(DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE != 0);
static_assert(HOST_SPIN_0_UNTIL_SIGNAL_VALUE == HOST_SPIN_1_WHILE_SIGNAL_VALUE);

static_assert(DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE != DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE);

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Lstm_params_base {
  void *gmem_gemm_x = nullptr;
  // use if stage == 0
  void **host_addr = nullptr;
  void *gmem_input = nullptr;
  void *gmem_h_out = nullptr;
  void *gmem_dot_prod_weights = nullptr;
  float dot_prod_bias = 0.0f;
  // Store reduction output here as well as use for signaling
  int *host_ld_complete = nullptr;
  int use_gdrcopy = 0;
  /// gmem_input is in device memory (a staged input written through GDRCopy), not in pinned host memory
  int input_on_device = 0;

  int always_spin = 0;
  int max_smem_size = 0;
  int sm_arch = 0;
  int sm_count = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int L_, int NUM_INPUTS_, int NUM_UNITS_>
struct Lstm_bp_params : public Lstm_params_base {
  enum { NUM_LAYERS = L_ };
  enum { NUM_INPUTS = NUM_INPUTS_ };
  enum { NUM_UNITS = NUM_UNITS_ };

  void *gmem_weights[NUM_LAYERS] = {};
  void *gmem_biases[NUM_LAYERS] = {};
  void *gmem_c_in[NUM_LAYERS] = {};
  void *gmem_c_out[NUM_LAYERS] = {};
  void *gmem_gemm_h[NUM_LAYERS] = {};
  /// FP16 register-resident kernel on several CTAs: the words its CTAs exchange (fp16_reg/gemv_lstm_fp16_reg.h)
  unsigned long long *gmem_xchg = nullptr;
  /// FP16 register-resident kernel with the resident SingleStep precompute (single_step/single_step_resident.cuh): the
  /// odd steps' R * h (gmem_gemm_h holds the even steps'), and the precompute's ready word; nullptr otherwise.
  void *gmem_gemm_h_odd[NUM_LAYERS] = {};
  unsigned *gmem_states_ready = nullptr;
  /// FP16 register-resident lstm_l kernel in its step's precompute graph (USE_FP16_REG_L_GRAPH): where it writes the
  /// step's layer-0 input product W * x (bias not included; per unit its 4 gates), for the precompute after it; nullptr
  /// otherwise.
  void *gmem_x_dot = nullptr;
};

using Lstm_s_bp_params = Lstm_bp_params<LSTM_S_NUM_LAYERS, LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS>;
using Lstm_m_bp_params = Lstm_bp_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;

template <int L_, int NUM_INPUTS_, int NUM_UNITS_>
struct Lstm_bpt_params : public Lstm_bp_params<L_, NUM_INPUTS_, NUM_UNITS_> {
  int *gmem_ld_complete = nullptr;
};

template <int L_, int NUM_INPUTS_, int NUM_UNITS_>
struct Lstm_pt_params : public Lstm_bpt_params<L_, NUM_INPUTS_, NUM_UNITS_> {};

using Lstm_s_bpt_params = Lstm_bpt_params<LSTM_S_NUM_LAYERS, LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS>;
using Lstm_m_bpt_params = Lstm_bpt_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;
using Lstm_l_bpt_params = Lstm_bpt_params<LSTM_L_NUM_LAYERS, LSTM_L_NUM_INPUTS, LSTM_L_NUM_UNITS>;

using Lstm_s_tbc_params = Lstm_bp_params<LSTM_S_NUM_LAYERS, LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS>;
using Lstm_m_tbc_params = Lstm_bp_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;

using Lstm_s_tbc_t_params = Lstm_bpt_params<LSTM_S_NUM_LAYERS, LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS>;
using Lstm_m_tbc_t_params = Lstm_bpt_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;
using Lstm_l_tbc_t_params = Lstm_bpt_params<LSTM_L_NUM_LAYERS, LSTM_L_NUM_INPUTS, LSTM_L_NUM_UNITS>;

using Lstm_m_dev_t_params = Lstm_pt_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;
using Lstm_l_dev_t_params = Lstm_pt_params<LSTM_L_NUM_LAYERS, LSTM_L_NUM_INPUTS, LSTM_L_NUM_UNITS>;

}  // namespace nvidia::quant::lowlat_infer::ssgemv
