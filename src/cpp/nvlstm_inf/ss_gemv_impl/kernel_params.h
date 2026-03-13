/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <stdint.h>

#include "kernel_traits.h"
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


#define LSTM_S_NUM_LAYERS 2
#define LSTM_S_NUM_INPUTS 128
#define LSTM_S_NUM_UNITS  96
#define LSTM_S_NUM_STEPS  64

#define LSTM_M_NUM_LAYERS 3
#define LSTM_M_NUM_INPUTS 192
#define LSTM_M_NUM_UNITS  160
#define LSTM_M_NUM_STEPS  96

#define LSTM_L_NUM_LAYERS 4
#define LSTM_L_NUM_INPUTS 512
#define LSTM_L_NUM_UNITS  736
#define LSTM_L_NUM_STEPS  128


#define GEMM_H_VECTORIZED 0

#define AS_STATES_LD_BAR 1
// AS_STATES_LD_BAR_IN_NULL_MODEL 0 - WIP: Lacks separate host signal for start/finish
#define AS_STATES_LD_BAR_IN_NULL_MODEL 1

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
  int* host_ld_complete = nullptr;
  int use_gdrcopy = 0;

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
  enum { NUM_UNITS  = NUM_UNITS_ };

  void *gmem_weights[NUM_LAYERS] = {};
  void *gmem_biases[NUM_LAYERS] = {};
  void *gmem_c_in[NUM_LAYERS] = {};
  void *gmem_c_out[NUM_LAYERS] = {};
  void *gmem_gemm_h[NUM_LAYERS] = {};
};

using Lstm_s_bp_params = Lstm_bp_params<LSTM_S_NUM_LAYERS, LSTM_S_NUM_INPUTS, LSTM_S_NUM_UNITS>;
using Lstm_m_bp_params = Lstm_bp_params<LSTM_M_NUM_LAYERS, LSTM_M_NUM_INPUTS, LSTM_M_NUM_UNITS>;

template <int L_, int NUM_INPUTS_, int NUM_UNITS_>
struct Lstm_bpt_params : public Lstm_bp_params<L_, NUM_INPUTS_, NUM_UNITS_> {

  int* gmem_ld_complete = nullptr;

  // for the custom grid barrier
  int* gmem_start = nullptr;
};

template <int L_, int NUM_INPUTS_, int NUM_UNITS_>
struct Lstm_pt_params : public Lstm_bpt_params<L_, NUM_INPUTS_, NUM_UNITS_> {
};

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

} // namespace ssgemv