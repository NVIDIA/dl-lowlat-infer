/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// FP16 register-resident LSTM kernel (README.md): lstm_s on one CTA, resident; lstm_m on several CTAs, resident;
// lstm_l on several CTAs, launched per inference.
//
// One kernel template for all three. Each CTA computes the compute tile of gemv_lstm_fp16_reg_tile.cuh for its units of
// every layer, with its weights on chip, loaded at every launch from the tile layout the host prepares
// (gemv_lstm_fp16_reg.h). The configuration of each size (kernel_params.h) decides the rest at compile time:
//   - one CTA (lstm_s): the hidden state goes from layer to layer through shared memory;
//   - several CTAs (lstm_m, lstm_l): each layer's hidden state is exchanged as tagged words
//     (gemv_lstm_tagged_word.cuh), one copy per group of cluster_ctas CTAs, or with dsmem_exchange (lstm_m by default)
//     through distributed shared memory (all CTAs one hardware cluster); each CTA's part of the output goes to CTA 0
//     as a tagged word; a cooperative launch, as the CTAs wait for each other.
//
// The host protocol is that of the other persistent kernels (gemv_lstm_bp_impl.cu): the ready signal once the weights
// are loaded (gpu_cpu_sync_part_0), the output to the host (gpu_cpu_sync_part_2), and per step, resident, two host
// signals (AS_STATES_LD_BAR == 1): the first releases the states (with the exit and init values), the kernel loads them
// and tells the host (gpu_cpu_sync_part_0), the second starts the timed step; launched per inference, the states are
// loaded before the ready signal. The states come from the precompute, the cell state from c_in or c_out by the parity
// of the step; the input from host memory (STAGE == 0: through a pointer to the current input, STAGE == 1: directly)
// or, with --gdrcopy, from device memory. On several CTAs only CTA 0 talks to the host: it waits for the signal and
// passes it on to the other CTAs with the go word (and the input as tagged words, unless every CTA reads it from device
// memory), and it sends the output. Resident on several CTAs, CTA 0 passes the first signal on as FP16_REG_STATES,
// every CTA loads its states and acknowledges with a tagged word, CTA 0 tells the host once all have, and then passes
// the second signal on as FP16_REG_GO.

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/ptx>
#include <mutex>
#include <stdexcept>
#include <stdint.h>

#include "../dispatch_utils.cuh"
#include "../gemv_lstm.h"
#include "../gemv_lstm_utils.cuh"
#include "../kernel_params.h"
#include "gemv_lstm_fp16_reg.h"
#include "gemv_lstm_fp16_reg_tile.cuh"
#include "gemv_lstm_tagged_word.cuh"

namespace nvidia::quant::lowlat_infer::ssgemv {

/// The go word's payload (several CTAs): CTA 0's command, and FP16_REG_ODD_STEP.
enum : unsigned {
  /// Start the timed step: read the input, compute every layer, send the output part.
  FP16_REG_GO = 0,
  /// Resident with one host signal per step: the model was re-initialised; the next signal is the first step's.
  FP16_REG_INIT = 1,
  /// Resident: leave the kernel.
  FP16_REG_EXIT = 2,
  /// Resident with two host signals per step: load the step's states and acknowledge (the GO follows).
  FP16_REG_STATES = 3,
  /// With FP16_REG_STATES and the per-CTA go: the value the host's second signal will have,
  /// DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE if set, else 0.
  FP16_REG_GO_VALUE = 0x200,
  /// Set on odd steps: the cell state is double-buffered, and an odd step reads it from c_out (an even one from c_in).
  /// CTA 0 keeps the parity for all CTAs, as it resets on INIT, which the other CTAs may not see, and sends it with the
  /// command, in a bit above the commands. Always 0 launched per inference (one step per launch).
  FP16_REG_ODD_STEP = 0x100
};

/// The kernel's traits: the tile's, and the words its CTAs exchange (Fp16_reg_exchange_layout).
template <Fp16_reg_model_dims MODEL, Fp16_reg_kernel_config CONFIG>
struct Fp16_reg_kernel_traits : public Fp16_reg_tile_traits<MODEL, CONFIG.tile> {
  using Base = Fp16_reg_tile_traits<MODEL, CONFIG.tile>;  ///< the tile's traits
  /// The words the CTAs exchange.
  static __host__ __device__ constexpr Fp16_reg_exchange_layout exchange() { return {MODEL, CONFIG}; }
  /// CTAs of the kernel, and whether it is one.
  enum { GRID_SIZE = exchange().ctas(), ONE_CTA = GRID_SIZE == 1 };
  /// CTAs per copy of the hidden state and input, and copies.
  enum { CLUSTER_CTAS = exchange().cluster_ctas(), CLUSTERS = exchange().clusters() };
  /// Words of a layer's hidden values, of one CTA's, and of the input.
  enum {
    H_WORDS = exchange().h_words(),
    H_WORDS_PER_CTA = exchange().h_words_per_cta(),
    INPUT_WORDS = exchange().input_words()
  };
  /// Offsets of the exchange buffer's parts, and its words (Fp16_reg_exchange_layout).
  enum {
    XCHG_H = exchange().h_offset(),
    XCHG_DOT = exchange().dot_offset(),
    XCHG_GO = exchange().go_offset(),
    XCHG_ACK = exchange().ack_offset(),
    XCHG_INPUT = exchange().input_offset(),
    XCHG_WORDS = exchange().words()
  };
  /// Warps of CTA 0 that add up the CTAs' output parts.
  enum { PART_WARPS = cuda::ceil_div(GRID_SIZE, Base::WARP_SIZE) };
  /// Dynamic shared memory: the weights in shared memory, then the layer-0 input and two ping-pong buffers for the
  /// layers above.
  enum {
    SMEM_SIZE =
        Base::SMEM_W_BYTES + (Base::LAYER0_TRAITS::X_ELEMS + 2 * Base::LAYERS_ABOVE_TRAITS::X_ELEMS) * sizeof(uint16_t)
  };
};

/// One layer's recurrent pre-activation (4 gates) and cell state for `unit`.
template <int NUM_UNITS>
static __device__ __forceinline__ void fp16_reg_load_state(
    const void* gmem_gemm_h_v, const void* gmem_c_v, bool valid_unit, int unit, uint2& gemm_h, uint16_t& c_in) {
#if GEMM_H_VECTORIZED == 1
  gemm_h = valid_unit ? reinterpret_cast<const uint2*>(gmem_gemm_h_v)[unit] : make_uint2(0u, 0u);
#else
  // Gate-major layout: gate g of `unit` is at g * NUM_UNITS + unit.
  auto g = reinterpret_cast<const uint16_t*>(gmem_gemm_h_v);
  const uint32_t x0 = valid_unit ? g[unit] : 0u;
  const uint32_t x1 = valid_unit ? g[unit + 1 * NUM_UNITS] : 0u;
  const uint32_t x2 = valid_unit ? g[unit + 2 * NUM_UNITS] : 0u;
  const uint32_t x3 = valid_unit ? g[unit + 3 * NUM_UNITS] : 0u;
  gemm_h = make_uint2(x0 | (x1 << 16), x2 | (x3 << 16));
#endif
  c_in = valid_unit ? reinterpret_cast<const uint16_t*>(gmem_c_v)[unit] : uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// The FP16 register-resident LSTM kernel (see the top of this file): one step per launch, or resident until the host's
/// exit. MODEL / CONFIG: the model's sizes and its configuration (kernel_params.h); STAGE: the input staged in
/// params.gmem_input, else through the pointer in params.host_addr; DEVICE_ALWAYS_SPIN: resident; GDRCOPY_ENABLED:
/// the host's signal word in device memory (GDRCopy); READY: resident with the resident SingleStep precompute (its
/// ready word in params.gmem_states_ready, USE_SINGLE_STEP_RESIDENT), a separate instance so that the others compile as
/// without it; X_DOT: launched per inference in its step's precompute graph, also writes layer 0's W * x for it
/// (params.gmem_x_dot, USE_FP16_REG_L_GRAPH), a separate instance as well.
template <
    Fp16_reg_model_dims MODEL,
    Fp16_reg_kernel_config CONFIG,
    bool STAGE = false,
    bool DEVICE_ALWAYS_SPIN = 0,
    bool GDRCOPY_ENABLED = 0,
    bool READY = false,
    bool X_DOT = false>
__launch_bounds__(Fp16_reg_kernel_traits<MODEL, CONFIG>::BLOCK_THREADS, 1) __global__
    void gemv_lstm_fp16_reg_kernel(Lstm_bpt_params<MODEL.layers, MODEL.inputs, MODEL.units> params) {
  using TRAITS = Fp16_reg_kernel_traits<MODEL, CONFIG>;
  using LAYER0_TRAITS = typename TRAITS::LAYER0_TRAITS;
  using LAYERS_ABOVE_TRAITS = typename TRAITS::LAYERS_ABOVE_TRAITS;
  constexpr Data_kind DATA_KIND = TRAITS::DATA_KIND;
  using elem_type = uint16_t;  // one FP16 value

  enum { NUM_LAYERS = MODEL.layers, NUM_INPUTS = MODEL.inputs, NUM_UNITS = MODEL.units };
  enum { WARP_SIZE = TRAITS::WARP_SIZE, NUM_WARPS = TRAITS::NUM_WARPS, BLOCK_THREADS = TRAITS::BLOCK_THREADS };
  enum { UNITS_PER_CTA = TRAITS::UNITS_PER_CTA, GRID_SIZE = TRAITS::GRID_SIZE, H_WORDS = TRAITS::H_WORDS };
  enum { H_WORDS_PER_CTA = TRAITS::H_WORDS_PER_CTA, INPUT_WORDS = TRAITS::INPUT_WORDS };
  // Every CTA's units a whole number of words (word w = units 2w, 2w + 1): the received words copied as they are.
  enum { H_WORDS_PLAIN = NUM_UNITS % UNITS_PER_CTA == 0 && UNITS_PER_CTA % 2 == 0 };

  static_assert(DEBUG_OUTPUTS == 0, "DEBUG_OUTPUTS is not supported by the register-resident kernels");
  static_assert(TRAITS::ONE_CTA || H_WORDS_PER_CTA <= WARP_SIZE, "warp 0 sends the CTA's words");
  static_assert(GRID_SIZE <= BLOCK_THREADS, "CTA 0 polls one output part per thread");

  // The hidden state through distributed shared memory: all CTAs one hardware cluster (the launch's).
  constexpr bool DSMEM = CONFIG.dsmem_exchange && !TRAITS::ONE_CTA;
  static_assert(!DSMEM || GRID_SIZE <= 8, "dsmem_exchange: the CTAs form one cluster, at most 8 (portable)");

  // The input: on several CTAs CTA 0 reads it for all and broadcasts it, unless it is staged in device memory
  // (params.input_on_device: GDRCopy, the host writes it before the signal) and every CTA reads it itself once CTA 0's
  // go word arrives.
  const bool input_direct = !TRAITS::ONE_CTA && STAGE && params.input_on_device != 0;
  // Resident with two host signals per step (AS_STATES_LD_BAR == 1): the states are loaded between them, out of the
  // timed step.
  constexpr bool STATES_BEFORE_GO = DEVICE_ALWAYS_SPIN && AS_STATES_LD_BAR == 1;
  // The per-CTA go: resident on several CTAs with the input in device memory and the signal word too (GDRCopy), every
  // CTA waits for the host's second signal itself instead of for CTA 0's GO, which saves a hand-off in the timed step.
  // The word alternates between two values (DISABLE_ATOMIC_EXCHANGE == 1), so waiting does not consume it; CTA 0 sends
  // the value to wait for with FP16_REG_STATES.
#if DISABLE_ATOMIC_EXCHANGE == 1
  const bool per_cta_go = STATES_BEFORE_GO && GDRCOPY_ENABLED && input_direct;
#else
  constexpr bool per_cta_go = false;
#endif
  // Otherwise, resident in one cluster (DSMEM): CTA 0 stores the step's input into every other CTA's shared memory
  // (st.async, completing on that CTA's input barrier), which is the go too, instead of the GO word and the input as
  // tagged words in global memory: no hand-off through L2 in the timed step.
  // Resident with the resident SingleStep precompute (params.gmem_states_ready, USE_SINGLE_STEP_RESIDENT): one host
  // signal per step, taken as a change of the device signal word (the switch protocol: the word alternates and nobody
  // resets it); every CTA loads its states before it, once the precompute's ready word says they are published.
  static_assert(
      !READY || (USE_SINGLE_STEP_RESIDENT && DEVICE_ALWAYS_SPIN), "READY: resident, in a build with the option");
  constexpr bool device_ready = READY;
  static_assert(
      !X_DOT || (USE_FP16_REG_L_GRAPH && !DEVICE_ALWAYS_SPIN), "X_DOT: per inference, in a build with the option");
  // On several CTAs there with the input and the signal in device memory: every CTA waits for the host's signal and
  // reads the input itself; otherwise CTA 0 passes the go on with the go word and the input as tagged words.
  const bool per_cta_go_ready = !TRAITS::ONE_CTA && device_ready && GDRCOPY_ENABLED && input_direct;
  const bool dsmem_go = DSMEM && STATES_BEFORE_GO && !per_cta_go && !device_ready;

  // Barriers are __syncthreads(): after a spin loop, block.sync() (cooperative groups) leaves the compiler unsure that
  // the warps have converged, and it adds a divergent fallback copy of the rest of the step (twice the code).
  auto block = cooperative_groups::this_thread_block();

  // Dynamic shared memory: the weights in shared memory, then the layer-0 input and two ping-pong buffers for the
  // layers above, all slice-padded (Fp16_reg_layer_geometry::x_index).
  extern __shared__ __align__(16) unsigned char smem_storage[];
  uint2* smem_weights = reinterpret_cast<uint2*>(smem_storage);
  elem_type* smem_input = reinterpret_cast<elem_type*>(smem_storage + TRAITS::SMEM_W_BYTES);
  elem_type* smem_h[2] = {
      smem_input + LAYER0_TRAITS::X_ELEMS, smem_input + LAYER0_TRAITS::X_ELEMS + LAYERS_ABOVE_TRAITS::X_ELEMS};
  __shared__ uint2 s_part[TRAITS::LANES_PER_UNIT > WARP_SIZE ? UNITS_PER_CTA * TRAITS::WARPS_PER_UNIT : 1];
  __shared__ float s_dot[NUM_WARPS];
  __shared__ __align__(4) elem_type s_h_out[2 * H_WORDS_PER_CTA];
  // DSMEM: every CTA's words of an exchange, by its parity (a CTA can be at most one exchange ahead of the slowest,
  // since every exchange needs every CTA's words), the barriers they complete on, and each barrier's phase bit.
  __shared__ __align__(16) unsigned s_hx[2][DSMEM ? H_WORDS : 1];
  __shared__ __align__(8) uint64_t s_hbar[2];
  [[maybe_unused]] unsigned hx_phase = 0;
  // dsmem_go: the step's input as CTA 0 stores it (two values per word), the barrier it completes on, and its phase.
  __shared__ __align__(16) unsigned s_in[DSMEM ? INPUT_WORDS : 1];
  __shared__ __align__(8) uint64_t s_inbar;
  [[maybe_unused]] unsigned in_phase = 0;
  __shared__ float s_dot_parts[TRAITS::PART_WARPS == 1 ? GRID_SIZE : TRAITS::PART_WARPS];
  __shared__ uint s_exit_storage;
  __shared__ unsigned s_go, s_go_epoch;
  __shared__ elem_type* s_host_ptr;
  auto s_exit = &s_exit_storage;

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int grid_block_rank = blockIdx.x;
  const int unit0 = grid_block_rank * UNITS_PER_CTA;  // first unit of the CTA

  // The exchanged words (several CTAs); the hidden state and the input from this CTA's cluster's copy.
  unsigned long long* gmem_xchg = params.gmem_xchg;
  const int cluster = grid_block_rank / TRAITS::CLUSTER_CTAS;
  unsigned long long* gmem_h_xchg = gmem_xchg + TRAITS::XCHG_H;
  unsigned long long* gmem_h_mine = gmem_h_xchg + size_t(cluster) * 2 * H_WORDS;
  unsigned long long* gmem_dot_xchg = gmem_xchg + TRAITS::XCHG_DOT;
  unsigned long long* gmem_go = gmem_xchg + TRAITS::XCHG_GO;
  unsigned long long* gmem_ack = gmem_xchg + TRAITS::XCHG_ACK;
  unsigned long long* gmem_input_xchg = gmem_xchg + TRAITS::XCHG_INPUT;

  // Weights and biases, at every launch: this CTA's part, from the tile layout.
  Fp16_reg_tile<TRAITS> tile;
  // The last CTA may hold fewer units than the tile: its lanes past the model's last unit are not valid (their weights
  // are zero in the tile layout; their bias, state and output weight are not read).
  tile.ln.valid = tile.ln.valid && unit0 + tile.ln.unit < NUM_UNITS;
  const auto& ln = tile.ln;
  if constexpr (!TRAITS::ONE_CTA) {
    for (int ii = threadIdx.x; ii < 2 * H_WORDS_PER_CTA; ii += BLOCK_THREADS) s_h_out[ii] = elem_type(0);
  }
  {
    const uint2* g_w[NUM_LAYERS];
    const uint2* g_bias[NUM_LAYERS];
#pragma unroll
    for (int l = 0; l < NUM_LAYERS; l++) {
      g_w[l] = reinterpret_cast<const uint2*>(params.gmem_weights[l]) +
               size_t(grid_block_rank) * TRAITS::geometry().cta_weight_words(l);
      g_bias[l] = reinterpret_cast<const uint2*>(params.gmem_biases[l]) + unit0;
    }
    tile.load(g_w, g_bias, smem_weights);
  }
  const float dot_product_weight =
      ln.valid ? reinterpret_cast<const float*>(params.gmem_dot_prod_weights)[unit0 + ln.unit] : 0.0f;

  // Zero padding of the input buffers (it meets zero weights, and 0 * NaN would not be 0).
  for (int ii = threadIdx.x; ii < LAYER0_TRAITS::X_ELEMS + 2 * LAYERS_ABOVE_TRAITS::X_ELEMS; ii += BLOCK_THREADS)
    smem_input[ii] = 0;
  // The exchanged words zeroed: no epoch matches them (epochs start at 1).
  if constexpr (!TRAITS::ONE_CTA) {
    for (int ii = grid_block_rank * BLOCK_THREADS + threadIdx.x; ii < TRAITS::XCHG_WORDS;
         ii += GRID_SIZE * BLOCK_THREADS)
      gmem_xchg[ii] = 0ull;
  }
  // DSMEM: each barrier armed for one exchange's bytes (every CTA's words), before any peer may store into it.
  if constexpr (DSMEM) {
    if (threadIdx.x == 0) {
      for (int b = 0; b < 2; b++) {
        cuda::ptx::mbarrier_init(&s_hbar[b], 1);
        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &s_hbar[b], H_WORDS * 4u);
      }
      cuda::ptx::mbarrier_init(&s_inbar, 1);
      cuda::ptx::mbarrier_arrive_expect_tx(
          cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &s_inbar, INPUT_WORDS * 4u);
      cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
    }
    cooperative_groups::this_cluster().sync();
  }

  uint2 gemm_h[NUM_LAYERS];
  elem_type c_in[NUM_LAYERS];
  auto load_state = [&](int l, bool odd_step) {
    // READY: the resident precompute's odd steps' R * h slot (the cuBLAS precompute has one R * h for every step)
    void* const gemm_h_in = READY && odd_step ? params.gmem_gemm_h_odd[l] : params.gmem_gemm_h[l];
    fp16_reg_load_state<NUM_UNITS>(
        gemm_h_in,
        odd_step ? params.gmem_c_out[l] : params.gmem_c_in[l],
        ln.valid,
        unit0 + ln.unit,
        gemm_h[l],
        c_in[l]);
  };
  // Launched for one step: every layer's state before the ready signal, out of the timed step, as in the other kernels.
  // Resident with two host signals per step: every layer's between them (STATES_BEFORE_GO); resident with one, once
  // the CTA knows of the step.
  auto load_step_states = [&](bool odd_step) {
    if (DEVICE_ALWAYS_SPIN) {
#pragma unroll
      for (int l = 0; l < NUM_LAYERS; l++) load_state(l, odd_step);
    }
  };
  if (!DEVICE_ALWAYS_SPIN) {
#pragma unroll
    for (int l = 0; l < NUM_LAYERS; l++) load_state(l, false);
  }

#if DISABLE_ATOMIC_EXCHANGE == 1
  // Launched for one step: the ready signal only once the host has reset the device signal word for this launch (0),
  // which it does after it has read the last step's output, whenever the kernel starts (several steps' kernels in one
  // graph): neither is the last step's go taken for this one's, nor the last step's output overwritten.
  if (!DEVICE_ALWAYS_SPIN && grid_block_rank == 0 && threadIdx.x == 0) {
    while (*reinterpret_cast<volatile unsigned*>(params.gmem_ld_complete) != 0u) {
    }
  }
#endif

  // notify host that we are ready for compute; on several CTAs once every CTA has its weights and the exchanged words
  // are zero
  if constexpr (TRAITS::ONE_CTA) {
    gpu_cpu_sync_part_0(params.host_ld_complete, block);
  } else {
    auto grid = cooperative_groups::this_grid();
    gpu_cpu_sync_part_0(params.host_ld_complete, grid);
  }

  bool odd_step = false;
  unsigned go_epoch = 0;  // epoch of the last go word
  unsigned steps = 0;     // steps done: the epochs of the hidden-state exchanges and the output parts follow from it
#if DISABLE_ATOMIC_EXCHANGE == 1
  uint32_t device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
#endif
  // device_ready: thread 0's ready value of this step's states (the ready word at the last init, plus the steps since),
  // whether this CTA has them, and the device signal word's last value (none yet).
  unsigned ready_target = 0;
  if (device_ready && threadIdx.x == 0)
    ready_target = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*params.gmem_states_ready)
                       .load(cuda::memory_order_acquire);
  bool states_loaded = false;
  unsigned inits = 0;  // device_ready: INIT signals seen, the epochs of their acknowledgements
  unsigned last_signal = 0x5a5a5a5au;
  auto load_ready_states = [&]() {
    if (threadIdx.x == 0) {
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(*params.gmem_states_ready);
      while (static_cast<int>(ready.load(cuda::memory_order_acquire) - ready_target) < 0) {
      }
    }
    __syncthreads();
#pragma unroll
    for (int l = 0; l < NUM_LAYERS; l++) load_state(l, odd_step);
    states_loaded = true;
  };
  do {
    // device_ready: the step's states before the host's signal, out of the timed step (the host signals only once the
    // precompute has published them).
    if (device_ready && !states_loaded) load_ready_states();
    // The host's signal: CTA 0 waits for it, then passes it on with the go word (device_ready with the per-CTA go:
    // every CTA waits for it).
    unsigned command = FP16_REG_GO;
    if (TRAITS::ONE_CTA || grid_block_rank == 0 || per_cta_go_ready) {
      if (device_ready) {
        // Thread 0 waits for the device signal word to change (one compare per poll; the go alternates between two
        // values, the exit and init signals are others), and the CTA learns whether it is the go from the barrier
        // (bar.red), the command from shared memory only if not.
        unsigned special = 0;
        if (threadIdx.x == 0) {
          constexpr cuda::thread_scope SCOPE = GDRCOPY_ENABLED ? cuda::thread_scope_device : cuda::thread_scope_system;
          cuda::atomic_ref<unsigned, SCOPE> signal(*reinterpret_cast<unsigned*>(params.gmem_ld_complete));
          unsigned value;
          while ((value = signal.load(cuda::memory_order_acquire)) == last_signal) {
          }
          last_signal = value;
          if (value == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE || value == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE) {
            special = value;
            *s_exit = value;
          }
        }
        if (__syncthreads_or(special != 0)) {
          if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE) {
            command = FP16_REG_EXIT;
          } else {
            odd_step = false;
            command = FP16_REG_INIT;
          }
        }
      } else {
#if DISABLE_ATOMIC_EXCHANGE == 1
        gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(
            params.gmem_ld_complete, block, &device_spin_switch, s_exit);
#else
        gpu_cpu_sync_part_1<cooperative_groups::thread_group, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED>(
            params.gmem_ld_complete, block, s_exit);
#endif
        if (DEVICE_ALWAYS_SPIN == 1) {
          if (*s_exit == DEVICE_SPIN_UNTIL_EXIT_SIGNAL_VALUE) {
            command = FP16_REG_EXIT;
          } else if (*s_exit == DEVICE_SPIN_UNTIL_INIT_SIGNAL_VALUE) {
            odd_step = false;
#if AS_STATES_LD_BAR == 1
            // Same INIT handling as the other kernels: take the PreProcessFirstImpl kick and fall through, so that this
            // iteration is the first real step.
#if DISABLE_ATOMIC_EXCHANGE == 1
            device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
            gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(
                params.gmem_ld_complete, block, &device_spin_switch);
#else
            gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(
                params.gmem_ld_complete, block);
#endif
#else
#if DISABLE_ATOMIC_EXCHANGE == 1
            device_spin_switch = SIGNAL_SWITCH_INIT_VALUE;
#endif
            command = FP16_REG_INIT;
#endif
          }
#if AS_STATES_LD_BAR == 1
          // The signal was the states' release: load them, tell the host, and wait for its second signal, which starts
          // the timed step (as gemv_lstm_bp_impl.cu). On several CTAs every CTA loads its own: CTA 0 passes the release
          // on, and tells the host once every other CTA has acknowledged (tagged with the step).
          if (command == FP16_REG_GO) {
            if constexpr (!TRAITS::ONE_CTA) {
              const unsigned epoch = go_epoch + 1;
              unsigned go_value = 0u;
#if DISABLE_ATOMIC_EXCHANGE == 1
              go_value = device_spin_switch ? FP16_REG_GO_VALUE : 0u;  // thread 0's: the second signal's
#endif
              if (threadIdx.x == 0)
                tagged_store_release(
                    gmem_go, tagged_word(FP16_REG_STATES | go_value | (odd_step ? FP16_REG_ODD_STEP : 0u), epoch));
              go_epoch = epoch;
            }
#pragma unroll
            for (int l = 0; l < NUM_LAYERS; l++) load_state(l, odd_step);
            if constexpr (!TRAITS::ONE_CTA) {
              // acquire: each CTA's state loads before the host may overwrite the states
              if (threadIdx.x > 0 && threadIdx.x < GRID_SIZE)
                tagged_poll(gmem_ack + threadIdx.x, steps + 1, tagged_load_acquire);
            }
            gpu_cpu_sync_part_0(params.host_ld_complete, block);
#if DISABLE_ATOMIC_EXCHANGE == 1
            gpu_cpu_sync_part_1_no_aex<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(
                params.gmem_ld_complete, block, &device_spin_switch);
#else
            gpu_cpu_sync_part_1<cooperative_groups::thread_group, false, GDRCOPY_ENABLED>(
                params.gmem_ld_complete, block);
#endif
          }
#endif
        }
      }  // device_ready
      if (!TRAITS::ONE_CTA && !per_cta_go_ready) {
        // The go word, before CTA 0 reads the input, so that the other CTAs load their states while it does. Release:
        // they read the states the host's signal stands for after they see it. Not the GO with the per-CTA go or
        // dsmem_go.
        if ((!per_cta_go && !dsmem_go) || command != FP16_REG_GO) {
          const unsigned epoch = go_epoch + 1;
          if (threadIdx.x == 0)
            tagged_store_release(gmem_go, tagged_word(command | (odd_step ? FP16_REG_ODD_STEP : 0u), epoch));
          go_epoch = epoch;
        }
        // The input broadcast: CTA 0 reads the input, once for all CTAs, into every cluster's copy, as tagged words of
        // the step, or with dsmem_go into every other CTA's shared memory.
        if (!input_direct || dsmem_go) {
          if (command == FP16_REG_GO) {
            if (threadIdx.x == 0)
              s_host_ptr = STAGE ? reinterpret_cast<elem_type*>(params.gmem_input)
                                 : reinterpret_cast<elem_type**>(params.host_addr)[0];
            __syncthreads();
            // One 4-byte read per two values: from host memory each read crosses PCIe, and a thread's two 2-byte reads
            // took two round trips. The staging buffers and the rows of the input are 4-byte aligned; a pointer a
            // caller gave Input() need not be.
            const bool word_reads = NUM_INPUTS % 2 == 0 && reinterpret_cast<uintptr_t>(s_host_ptr) % 4 == 0;
            for (int ii = threadIdx.x; ii < INPUT_WORDS; ii += BLOCK_THREADS) {
              const int k = 2 * ii;
              // __ldcv: the input buffer changes every step (pinned host memory, or device memory with --gdrcopy)
              unsigned payload;
              if (word_reads) {
                payload = __ldcv(reinterpret_cast<const unsigned*>(s_host_ptr) + ii);
              } else {
                const elem_type y0 = __ldcv(&s_host_ptr[k]);
                const elem_type y1 = k + 1 >= NUM_INPUTS ? elem_type(0) : __ldcv(&s_host_ptr[k + 1]);
                payload = y0 | (unsigned(y1) << 16);
              }
              smem_input[LAYER0_TRAITS::x_index(k)] = elem_type(payload);
              if (k + 1 < NUM_INPUTS) smem_input[LAYER0_TRAITS::x_index(k + 1)] = elem_type(payload >> 16);
              if constexpr (DSMEM) {
                if (dsmem_go) {
                  auto cl = cooperative_groups::this_cluster();
                  for (int r = 1; r < GRID_SIZE; r++)
                    cuda::ptx::st_async(cl.map_shared_rank(&s_in[ii], r), payload, cl.map_shared_rank(&s_inbar, r));
                  continue;
                }
              }
              const unsigned long long word = tagged_word(payload, steps + 1);
              for (int c = 0; c < TRAITS::CLUSTERS; c++) tagged_store(gmem_input_xchg + c * INPUT_WORDS + ii, word);
            }
          }
        }
      }
    } else {
      // The other CTAs: CTA 0's go word. CTA 0 may move on past an INIT before a CTA has seen it, so the CTA takes the
      // latest epoch; the parity of the step comes with it.
      if (threadIdx.x == 0) {
        const unsigned long long word = tagged_poll_at_least(gmem_go, go_epoch + 1, tagged_load_acquire);
        s_go = unsigned(word);
        s_go_epoch = unsigned(word >> 32);
      }
      __syncthreads();
      odd_step = (s_go & FP16_REG_ODD_STEP) != 0;
      // per inference it is always GO; FP16_REG_GO_VALUE is an argument of FP16_REG_STATES, not part of the command
      if (DEVICE_ALWAYS_SPIN) command = s_go & ~unsigned(FP16_REG_ODD_STEP | FP16_REG_GO_VALUE);
      go_epoch = s_go_epoch;
      if constexpr (STATES_BEFORE_GO) {
        // The states' release: load them, acknowledge (release: after the loads), then the host's second signal
        // (per-CTA go) or CTA 0's GO.
        if (command == FP16_REG_STATES) {
#pragma unroll
          for (int l = 0; l < NUM_LAYERS; l++) load_state(l, odd_step);
          __syncthreads();
          if (threadIdx.x == 0) {
            tagged_store_release(gmem_ack + grid_block_rank, tagged_word(0u, steps + 1));
            if (per_cta_go) {
              gpu_pingpong_handles::poll_load<uint, cuda::thread_scope_device>(
                  (uint*)params.gmem_ld_complete,
                  (s_go & FP16_REG_GO_VALUE) ? uint(DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE) : 0u);
              s_go = FP16_REG_GO | (s_go & FP16_REG_ODD_STEP);
            } else if (dsmem_go) {
              // CTA 0's input here once the barrier completes; re-armed for the next step's (which CTA 0 stores only
              // after this CTA's part of this step).
              if constexpr (DSMEM) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, &s_inbar, in_phase)) {
                }
                in_phase ^= 1u;
                cuda::ptx::mbarrier_arrive_expect_tx(
                    cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &s_inbar, INPUT_WORDS * 4u);
              }
              s_go = FP16_REG_GO | (s_go & FP16_REG_ODD_STEP);
            } else {
              const unsigned long long word = tagged_poll_at_least(gmem_go, go_epoch + 1, tagged_load_acquire);
              s_go = unsigned(word);
              s_go_epoch = unsigned(word >> 32);
            }
          }
          __syncthreads();
          command = s_go & ~unsigned(FP16_REG_ODD_STEP);
          go_epoch = s_go_epoch;
        }
      }
    }

    // Every warp converged again after the polls above, and the command visibly the same in a warp (from lane 0), so
    // that the compiler needs no divergent fallback for the step's shuffles.
    __syncwarp();
    if (DEVICE_ALWAYS_SPIN) command = __shfl_sync(0xffffffffu, command, 0);

    if (command == FP16_REG_EXIT) {
      if ((TRAITS::ONE_CTA || grid_block_rank == 0) && threadIdx.x == 0) {
        gpu_pingpong_handles::signal_atomic_relaxed<uint>(
            (uint*)params.host_ld_complete, HOST_SPIN_0_UNTIL_SIGNAL_VALUE);
      }
      if constexpr (DSMEM) cooperative_groups::this_cluster().sync();  // no CTA leaves while a peer may store into it
      return;
    }
    if (command == FP16_REG_INIT) {
      // the next signal is the first step's; device_ready: the precompute published step 0 before this signal, at the
      // ready word's value now
      if (device_ready && threadIdx.x == 0)
        ready_target = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*params.gmem_states_ready)
                           .load(cuda::memory_order_acquire);
      states_loaded = false;
      if (device_ready) {
        // Acknowledged to the host (InitX waits for it before the next step's go), once every CTA has it: the go must
        // not replace the INIT in the signal word before a CTA has read it (it would keep the last run's parity and
        // ready value). On several CTAs each one acknowledges with a tagged word, CTA 0 collects them.
        inits++;
        if constexpr (!TRAITS::ONE_CTA) {
          if (threadIdx.x == 0) tagged_store_release(gmem_ack + grid_block_rank, tagged_word(0u, inits));
          if (grid_block_rank == 0 && threadIdx.x > 0 && threadIdx.x < GRID_SIZE)
            tagged_poll(gmem_ack + threadIdx.x, inits, tagged_load_acquire);
        }
        if (TRAITS::ONE_CTA || grid_block_rank == 0) gpu_cpu_sync_part_0(params.host_ld_complete, block);
      }
      continue;
    }
    if (device_ready && !states_loaded) load_ready_states();

    if constexpr (!STATES_BEFORE_GO) load_step_states(odd_step);  // resident with one host signal per step

    // Input -> slice-padded shared memory.
    {
      if constexpr (TRAITS::ONE_CTA) {
        if constexpr (STAGE == 0) {
          if (threadIdx.x == 0) s_host_ptr = reinterpret_cast<elem_type**>(params.host_addr)[0];
          __syncthreads();
          for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS)
            smem_input[LAYER0_TRAITS::x_index(ii)] = s_host_ptr[ii];
        } else {
          auto gmem_input = reinterpret_cast<const elem_type*>(params.gmem_input);
          for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS)
            smem_input[LAYER0_TRAITS::x_index(ii)] = __ldcv(&gmem_input[ii]);
        }
      } else if (dsmem_go) {
        // CTA 0's stores, here since the go (CTA 0 has the input already).
        if (grid_block_rank != 0) {
          for (int ii = threadIdx.x; ii < INPUT_WORDS; ii += BLOCK_THREADS) {
            const unsigned payload = s_in[ii];
            smem_input[LAYER0_TRAITS::x_index(2 * ii)] = elem_type(payload);
            if (2 * ii + 1 < NUM_INPUTS) smem_input[LAYER0_TRAITS::x_index(2 * ii + 1)] = elem_type(payload >> 16);
          }
        }
      } else if (input_direct) {
        auto gmem_input = reinterpret_cast<const elem_type*>(params.gmem_input);
        for (int ii = threadIdx.x; ii < NUM_INPUTS; ii += BLOCK_THREADS)
          smem_input[LAYER0_TRAITS::x_index(ii)] = __ldcv(&gmem_input[ii]);
      } else if (grid_block_rank != 0) {
        // The broadcast: CTA 0's copy, tagged with the step (CTA 0 has the input already).
        for (int ii = threadIdx.x; ii < INPUT_WORDS; ii += BLOCK_THREADS) {
          const unsigned payload = tagged_poll(gmem_input_xchg + cluster * INPUT_WORDS + ii, steps + 1);
          smem_input[LAYER0_TRAITS::x_index(2 * ii)] = elem_type(payload);
          if (2 * ii + 1 < NUM_INPUTS) smem_input[LAYER0_TRAITS::x_index(2 * ii + 1)] = elem_type(payload >> 16);
        }
        __syncwarp();  // converged after the polls (no divergent fallback)
      }
    }
    __syncthreads();

    // The layers; between two layers, the hidden state to the next layer's input.
    float hx = 0.0f, cx = 0.0f;
    uint2 x_dot = make_uint2(0u, 0u);  // X_DOT: layer 0's W * x of the unit
#pragma unroll
    for (int l = 0; l < NUM_LAYERS; l++) {
      const uint2 dot =
          tile.layer(l, l == 0 ? smem_input : smem_h[(l - 1) & 1], smem_weights, s_part, gemm_h[l], c_in[l], hx, cx);
      if (X_DOT && l == 0) x_dot = dot;
      if (l == NUM_LAYERS - 1) break;

      if constexpr (TRAITS::ONE_CTA) {
        // One CTA: the hidden values go straight into the next layer's input.
        if (ln.leader && ln.valid) smem_h[l & 1][LAYERS_ABOVE_TRAITS::x_index(ln.unit)] = from_float<DATA_KIND>(hx);
        __syncthreads();
      } else {
        // Exchange: this CTA's hidden values to every cluster's copy, then the whole layer from this cluster's copy.
        if (ln.leader && ln.valid) s_h_out[ln.unit] = from_float<DATA_KIND>(hx);
        __syncthreads();
        const unsigned epoch = steps * (NUM_LAYERS - 1) + l + 1;
        const int parity = epoch & 1;
        if (threadIdx.x < WARP_SIZE) {
          // Lanes per word, each taking every LANES_PER_WORD-th copy.
          constexpr int LANES_PER_WORD = WARP_SIZE / H_WORDS_PER_CTA;
          const int w = lane_id % H_WORDS_PER_CTA;
          if (lane_id < LANES_PER_WORD * H_WORDS_PER_CTA) {
            const unsigned payload = reinterpret_cast<const unsigned*>(s_h_out)[w];
            if constexpr (DSMEM) {
              // into every CTA's (this one's too) words of this parity, completing on that CTA's barrier
              auto cl = cooperative_groups::this_cluster();
              unsigned* const word = &s_hx[parity][grid_block_rank * H_WORDS_PER_CTA + w];
              for (int r = lane_id / H_WORDS_PER_CTA; r < GRID_SIZE; r += LANES_PER_WORD)
                cuda::ptx::st_async(cl.map_shared_rank(word, r), payload, cl.map_shared_rank(&s_hbar[parity], r));
            } else {
              for (int c = lane_id / H_WORDS_PER_CTA; c < TRAITS::CLUSTERS; c += LANES_PER_WORD)
                tagged_store(
                    gmem_h_xchg + (size_t(c) * 2 + parity) * H_WORDS + grid_block_rank * H_WORDS_PER_CTA + w,
                    tagged_word(payload, epoch));
            }
          }
        }
        if constexpr (DSMEM) {
          // Every CTA's words here once the barrier completes; re-armed for the exchange after next (the peers send
          // that one only after receiving this CTA's next words, which it sends after this point).
          while (!cuda::ptx::mbarrier_try_wait_parity(
              cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, &s_hbar[parity], (hx_phase >> parity) & 1u)) {
          }
          hx_phase ^= 1u << parity;
          if (threadIdx.x == 0)
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &s_hbar[parity], H_WORDS * 4u);
        }
        elem_type* x = smem_h[l & 1];
        for (int ii = threadIdx.x; ii < H_WORDS; ii += BLOCK_THREADS) {
          const unsigned payload = DSMEM ? s_hx[parity][ii] : tagged_poll(gmem_h_mine + parity * H_WORDS + ii, epoch);
          if constexpr (H_WORDS_PLAIN) {
            x[LAYERS_ABOVE_TRAITS::x_index(2 * ii)] = elem_type(payload);
            x[LAYERS_ABOVE_TRAITS::x_index(2 * ii + 1)] = elem_type(payload >> 16);
          } else {
            // Word j of CTA c: its units 2j and 2j + 1, unless past the CTA's or the model's units (padding).
            const int c = ii / H_WORDS_PER_CTA, j2 = 2 * (ii % H_WORDS_PER_CTA), u = c * UNITS_PER_CTA + j2;
            if (u < NUM_UNITS) x[LAYERS_ABOVE_TRAITS::x_index(u)] = elem_type(payload);
            if (j2 + 1 < UNITS_PER_CTA && u + 1 < NUM_UNITS)
              x[LAYERS_ABOVE_TRAITS::x_index(u + 1)] = elem_type(payload >> 16);
          }
        }
        __syncthreads();
      }
    }

    // Output: the dot product of the last layer's hidden state with the output weights.
    const float dot = tile.warp_dot(dot_product_weight, hx);
    if (lane_id == 0) s_dot[warp_id] = dot;
    __syncthreads();

    float dot_product = 0.0f;
    if constexpr (TRAITS::ONE_CTA) {
      if (threadIdx.x == 0) {
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) dot_product += s_dot[w];
        dot_product += params.dot_prod_bias;
      }
      gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, block);
    } else {
      // This CTA's part to CTA 0, which adds the parts up (by thread 0 up to a warp's worth, by each warp with
      // shuffles first beyond) and sends the output.
      const unsigned epoch = steps + 1;
      const int parity = epoch & 1;
      if (threadIdx.x == 0) {
        float part = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) part += s_dot[w];
        tagged_store(gmem_dot_xchg + parity * GRID_SIZE + grid_block_rank, tagged_word(__float_as_uint(part), epoch));
      }
      if (grid_block_rank == 0) {
        if constexpr (TRAITS::PART_WARPS == 1) {
          if (threadIdx.x < GRID_SIZE)
            s_dot_parts[threadIdx.x] =
                __uint_as_float(tagged_poll(gmem_dot_xchg + parity * GRID_SIZE + threadIdx.x, epoch));
        } else if (warp_id < TRAITS::PART_WARPS) {
          float v = threadIdx.x < GRID_SIZE
                        ? __uint_as_float(tagged_poll(gmem_dot_xchg + parity * GRID_SIZE + threadIdx.x, epoch))
                        : 0.0f;
#pragma unroll
          for (int off = WARP_SIZE / 2; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffffu, v, off);
          if (lane_id == 0) s_dot_parts[warp_id] = v;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          for (int c = 0; c < (TRAITS::PART_WARPS == 1 ? GRID_SIZE : TRAITS::PART_WARPS); c++)
            dot_product += s_dot_parts[c];
          dot_product += params.dot_prod_bias;
        }
        gpu_cpu_sync_part_2(params.host_ld_complete, dot_product, block);
      }
    }
    // X_DOT, after the output: layer 0's W * x for the precompute after the kernel (its input is this step's)
    if (X_DOT && ln.leader && ln.valid) reinterpret_cast<uint2*>(params.gmem_x_dot)[unit0 + ln.unit] = x_dot;
    odd_step = !odd_step;
    steps++;
    ready_target++;  // device_ready: the next step's states (thread 0)
    states_loaded = false;
  } while (DEVICE_ALWAYS_SPIN);
  if constexpr (DSMEM) cooperative_groups::this_cluster().sync();  // launched per inference: as at the exit
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Launches the kernel: one CTA with a plain launch, several with a cooperative launch (all resident at once), as one
/// cluster with dsmem_exchange.
template <
    Fp16_reg_model_dims MODEL,
    Fp16_reg_kernel_config CONFIG,
    bool STAGE,
    bool DEVICE_ALWAYS_SPIN,
    bool GDRCOPY_ENABLED,
    bool READY = false,
    bool X_DOT = false>
inline void launch_lstm_fp16_reg_t(
    Lstm_bpt_params<MODEL.layers, MODEL.inputs, MODEL.units>& params, cudaStream_t stream) {
  using TRAITS = Fp16_reg_kernel_traits<MODEL, CONFIG>;
  constexpr int GRID_SIZE = TRAITS::GRID_SIZE;
  constexpr int BLOCK_THREADS = TRAITS::BLOCK_THREADS;
  constexpr int SMEM_SIZE = TRAITS::SMEM_SIZE;
  auto kernel = gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED, READY, X_DOT>;

  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
  if constexpr (GRID_SIZE == 1) {
    kernel<<<1, BLOCK_THREADS, SMEM_SIZE, stream>>>(params);
  } else {
    // Cooperative: every CTA resident, as they wait for each other; with dsmem_exchange also one cluster.
    if constexpr (CONFIG.dsmem_exchange) {
      cudaLaunchAttribute attrs[2];
      attrs[0].id = cudaLaunchAttributeCooperative;
      attrs[0].val.cooperative = 1;
      attrs[1].id = cudaLaunchAttributeClusterDimension;
      attrs[1].val.clusterDim.x = GRID_SIZE;
      attrs[1].val.clusterDim.y = 1;
      attrs[1].val.clusterDim.z = 1;
      cudaLaunchConfig_t launch{};
      launch.gridDim = dim3(GRID_SIZE);
      launch.blockDim = dim3(BLOCK_THREADS);
      launch.dynamicSmemBytes = SMEM_SIZE;
      launch.stream = stream;
      launch.attrs = attrs;
      launch.numAttrs = 2;
      CHECK_CUDA(cudaLaunchKernelEx(&launch, kernel, params));
    } else {
      void* kernel_args[] = {(void*)&params};
      CHECK_CUDA(cudaLaunchCooperativeKernel(
          (const void*)kernel, dim3(GRID_SIZE), dim3(BLOCK_THREADS), kernel_args, SMEM_SIZE, stream));
    }
  }
  CHECK_CUDA(cudaGetLastError());
}

/// FP16 only: the host keeps these models' weights in FP16 (fp16_reg_permute_weights). STAGE / DEVICE_ALWAYS_SPIN /
/// GDRCOPY_ENABLED as for the other kernels; the configuration from kernel_params.h. SEVERAL_RESIDENT: whether a model
/// on several CTAs runs resident too (always_spin); if not, no resident instance of it is compiled.
template <bool SEVERAL_RESIDENT, typename Params>
void run_lstm_fp16_reg_t(Params& params, cudaStream_t stream) {
  constexpr Fp16_reg_model_dims MODEL{Params::NUM_INPUTS, Params::NUM_UNITS, Params::NUM_LAYERS};
  Lstm_bpt_params<MODEL.layers, MODEL.inputs, MODEL.units>& base_params = params;
  constexpr Fp16_reg_kernel_config CONFIG = fp16_reg_kernel_config(MODEL);
  static_assert(CONFIG.tile.units_per_cta > 0, "not one of this build's model sizes");
  static std::once_flag printed;
  std::call_once(printed, [&] {
    fprintf(
        stderr,
        "[fp16_reg] run_lstm_fp16_reg_t dispatched L=%d INPUTS=%d UNITS=%d: %d CTAs of %d units, %d lanes per unit\n",
        MODEL.layers,
        MODEL.inputs,
        MODEL.units,
        Fp16_reg_exchange_layout{MODEL, CONFIG}.ctas(),
        CONFIG.tile.units_per_cta,
        CONFIG.tile.lanes_per_unit);
  });
  constexpr bool ONE_CTA = Fp16_reg_kernel_traits<MODEL, CONFIG>::ONE_CTA;
  if (!ONE_CTA && !SEVERAL_RESIDENT && params.always_spin != 0)
    throw std::runtime_error("FP16 register kernel: this model's CTAs are launched per inference only");
  if (!ONE_CTA && params.gmem_xchg == nullptr)
    throw std::runtime_error("FP16 register kernel: several CTAs need params.gmem_xchg (fp16_reg_exchange_words)");
#if USE_SINGLE_STEP_RESIDENT
  if (params.gmem_states_ready != nullptr) {
    // with the resident SingleStep precompute: its own instances, resident only
    if (params.always_spin == 0)
      throw std::runtime_error("FP16 register kernel: the resident precompute needs it resident");
    kernel_dispatch::dispatch_3bools(
        params.host_addr == nullptr, true, params.use_gdrcopy != 0, [&]<bool S, bool P, bool G>() {
          if constexpr (P && (ONE_CTA || SEVERAL_RESIDENT))
            launch_lstm_fp16_reg_t<MODEL, CONFIG, S, P, G, true>(base_params, stream);
        });
    return;
  }
#endif
#if USE_FP16_REG_L_GRAPH
  if constexpr (MODEL == FP16_REG_LSTM_L_DIMS) {
    if (params.gmem_x_dot != nullptr) {
      // in its step's precompute graph (ModelSingleStepGemv, --lstm-l-in-graph): its own instances, per inference only
      if (params.always_spin != 0)
        throw std::runtime_error("FP16 register kernel: W * x is written per inference only");
      kernel_dispatch::dispatch_3bools(
          params.host_addr == nullptr, false, params.use_gdrcopy != 0, [&]<bool S, bool P, bool G>() {
            if constexpr (!P) launch_lstm_fp16_reg_t<MODEL, CONFIG, S, P, G, false, true>(base_params, stream);
          });
      return;
    }
  }
#endif
  kernel_dispatch::dispatch_3bools(
      params.host_addr == nullptr, params.always_spin != 0, params.use_gdrcopy != 0, [&]<bool S, bool P, bool G>() {
        if constexpr (ONE_CTA || SEVERAL_RESIDENT || !P)
          launch_lstm_fp16_reg_t<MODEL, CONFIG, S, P, G>(base_params, stream);
      });
}

/// lstm_s: one CTA, resident (ModelAsSingleStepGemv).
void run_lstm_s_fp16_reg_t(Lstm_s_bpt_params& params, cudaStream_t stream) {
  run_lstm_fp16_reg_t<false>(params, stream);
}

/// lstm_m: several CTAs, resident (ModelAsSingleStepGemv).
void run_lstm_m_fp16_reg_t(Lstm_m_tbc_t_params& params, cudaStream_t stream) {
  run_lstm_fp16_reg_t<true>(params, stream);
}

/// Whether num_instances of lstm_m fit the device at once in every variant the kernel may run as (gemv_lstm.h).
bool verify_lstm_m_fp16_reg_occupancy_reqs(int num_instances) {
  constexpr Fp16_reg_model_dims MODEL = FP16_REG_LSTM_M_DIMS;
  constexpr Fp16_reg_kernel_config CONFIG = fp16_reg_kernel_config(MODEL);
  using TRAITS = Fp16_reg_kernel_traits<MODEL, CONFIG>;
  int device = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
    return false;
  bool fits = true;
  constexpr bool DSMEM = CONFIG.dsmem_exchange;
  // every variant it may be launched as (resident: staged or not, GDRCopy or not)
  auto check = [&](auto kernel) {
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TRAITS::SMEM_SIZE) != cudaSuccess) {
      (void)cudaGetLastError();  // not sticky: clear it
      fits = false;
      return;
    }
    if constexpr (DSMEM) {
      // one cluster of all its CTAs per instance
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = TRAITS::GRID_SIZE;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;
      cudaLaunchConfig_t launch{};
      launch.gridDim = dim3(TRAITS::GRID_SIZE);
      launch.blockDim = dim3(TRAITS::BLOCK_THREADS);
      launch.dynamicSmemBytes = TRAITS::SMEM_SIZE;
      launch.attrs = &attr;
      launch.numAttrs = 1;
      int clusters = 0;
      if (cudaOccupancyMaxActiveClusters(&clusters, (void*)kernel, &launch) != cudaSuccess) {
        (void)cudaGetLastError();
        fits = false;
        return;
      }
      fits = fits && clusters >= num_instances;
    } else {
      int per_sm = 0;
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, TRAITS::BLOCK_THREADS, TRAITS::SMEM_SIZE) !=
          cudaSuccess) {
        (void)cudaGetLastError();
        fits = false;
        return;
      }
      fits = fits && per_sm * sms >= TRAITS::GRID_SIZE * num_instances;
    }
  };
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, true, false>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, true, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, true, false>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, true, true>);
#if USE_SINGLE_STEP_RESIDENT
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, true, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, true, true, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, true, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, true, true, true>);
#endif
  // and each instance's green-context partition (CreateGreenContexts): its CTAs rounded up to 8 SMs
  return fits && cuda::round_up(static_cast<int>(TRAITS::GRID_SIZE), 8) * num_instances <= sms;
}

/// Whether all of lstm_l's CTAs fit the device at once in every variant the kernel may run as (gemv_lstm.h).
bool verify_lstm_l_fp16_reg_occupancy_reqs() {
  constexpr Fp16_reg_model_dims MODEL = FP16_REG_LSTM_L_DIMS;
  constexpr Fp16_reg_kernel_config CONFIG = fp16_reg_kernel_config(MODEL);
  using TRAITS = Fp16_reg_kernel_traits<MODEL, CONFIG>;
  int device = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
    return false;
  bool fits = true;
  // every variant it may be launched as (launched per inference: staged or not, GDRCopy or not)
  auto check = [&](auto kernel) {
    int per_sm = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TRAITS::SMEM_SIZE) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, TRAITS::BLOCK_THREADS, TRAITS::SMEM_SIZE) !=
            cudaSuccess) {
      (void)cudaGetLastError();  // not sticky: clear it
      fits = false;
      return;
    }
    fits = fits && per_sm * sms >= TRAITS::GRID_SIZE;
  };
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, false, false>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, false, false>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, false, true>);
#if USE_FP16_REG_L_GRAPH
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, false, false, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, false, false, true, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, false, false, false, true>);
  check(gemv_lstm_fp16_reg_kernel<MODEL, CONFIG, true, false, true, false, true>);
#endif
  return fits;
}

/// lstm_l: several CTAs, launched per inference only (ModelSingleStepGemv).
void run_lstm_l_fp16_reg_t(Lstm_l_dev_t_params& params, cudaStream_t stream) {
  run_lstm_fp16_reg_t<false>(params, stream);
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv
