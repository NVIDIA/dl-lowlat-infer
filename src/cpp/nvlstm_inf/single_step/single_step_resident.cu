#include <atomic>
#include <cstdio>
#include <cuda/cmath>
/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "single_step_resident.cuh"
#include "single_step_resident_kernel.cuh"

#if GDRCOPY_AVAILABLE
#include "../gdrcopy_util.h"
#endif

namespace nvidia::quant::lowlat_infer::single_step {

namespace {

// A device buffer of n T, owned by `owned` (freed with it), for the kernel's params.
template <typename T>
T* device_alloc(std::size_t n, std::vector<cex::DevicePtr<unsigned char>>& owned) {
  owned.push_back(cex::Malloc<unsigned char>(n * sizeof(T)));
  return reinterpret_cast<T*>(owned.back().get());
}

}  // namespace

struct SingleStepResident::Impl {
  cudaStream_t stream;  // the kernel's, occupied while it runs
  cex::StreamPtr aux;   // for clearing buffers between passes
  Rp_params p{};
  int ctas = 0;
  std::size_t smem = 0;
  std::vector<cex::DevicePtr<unsigned char>> owned;  // device allocations
  cex::PinnedPtr<unsigned> host_words;               // pinned, mapped: go seq, go arg, done, published
  cex::PinnedPtr<half> host_input_pinned;            // the input, pinned and mapped (without GDRCopy)
  std::vector<half> host_input_staging;              // the host's staging copy of the input (with GDRCopy)
  half* host_input = nullptr;                        // whichever of the two
#if GDRCOPY_AVAILABLE
  // SingleStepResidentConfig::use_gdrcopy: [go seq, go arg] at 0, the input at GDR_INPUT_OFFSET, in device memory
  std::unique_ptr<gdrcopy_utils::GdrcopyBuffer<unsigned char>> gdr;
#endif
  static constexpr std::size_t GDR_INPUT_OFFSET = 64;
  void gdr_write(std::size_t offset, void const* src, std::size_t bytes) {
#if GDRCOPY_AVAILABLE
    int const ret = gdr_copy_to_mapping(gdr->GetMemHandlePtr()->mh, gdr->GetHostPtr() + offset, src, bytes);
    if (ret != 0) throw std::runtime_error("resident precompute: gdr_copy_to_mapping failed");
#endif
  }
  bool gdr_on() const {
#if GDRCOPY_AVAILABLE
    return gdr != nullptr;
#else
    return false;
#endif
  }
  // The go words: the argument, then the pass number (after it, and after the input).
  void write_go(unsigned arg, unsigned next) {
    if (gdr_on()) {
#if GDRCOPY_AVAILABLE
      gdr_write(4, &arg, sizeof(arg));
      gdrcopy_utils::storefence();
      gdr_write(0, &next, sizeof(next));
      gdrcopy_utils::storefence();
#endif
    } else {
      cuda::atomic_ref<unsigned, cuda::thread_scope_system>(host_words[1]).store(arg, cuda::memory_order_relaxed);
      cuda::atomic_ref<unsigned, cuda::thread_scope_system>(host_words[0]).store(next, cuda::memory_order_release);
    }
  }
  // The input, from host_input to where the kernel reads it (with GDRCopy: through the mapping).
  void publish_input() {
    if (gdr_on()) gdr_write(GDR_INPUT_OFFSET, host_input, p.num_inputs * sizeof(half));
  }
  std::atomic<unsigned> seq{0};
  // The ready word only grows: Init publishes step 0's states at ready_base (odd, so that step k's slot k % 2 is the
  // kernel's (arg - 1) % 2), step k's at ready_base + k; ready_last is the last value given.
  unsigned ready_base = 0, ready_last = 0;  // the last pass given (Step on one thread, Wait possibly on another)
  bool running = false;
  bool reload =
      false;  // the next go carries GO_RELOAD: the state buffers were cleared under the critical warp's prefetch

  std::size_t act_elems() const { return static_cast<std::size_t>(p.num_layers) * p.act_rows * p.kpad_units; }
  void go(unsigned arg) {
    publish_input();
    unsigned const next = seq.load(std::memory_order_relaxed) + 1;
    if (reload) {
      arg |= GO_RELOAD;
      reload = false;
    }
    write_go(arg, next);
    seq.store(next, std::memory_order_release);
  }
  void wait(int word = 2) const {
    cuda::atomic_ref<unsigned, cuda::thread_scope_system> done(host_words[word]);
    unsigned const last = seq.load(std::memory_order_acquire);
    while (done.load(cuda::memory_order_acquire) != last) {
    }
  }
  void launch() {
    // go_relay: the other CTAs wait for gmem_go to show their pass: clear the last launch's (or its exit) first.
    CudaCheck(cudaMemsetAsync(p.gmem_go, 0, 2 * sizeof(unsigned), stream));
    p.first_pass = seq.load() + 1;
    void* args[] = {&p};
    CudaCheck(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(single_step_resident_kernel),
        dim3(p.unit_ctas, p.step_splits),
        dim3(CTA_THREADS),
        args,
        smem,
        stream));
    running = true;
  }
  void stop() {
    if (!running) return;
    wait();
    write_go(0u, GO_EXIT);
    CudaCheck(cudaStreamSynchronize(stream));
    // The host's go word back to the last pass, so that the next launch waits for seq + 1.
    write_go(0u, seq.load());
    running = false;
  }
  void set_input(float const* x) {
    for (int k = 0; k < p.num_inputs; ++k) host_input[k] = __float2half(x[k]);
  }
};

SingleStepResident::SingleStepResident(
    ModelDef const& model_def, SingleStepResidentConfig const& config, cudaStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  auto& m = *impl_;
  auto& p = m.p;
  m.stream = stream;
  m.aux = cex::CreateStream(cudaStreamNonBlocking);
  p.num_layers = static_cast<int>(model_def.NumLayers());
  p.num_units = static_cast<int>(model_def.NumUnits());
  p.num_inputs = static_cast<int>(model_def.NumFeatures());
  p.num_steps = static_cast<int>(model_def.NumSteps());
  if (p.num_layers > MAX_LAYERS || p.num_steps < 2 || p.num_steps > MAX_N_TILES * 8 ||
      cuda::round_up(p.num_inputs, K_CHUNK) > MAX_KPAD_INPUTS ||
      cuda::round_up(p.num_units, K_CHUNK) > MAX_KPAD_UNITS) {
    throw std::runtime_error(
        "resident precompute: at most 8 layers, 2 to 128 window steps, at most 1024 inputs and 1024 units");
  }
  p.num_n_tiles = cuda::ceil_div(p.num_steps, 8);
  p.step_splits = std::clamp(config.step_splits, 1, p.num_n_tiles);
  p.unit_ctas = cuda::ceil_div(p.num_units, UNITS_PER_CTA);
  p.kpad_inputs = cuda::round_up(p.num_inputs, K_CHUNK);
  p.kpad_units = cuda::round_up(p.num_units, K_CHUNK);
  p.act_rows = p.num_steps + 1;
  m.ctas = p.unit_ctas * p.step_splits;
  int const n_tiles_per_split = cuda::ceil_div(p.num_n_tiles, p.step_splits);
  int const units = p.num_units;

  // The model's weights gate-major in the gate order of ModelSingleStep (i, f, c, o), the two biases added.
  std::vector<half> wh_all, wxr_all, wx0;
  for (int layer = 0; layer < p.num_layers; ++layer) {
    int const k = (layer == 0) ? p.num_inputs : units;
    int const kpad = (layer == 0) ? p.kpad_inputs : p.kpad_units;
    auto const& W = model_def.LstmW().at(layer).Data();
    auto const& R = model_def.LstmR().at(layer).Data();
    auto const& B = model_def.LstmB().at(layer).Data();
    std::vector<half> w(static_cast<std::size_t>(4) * units * k), r(static_cast<std::size_t>(4) * units * units),
        b(static_cast<std::size_t>(4) * units);
    for (std::size_t g = 0; g < kNumGates; ++g) {
      auto const gate = kGates.at(g);
      float const* wsrc = W.data() + model_def.OffsetW(gate, layer == 0);
      float const* rsrc = R.data() + model_def.OffsetR(gate);
      float const* b1 = B.data() + model_def.OffsetB(gate, false);
      float const* b2 = B.data() + model_def.OffsetB(gate, true);
      for (int u = 0; u < units; ++u) {
        for (int i = 0; i < k; ++i)
          w[(g * units + u) * k + i] = __float2half(wsrc[static_cast<std::size_t>(u) * k + i]);
        for (int i = 0; i < units; ++i)
          r[(g * units + u) * units + i] = __float2half(rsrc[static_cast<std::size_t>(u) * units + i]);
        b[g * units + u] = __float2half(b1[u] + b2[u]);
      }
    }
    auto wx = pack_weights(w, units, k, kpad, UNITS_PER_CTA);
    auto wh = pack_weights(r, units, units, p.kpad_units, UNITS_PER_CTA);
    wh_all.insert(wh_all.end(), wh.begin(), wh.end());
    if (layer == 0) {
      wx0 = wx;
    } else {
      wxr_all.insert(wxr_all.end(), wx.begin(), wx.end());
    }
    half* bd = device_alloc<half>(b.size(), m.owned);
    CudaCheck(cudaMemcpyAsync(bd, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    p.gmem_biases[layer] = bd;
    // The copies above read host vectors that go out of scope: wait for them.
    CudaCheck(cudaStreamSynchronize(stream));
  }
  auto upload = [&](std::vector<half> const& v) {
    half* d = device_alloc<half>(std::max<std::size_t>(v.size(), 1), m.owned);
    if (!v.empty()) CudaCheck(cudaMemcpy(d, v.data(), v.size() * sizeof(half), cudaMemcpyHostToDevice));
    return d;
  };
  half* const wh_dev = upload(wh_all);
  half* const wx0_dev = upload(wx0);
  half* const wxr_dev = upload(wxr_all);
  std::size_t const wh_layer = static_cast<std::size_t>(p.unit_ctas) * UNITS_PER_CTA * 4 * p.kpad_units;
  p.gmem_weights_x0 = wx0_dev;
  p.gmem_weights_xr = wxr_dev;
  for (int layer = 0; layer < p.num_layers; ++layer) p.gmem_weights_h[layer] = wh_dev + layer * wh_layer;

  // State buffers (both zero: row 0 is never written and stays the zero state of a new window), the slots, the flags.
  for (int b = 0; b < 2; ++b) {
    p.gmem_h[b] = device_alloc<half>(m.act_elems(), m.owned);
    p.gmem_c[b] = device_alloc<half>(m.act_elems(), m.owned);
    p.gmem_slot_gemm_h[b] = device_alloc<half>(static_cast<std::size_t>(p.num_layers) * 4 * units, m.owned);
    p.gmem_slot_c[b] = device_alloc<half>(static_cast<std::size_t>(p.num_layers) * units, m.owned);
  }
  p.sync_flags = device_alloc<unsigned>(static_cast<std::size_t>(RP_FLAG_POINTS) * m.ctas, m.owned);
  p.gmem_go = device_alloc<unsigned>(2, m.owned);
  p.gmem_x = device_alloc<half>(p.kpad_inputs, m.owned);
  CudaCheck(cudaMemset(p.gmem_go, 0, 2 * sizeof(unsigned)));
  p.gmem_crit_tagged =
      device_alloc<unsigned long long>(static_cast<std::size_t>(p.num_layers) * p.kpad_units / 2, m.owned);
  CudaCheck(cudaMemset(
      p.gmem_crit_tagged, 0, static_cast<std::size_t>(p.num_layers) * p.kpad_units / 2 * sizeof(unsigned long long)));
  p.ready = device_alloc<unsigned>(1, m.owned);
  CudaCheck(cudaMemset(p.sync_flags, 0, static_cast<std::size_t>(RP_FLAG_POINTS) * m.ctas * sizeof(unsigned)));
  m.host_words = cex::MallocHost<unsigned>(4, cudaHostAllocMapped);
  m.host_input_pinned = cex::MallocHost<half>(p.kpad_inputs, cudaHostAllocMapped);
  m.host_input = m.host_input_pinned.get();
  std::fill(m.host_words.get(), m.host_words.get() + 4, 0u);
  std::fill(m.host_input, m.host_input + p.kpad_inputs, __float2half(0.f));
  unsigned* dev_words = nullptr;
  CudaCheck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev_words), m.host_words.get(), 0));
  CudaCheck(cudaHostGetDevicePointer(reinterpret_cast<void**>(const_cast<half**>(&p.host_input)), m.host_input, 0));
  p.host_go_seq = dev_words;
  p.host_go_arg = dev_words + 1;
  p.go_relay = !config.use_gdrcopy;
  if (config.use_gdrcopy) {
#if GDRCOPY_AVAILABLE
    m.gdr = std::make_unique<gdrcopy_utils::GdrcopyBuffer<unsigned char>>(
        Impl::GDR_INPUT_OFFSET + p.kpad_inputs * sizeof(half));
    auto* const dev = m.gdr->GetDevicePtr();
    p.host_go_seq = reinterpret_cast<unsigned*>(dev);
    p.host_go_arg = reinterpret_cast<unsigned*>(dev) + 1;
    p.host_input = reinterpret_cast<half*>(dev + Impl::GDR_INPUT_OFFSET);
    // the host's staging copy of the input (padding stays zero in the device buffer)
    m.host_input_pinned.reset();
    m.host_input_staging.assign(p.kpad_inputs, __float2half(0.f));
    m.host_input = m.host_input_staging.data();
#else
    throw std::runtime_error("resident precompute: built without GDRCopy");
#endif
  }
  p.host_done = dev_words + 2;
  p.host_published = dev_words + 3;

  int const tile_rows = UNITS_PER_CTA * 4;
  int const box_rows = n_tiles_per_split * 8;
  make_tensor_map(
      &p.map_wh, wh_dev, p.kpad_units, static_cast<std::uint64_t>(p.num_layers) * p.unit_ctas * tile_rows, tile_rows);
  make_tensor_map(
      &p.map_wxr,
      wxr_dev,
      p.kpad_units,
      std::max<std::uint64_t>(1, static_cast<std::uint64_t>(p.num_layers - 1) * p.unit_ctas * tile_rows),
      tile_rows);
  for (int b = 0; b < 2; ++b)
    make_tensor_map(
        &p.map_h[b], p.gmem_h[b], p.kpad_units, static_cast<std::uint64_t>(p.num_layers) * p.act_rows, box_rows);

  m.smem = fused_smem_bytes(UNITS_PER_CTA, BLOCK_THREADS, K_CHUNK, NUM_STAGES, K_GROUPS, n_tiles_per_split);
  // Resident weights after the stage buffers, as many slabs as fit: R of layer 0, 1, .., then the input weights of
  // layer 1, 2, ...
  // The critical warp's previous h after the stage buffers, then the resident weights.
  p.crit_prev_offset = static_cast<int>(m.smem - 1024);
  std::size_t const crit_prev_bytes = static_cast<std::size_t>(p.num_layers) * p.kpad_units * sizeof(half);
  p.resident_offset = cuda::round_up(p.crit_prev_offset + static_cast<int>(crit_prev_bytes), 1024);
  m.smem = static_cast<std::size_t>(p.resident_offset) + 1024;
  if (config.resident_weights) {
    int device = 0, max_shared_memory = 0;
    CudaCheck(cudaGetDevice(&device));
    CudaCheck(cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attr{};
    CudaCheck(cudaFuncGetAttributes(&attr, single_step_resident_kernel));
    std::size_t const slab = static_cast<std::size_t>(UNITS_PER_CTA) * 4 * p.kpad_units * sizeof(half);
    std::size_t const x0_slab = static_cast<std::size_t>(UNITS_PER_CTA) * 4 * p.kpad_inputs * sizeof(half);
    std::size_t used = static_cast<std::size_t>(p.resident_offset) + 1024;  // + the alignment slack
    if (used + x0_slab + attr.sharedSizeBytes <= static_cast<std::size_t>(max_shared_memory)) {
      p.resident_x0 = true;
      used += x0_slab;
    }
    auto fits = [&] { return used + slab + attr.sharedSizeBytes <= static_cast<std::size_t>(max_shared_memory); };
    for (int layer = 0; layer < p.num_layers && fits(); ++layer, used += slab) p.resident_wh |= 1u << layer;
    for (int layer = 1; layer < p.num_layers && fits(); ++layer, used += slab) p.resident_wxr |= 1u << layer;
    m.smem = used;
  }
  CudaCheck(cudaFuncSetAttribute(
      single_step_resident_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(m.smem)));

  // Zero states, slots and the ready word; the kernel starts with Start() or Init().
  for (int b = 0; b < 2; ++b) {
    CudaCheck(cudaMemsetAsync(p.gmem_h[b], 0, m.act_elems() * sizeof(half), stream));
    CudaCheck(cudaMemsetAsync(p.gmem_c[b], 0, m.act_elems() * sizeof(half), stream));
  }
  CudaCheck(cudaMemsetAsync(p.ready, 0, sizeof(unsigned), stream));
  CudaCheck(cudaStreamSynchronize(stream));
}

SingleStepResident::~SingleStepResident() {
  // Stops the kernel; the buffers and streams are freed with impl_ afterwards.
  try {
    impl_->stop();
  } catch (...) {
  }
}

void SingleStepResident::Stop() { impl_->stop(); }

void SingleStepResident::Start() {
  if (!impl_->running) impl_->launch();
}

void SingleStepResident::Init(float const* inputs, std::uint32_t rows) {
  auto& m = *impl_;
  auto& p = m.p;
  // Not started yet: run the passes with a launch of their own, and stop it after them.
  bool const own_launch = !m.running;
  if (own_launch) m.launch();
  m.wait();  // the kernel is idle between passes: the buffers can be cleared
  // The kernel occupies its stream: clear on the module's own stream.
  for (int b = 0; b < 2; ++b) {
    CudaCheck(cudaMemsetAsync(p.gmem_h[b], 0, m.act_elems() * sizeof(half), m.aux.get()));
    CudaCheck(cudaMemsetAsync(p.gmem_c[b], 0, m.act_elems() * sizeof(half), m.aux.get()));
  }
  unsigned const base = (m.ready_last + 1) | 1u;
  if (rows == 0) {
    // No inputs yet: step 0 starts from the zero state, as after ModelSingleStep's ClearHiddenState.
    CudaCheck(cudaMemsetAsync(
        p.gmem_slot_gemm_h[0],
        0,
        static_cast<std::size_t>(p.num_layers) * 4 * p.num_units * sizeof(half),
        m.aux.get()));
    CudaCheck(cudaMemsetAsync(
        p.gmem_slot_c[0], 0, static_cast<std::size_t>(p.num_layers) * p.num_units * sizeof(half), m.aux.get()));
    CudaCheck(cudaMemcpyAsync(p.ready, &base, sizeof(unsigned), cudaMemcpyHostToDevice, m.aux.get()));
  }
  CudaCheck(cudaStreamSynchronize(m.aux.get()));
  // The critical warp may have prefetched the oldest window's states before the clear (it does so before every go).
  m.reload = true;
  // A pass reads the states of the previous pass's parity: after clearing both, start the passes from either.
  for (std::uint32_t i = 0; i < rows; ++i) {
    m.set_input(inputs + static_cast<std::size_t>(i) * p.num_inputs);
    m.go(i + 1 == rows ? base : 0u);  // the last one publishes the states for step 0
    m.wait();
  }
  m.ready_base = m.ready_last = base;
  if (rows > 0) {
    // The done word does not wait for the ready word (CTA 0's critical warp stores it): read it back from the device
    // until it holds base, so that the inference kernel's read at the INIT that follows cannot see an older value.
    unsigned ready = 0;
    do {
      CudaCheck(cudaMemcpyAsync(&ready, p.ready, sizeof(unsigned), cudaMemcpyDeviceToHost, m.aux.get()));
      CudaCheck(cudaStreamSynchronize(m.aux.get()));
    } while (ready != base);
  }
  if (own_launch) m.stop();
}

void SingleStepResident::Step(std::uint32_t k, half const* input) {
  auto& m = *impl_;
  std::memcpy(m.host_input, input, m.p.num_inputs * sizeof(half));
  m.ready_last = m.ready_base + k + 1;
  m.go(m.ready_last);
}

void SingleStepResident::Step(std::uint32_t k, float const* input) {
  auto& m = *impl_;
  m.set_input(input);
  m.ready_last = m.ready_base + k + 1;
  m.go(m.ready_last);
}

void SingleStepResident::Wait() const { impl_->wait(); }
void SingleStepResident::WaitPublished() const { impl_->wait(3); }

half* SingleStepResident::GemmH(int slot, int layer) const {
  return impl_->p.gmem_slot_gemm_h[slot] + static_cast<std::size_t>(layer) * 4 * impl_->p.num_units;
}
half* SingleStepResident::CellState(int slot, int layer) const {
  return impl_->p.gmem_slot_c[slot] + static_cast<std::size_t>(layer) * impl_->p.num_units;
}
unsigned* SingleStepResident::ReadyWord() const { return impl_->p.ready; }
int SingleStepResident::Ctas() const { return impl_->ctas; }

int SingleStepResident::CtasFor(int num_units, int num_steps, int step_splits) {
  int const n_tiles =
      cuda::ceil_div(num_steps, 8);  // as the constructor: the step splits clamped to the tiles of 8 steps
  return cuda::ceil_div(num_units, UNITS_PER_CTA) * std::clamp(step_splits, 1, n_tiles);
}

}  // namespace nvidia::quant::lowlat_infer::single_step
