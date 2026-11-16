# Exchange between CTAs

When one model runs on several CTAs, every CTA computes some of a layer's units, and the next layer needs all of them:
after each layer, every CTA's hidden values have to reach every CTA. The kernels in this repository do that in these
ways.

| Kernel | Exchange | Where |
|---|---|---|
| device-wide `lstm_l` (`gemv_lstm_dev_impl.cu`) | plain stores to global memory, `grid.sync()`, every CTA reads all units | the whole grid (cooperative launch) |
| TBC `lstm_m` (`gemv_lstm_tbc_impl.cu`) | each CTA stores its values into every CTA's shared memory (DSMEM, `cluster.map_shared_rank`), then `cluster.sync()` | one thread block cluster |
| TBC `lstm_m` with `TBC_USE_MBARRIER` | the same stores as `st.async`, each completing on the receiver's mbarrier, which the receiver waits on | one thread block cluster |
| FP16 register-resident `lstm_m` (`fp16_reg/`, `USE_FP16_REG_KERNELS`; `dsmem_exchange`) | each CTA stores its values into every CTA's shared memory as `st.async`, completing on the receiver's mbarrier (one per parity of the exchange) | one thread block cluster |
| FP16 register-resident `lstm_l` (`fp16_reg/`, `USE_FP16_REG_KERNELS`) | tagged words in global memory, one copy per group of CTAs; no barrier | the whole grid (cooperative launch) |

A thread block cluster holds at most 8 CTAs portably, so DSMEM and mbarriers serve `lstm_m`; `lstm_l` needs more CTAs
than a cluster holds and exchanges through global memory. The FP16 register-resident `lstm_m` can also exchange tagged
words in global memory (`FP16_REG_M_DSMEM=0`). Tagged words are not limited to one cluster, so they allow more than 8
CTAs, which for some larger models is faster than 8 CTAs in one cluster.

## Tagged words

The FP16 register-resident kernel on several CTAs never waits on a barrier between layers; everything its CTAs pass to
each other is a tagged word (`fp16_reg/gemv_lstm_tagged_word.cuh`, the layout in `Fp16_reg_exchange_layout`):

- **Tagged word.** 8 bytes: a 32-bit payload and a 32-bit epoch, written with one relaxed store. An aligned 8-byte
  access is single-copy atomic, so a reader that sees the epoch it waits for also holds that store's payload: no
  separate flag, and no fence for the word itself.
- **Epochs.** The writer tags each word with the epoch of the exchange it belongs to (they follow from the step and the
  layer, so the CTAs share no counter); the readers poll until that epoch appears. Epochs start at 1 and the buffer is
  zeroed at every launch, so no stale word matches.
- **Two slots per word**, used by the parity of the epoch: a CTA can be at most one exchange ahead of the slowest,
  because every exchange needs every CTA's words.
- **What goes through them:** each layer's hidden state (two FP16 values per word, from every CTA to every CTA), each
  CTA's part of the output (CTA 0 adds them up and sends the result to the host), the go word (CTA 0's relay of the
  host's signal), and the input when CTA 0 reads it once for all (with `--gdrcopy` on a staged run of a PCIe build,
  every CTA reads it from device memory instead).
- **One copy per group of CTAs** (`cluster_ctas`, 4 at `lstm_l`; a software grouping, not a hardware cluster): the
  hidden state and the input are written once per group and each CTA polls its group's copy, so each word has a few
  readers instead of every CTA.
- **Acquire / release only where needed.** A tagged word orders its own 8 bytes; where it signals data elsewhere (the
  go word), the writer stores it with release and the readers load it with acquire.
- **Not a barrier.** A tagged word tells a reader that one word has arrived, nothing about the other CTAs' progress
  or about other memory: other data needs its own tag or a release / acquire pair. Reusing a word is safe without a
  barrier only because no CTA can be more than one exchange ahead (above), and zeroing the buffer at launch is
  followed by one grid-wide sync (the ready signal) before any CTA polls.

## Why not a grid barrier

A grid barrier makes every CTA wait for every other one before any of them reads. Tagged words let each CTA continue
as soon as the words it needs have arrived, but every reader polls, and many readers polling the same words cost about
as much as the barrier. Three ways to do one exchange (every CTA ends up with every CTA's words) were measured:

- **`grid.sync()`**: each CTA stores its words to one buffer in global memory, all CTAs meet at `grid.sync()`, then
  every CTA loads all words. This is the device-wide `lstm_l` kernel's scheme.
- **Tagged words, one copy**: each CTA stores its words as tagged words into one buffer; every CTA polls all words of
  that buffer until each carries the exchange's epoch. No barrier.
- **Tagged words, one copy per group of CTAs**: the CTAs form groups of C consecutive CTAs (a software grouping). Each
  CTA stores its words into every group's copy (more stores) and polls only its own group's copy (fewer readers per
  word). This is the FP16 register-resident kernel's scheme.

The exchange alone, with a standalone benchmark (not in this repository) on an RTX PRO 6000 Blackwell (188 SMs,
2272 MHz; 100,000 exchanges, median of 3 runs; each exchange also includes a block barrier and the packing and reading
of the words, about 0.03 us without any exchange), at the `lstm_l` size; mean (P99) per exchange in us:

| CTAs x units per CTA | `grid.sync()` | tagged words, one copy | tagged words, one copy per group |
|---|---|---|---|
| 92 x 8 (the FP16 register kernel's split) | 1.70 (2.33) | 1.12 (1.33) | 0.72 (0.92), C = 4 |
| 184 x 4 | 2.02 (2.96) | 2.01 (2.41) | 0.86 (1.04), C = 8 |

A single copy that every CTA polls gets slower with the number of CTAs: at 184 CTAs its mean is the same as
`grid.sync()`'s (its P99 is still lower).

What this comparison does not show:

- **It compares exchange schemes, not barriers.** `grid.sync()` is a general barrier that orders all memory between
  all CTAs; the tagged words order only the words of one exchange, and depend on its layout, the epochs and a zeroed
  buffer. Only cooperative groups' `grid.sync()` was measured, not a software grid barrier tuned for this case.
- **No compute between exchanges.** In the benchmark all CTAs reach every exchange at about the same time. In a kernel
  the CTAs arrive with some skew, and both schemes wait for the slowest CTA's words; how much of the difference
  remains depends on the kernel.
- **In a kernel it can come out the other way.** In the device-wide `lstm_l` kernel (184 CTAs, one tagged copy instead
  of `grid.sync()`), the exchange cost about 0.1 us more per exchange than `grid.sync()` with the pollers spinning,
  and 0.03 us more with a 128 ns backoff between polls (with a better P99).
