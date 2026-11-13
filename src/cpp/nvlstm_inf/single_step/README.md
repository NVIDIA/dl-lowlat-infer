# SingleStep

`single_step.cuh` holds the kernels of `ModelSingleStep`, the SingleStep algorithm on cuBLAS: per step, the
timestep-wide GEMMs and the LSTM cell of every open window. `gemm_algo_tune.cpp` and `lt_algo_cache.h` tune its GEMM
(README: "Tuning the precompute GEMM").

## Resident SingleStep (`SingleStepResident`)

The same algorithm as one kernel per model instance, launched once and kept for the whole run: the precompute of the
persistent FP16 `lstm_s` / `lstm_m` models, instead of `ModelSingleStep`'s graph launched every step. Built with
`USE_SINGLE_STEP_RESIDENT` (`sm_90` and newer): by default (`AUTO`) on x86 when the CUDA architectures allow it, with
`-DUSE_SINGLE_STEP_RESIDENT=ON` on other platforms. Where it is built, `nvLstmInf` runs it by default
(`--precompute-resident=0`: off; `--precompute-resident`: required), and runs the cuBLAS precompute instead where it
does not apply or its kernels do not fit (it prints which, and why).

Every input advances every open window by one step: the window that starts with this input (row 0, zero state), and
every older one (row s of the previous pass becomes row s + 1), as `ModelSingleStep` does. What a consumer needs for
the last step of the oldest window (the one that has seen num_steps - 1 inputs) is published per layer: the recurrent
term R * h (4 gates, gate-major i, f, c, o) and the cell state.

| File | |
|---|---|
| `single_step_resident.cuh` | the host class `SingleStepResident`: Init, Step, the slots and the ready word |
| `single_step_resident.cu` | weights into the tile's per-CTA slabs, buffers, TMA maps, the launch, the go words |
| `single_step_resident_kernel.cuh` | the kernel |
| `single_step_resident_tile.cuh` | the tensor-core tile (TMA pipeline, ldmatrix / mma, warp tiling, weight packing, tensor maps) |

### Contract

Per model instance:

| | written by | read by |
|---|---|---|
| input slot (pinned host memory) + go words (pass number, argument) | host | CTA 0 of the kernel |
| state slot x 2: per layer R * h (4 x units) and c (units) | the kernel | the consumer of the step |
| ready word (device memory) | the kernel, release, after the slot | the consumer, acquire |
| done word (pinned host memory) | the kernel, after every pass | host (Init, Wait) |

`Init(inputs, num_steps - 1)` clears every window and runs the first inputs through, the last pass publishing the
states for step 0. `Step(k, input)` publishes the states for step k + 1: the consumer of step k waits until the ready
word is at least k + 1 and reads slot k % 2 (the parity the inference kernels already use for c).

### Kernel

One CTA per 8 units (and per step split), 256 threads, K in chunks of 64 halves through a 2-stage TMA / mbarrier
pipeline, tensor-core mma for the GEMMs, FP32 accumulation, FP16 states. Per pass: every CTA waits for the host's go
words and copies the input into its shared memory (with GDRCopy; from pinned host memory CTA 0 alone reads them and
relays them through device memory); layer 0's input weights stay in registers; per layer the recurrent term of the
previous pass's windows, then (from layer 1 on) the previous layer's h of every CTA, the input term (layer 0: one GEMV,
the same for every window), the gates and the states into the other buffer one row down; then the published states and,
from CTA 0 once every CTA's part is in, the ready and done words. No grid barriers: every CTA sets an epoch flag (the
pass number) per layer and for its part of the slot, and waits only for the flags of the data it reads; the previous
pass's flags are checked before the go. A cooperative launch, so that all CTAs are resident.

Two warp roles per CTA: eight bulk warps advance every window as above, without publishing; a ninth, critical warp
computes only the oldest window's step and the consumer's slot right after the go (gates on the tensor cores from the
resident slabs, the units' h exchanged as tagged words), so the slot and the ready word come before the bulk is
through. Weight slabs are resident in shared memory as far as they fit (layer 0's input slab first, then R, then the
other input weights). Besides the done word the host gets a published word after the ready word
(`WaitPublished()`).

The critical warp prefetches the oldest window's states before every go. `Init` while the kernel runs clears the
state buffers after that prefetch (with no inputs it runs no pass), so its next go carries a reload bit, and the
critical warp prefetches again after it.

Sizes are run-time (any number of inputs and units, up to 8 layers and 128 window steps).

### Requirements

- **The switch signal protocol** (`DISABLE_ATOMIC_EXCHANGE=1`). The inference kernel takes the host's go as a change of
  the device signal word, with one compare per poll and several CTAs reading it (the per-CTA go of `lstm_m`), so the
  host alternates between two go values and nobody resets the word. That is the PCIe default and costs nothing there.
  **On a coherent system (Grace Hopper) the switch protocol is not wanted:** its default is the device's atomic
  exchange (the default `AUTO` leaves the resident precompute out there), and a build with
  `-DUSE_SINGLE_STEP_RESIDENT=ON` has the switch protocol throughout, which costs the cuBLAS precompute path there
  too, with or without `--precompute-resident` (GH200: `lstm_m` +0.3 to +0.4 us mean and P99, `lstm_s` /
  `lstm_l` within run-to-run spread). Accepted for now (PCIe is the target); a resident path for the
  atomic exchange is on the list below.
- **SMs for every instance's kernel at once.** Each instance's kernel has `ceil(units / 8) x step_splits` CTAs, one
  per SM, and all instances share the SMs that the inference kernels' green contexts leave; `nvLstmInf` refuses a
  configuration whose kernels need as many or more (they could not all be resident, and they would wait on each other;
  one SM stays free for the buffer clears of an `Init` on running kernels).

### Not yet

- **Coherent systems (Grace Hopper): the resident path with the atomic exchange** (`DISABLE_ATOMIC_EXCHANGE=0`), so that
  `USE_SINGLE_STEP_RESIDENT` leaves the switch protocol off there: thread 0 of CTA 0 consumes the go with the exchange
  and passes it on with the go word (no per-CTA go), and the build keeps the platform's default protocol.
- The input forwarded by the inference kernel instead of read by every CTA (over PCIe without GDRCopy).
- Slots and ready word in pinned host memory, for a consumer on the host (CPU inference).
