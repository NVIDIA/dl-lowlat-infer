# FP16 register-resident LSTM kernel

One kernel template for every model size: each CTA computes a compute tile for its units of every layer with its
weights on chip (registers, optionally part in shared memory), loaded at every launch from a "tile layout" the host
prepares. Each unit's four gates are computed by `lanes_per_unit` lanes, each owning one slice of the layer's inputs,
combined with XOR shuffles; the gate activations are FP16. On one CTA the hidden state goes from layer to layer through
shared memory; on several, it is exchanged as tagged 8-byte words (payload + epoch) through global memory, without
barriers, or, on at most 8 CTAs in one thread block cluster, stored into every CTA's shared memory with `st.async` (see
`docs/cta_exchange.md` for how they work and how they compare with the other kernels' exchanges).

| File | |
|---|---|
| `gemv_lstm_fp16_reg_geometry.h` | where every thread and weight sits; constexpr, shared by host and device |
| `gemv_lstm_fp16_reg.h` | host side: the configuration of a model, the weight routine into the tile layout |
| `gemv_lstm_fp16_reg_tile.cuh` | the compute tile |
| `gemv_lstm_fp16_reg_eltwise.cuh` | FP16 gate activations (`tanh.approx.f16`) |
| `gemv_lstm_tagged_word.cuh` | tagged words for the exchange between CTAs |
| `gemv_lstm_fp16_reg_impl.cu` | the kernel, its launcher and the host protocol |

With the CMake option `USE_FP16_REG_KERNELS` (on by default; `-DUSE_FP16_REG_KERNELS=OFF` keeps the other kernels),
FP16 runs (`--precision=16 --algo=SingleStepGemv`) use it instead of the older kernels of `ss_gemv_impl/`: the
single-block (`gemv_lstm_bp_impl.cu`), cluster (`gemv_lstm_tbc_impl.cu`) and device-wide (`gemv_lstm_dev_impl.cu`)
kernels:

- `lstm_s` on one CTA, resident (`ModelAsSingleStepGemv`), with the two host signals per step of the other persistent
  kernels (`AS_STATES_LD_BAR == 1`): the states are loaded between them, out of the timed step. With more than 32
  instances, where `lstm_s` is launched per inference, it keeps the older single-block kernel.
- `lstm_m` on 8 CTAs of 20 units in one cluster, resident (`ModelAsSingleStepGemv`), the hidden state through
  distributed shared memory, with the same two host signals per step: on the first, CTA 0 passes the states' release on,
  every CTA loads its states and acknowledges, and CTA 0 tells the host. The second, the go: with `--gdrcopy` on a
  staged run of a PCIe build, every CTA waits for it on the device signal word and reads the input from device memory
  itself; otherwise CTA 0 waits for it, reads the input and stores it into every other CTA's shared memory, which is
  their go (with `FP16_REG_M_DSMEM=0`, it sends the go word and the input as tagged words in global memory instead).
  Used when the CTAs of all instances fit the device (`main.cu` decides); otherwise the cluster kernel.
- `lstm_l` on several CTAs, launched per inference (`ModelSingleStepGemv`): the states are loaded before the ready
  signal. With `--gdrcopy` on a staged run (the default) of a PCIe build (`GDRCOPY_DEVICE_INPUT`), every CTA reads the
  step's input from device memory; otherwise CTA 0 reads it and passes it on.
- With the resident SingleStep precompute (a build with `USE_SINGLE_STEP_RESIDENT`, on by default there;
  `single_step/README.md`), `lstm_s` and `lstm_m` get one host signal per step instead of two: every CTA loads its
  states from the precompute's slot of the step's parity as soon as its ready word says they are published, before the
  go, and takes the go as a change of the device signal word (the switch protocol). On `lstm_m` with `--gdrcopy` and the
  input in device memory every CTA waits for the go itself; otherwise CTA 0 passes it on with the go word and the input
  as tagged words. An INIT is acknowledged by every CTA (tagged words, collected by CTA 0) before the host gives the
  next go. The default (`AUTO`) builds this on x86 when the build supports it; in a build that leaves it out
  (`-DUSE_SINGLE_STEP_RESIDENT=OFF`, or `AUTO` on other platforms or with CUDA architectures below 90), none of it is
  compiled.
- `lstm_l` in its step's precompute graph (a build with `USE_FP16_REG_L_GRAPH`, on by default there): the kernel
  launched per inference is captured at the root of each step's precompute graph (`ModelSingleStepGemv`), so a step is
  one graph launch and the precompute's first layer starts as the kernel exits, through an edge inside the graph instead
  of a second launch in the stream. That kernel, its own instance (`X_DOT`), also writes layer 0's input product W * x
  after the output (`params.gmem_x_dot`); the precompute's layer 0 reads it instead of its GEMV over the layer-0
  weights, and the precompute copies no input. `--steps-per-graph=N` (1 or even; by default 8) captures N steps per
  graph, so that the boundary between two graphs is paid once per N steps; each kernel of a graph waits for the host's
  reset of the device signal word before it signals ready (the switch protocol). The default (`AUTO`) builds this on x86
  with `USE_FP16_REG_KERNELS`; in a build that leaves it out (`-DUSE_FP16_REG_L_GRAPH=OFF`, or `AUTO` on other
  platforms), none of it is compiled, and with it and `--lstm-l-in-graph=0` the kernels run as without it.

FP32 and BF16 runs keep the other kernels. The configurations are in `kernel_params.h` (`FP16_REG_KERNEL_LSTM_S`,
`FP16_REG_KERNEL_LSTM_M`, `FP16_REG_KERNEL_LSTM_L`); a header given with `MODEL_PARAMS_HEADER` can set other model sizes
and `FP16_REG_*` knobs. For `lstm_m`, `FP16_REG_M_DSMEM=0` exchanges the hidden state as tagged words in global memory:
not limited to one cluster, it allows more than 8 CTAs (`FP16_REG_M_UNITS_PER_CTA`), which for some larger models is
faster than 8 CTAs in one cluster.

To check them, compare an FP16 run against an FP32 reference of the same inputs, as in the top-level README's "Sample
usage":

```bash
$NVLSTM_BIN_DIR/nvLstmInf --precision=32 --algo=SingleStep lstm_l $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 10 $NVLSTM_OUTPUT_DIR/ref
$NVLSTM_BIN_DIR/nvLstmInf --precision=16 --algo=SingleStepGemv lstm_l $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 5 $NVLSTM_OUTPUT_DIR
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/ref/lstm_l.1.1.npy --out=$NVLSTM_OUTPUT_DIR/lstm_l.1.1.npy
```

and the same with `lstm_m` / `lstm_s` in place of `lstm_l`. The FP16 run prints `[fp16_reg] ... dispatched` when it
uses the kernel. Where the CTAs of `lstm_m` or `lstm_l` do not fit the device, it says so and runs the cluster or
device-wide kernel instead.
