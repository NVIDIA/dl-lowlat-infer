# Inference Kernel Interface

This document describes how CUDA kernels implementing LSTM inference are integrated into the benchmark framework.

## Notation

We use the following abbreviations in this specification:

* `n`: the number of input features
* `s`: the length of the input sequence, i.e., number of time-steps per window
* `h`: the number of outputs per layer, i.e., size of the hidden states
* `l`: the number of layers in the network
* `b`: the batch size

## Inference Loop

The inference loop consists of the following steps. We describe them in pseudo Python code:

```python
def inference_loop():
    input = fetch_next_event()
    start_timing()
    copy_to_device(input);
    signal_input_ready();

    while not signal_output_ready():
        pass

    output = copy_from_device()
    stop_timing()
```

## Input / Output

New input is always provided at a fixed memory location on the device `dev_x`. The output is expected to be stored
at another fixed memory location, namely, `dev_y`. The framework assumes that `dev_x` can safely be overwritten after
`signal_output_ready()` and before `signal_input_ready()` and that `dev_y` stays valid during the same
time period. The format of `dev_x` and `dev_y` is described in the following sections.

Note that kernels should be configurable to use either `float` or `half` precision. The I/O arrays will be supplied with
the respective datatype.

### Input/Output Format

The codebase uses a single-step streaming architecture where the LSTM processes one timestep at a time. All current
models enforce sliding window mode (batch size = sequence length). Input is provided in two representations:

* **Primary input** (`dev_x_last_`): Single feature vector with `n` features. This is the most efficient representation
  used when SkipReplicate optimization is enabled (default).
* **Replicated input** (`dev_x_expanded_`): Same vector replicated `s` times in packed format. Used when algorithms need
  full sequence tensor representation for compatibility with sequence-based operations.

The inputs are already converted to the precision used for the network computations (`float` or `half`).

**Output format differs by implementation**:

* **ModelSingleStepBase implementations** (e.g., ModelSingleStep): Output (`dev_y_last_`) is an `h`-dimensional vector
  from the last network layer in `float` or `half` precision. The framework performs final aggregation separately by
  applying a scalar product and adding a fixed bias.

* **ModelPersistentBase implementations** (e.g., ModelSingleStepGemv, ModelAsSingleStepGemv): Output is always a single
  scalar value stored as `float`. The aggregation step (dot product with weights and bias addition) is fused into the
  kernel for maximum performance.

**Hidden and cell state handling**:

Internally, the kernel must keep track of `hx` and `cx` in the LSTM computations. Each of these arrays must
store `h * l` values. All values are initialized with `0` on startup. Before each inference call, the values must be
shifted up by one index within each layer.

State reset behavior differs by model:
* **Non-streaming models** (SingleStep, SingleStepGemv, etc.): The value at the first position of each layer is
  re-initialized with 0 after each sliding window shift.
* **Streaming models** (SingleStepStreaming): The value at the first position preserves the state from the previous
  iteration, maintaining continuity across sliding windows.

Optionally, the kernel might support a signal to reset all values in `hx` and `cx` to zero and reset all other internal
state.

## Class Hierarchy

Kernel implementations extend one of the following base classes:

* **ModelInterface** - Pure virtual base interface defining the contract for all models
* **ModelSingleStepBase** - Base class for single-step models providing helper methods for I/O and CUDA environment
* **ModelPersistentBase** - Base class for persistent timing models with atomic synchronization

### ModelInterface (Pure Virtual)

[`ModelInterface`](../src/cpp/nvlstm_inf/ModelInterface.h) defines the contract that all models must implement:

* `LstmForward()` - Signal input ready and start LSTM computation
* `OutputReady()` - Check if output is ready
* `OutputWait()` - Wait for output to complete
* `Aggregate()` - Perform final aggregation (can be disabled by overriding)
* `Input()` - Copy new input from host to device
* `Output()` - Copy inference results to host
* `Sliding()` - Indicates whether the model uses sliding window mode
* `IsPingPong()` - Indicates whether this is a PingPong baseline model
* `Precision()` - Returns precision (16 or 32 bits)

See ModelInterface.h for the full interface definition.

### ModelSingleStepBase (Protected Helpers)

[`ModelSingleStepBase`](../src/cpp/nvlstm_inf/ModelSingleStepBase.cuh) provides protected helper methods for derived classes:

* `Env()` - Returns [`CudaEnv`](../src/cpp/nvlstm_inf/CudaEnv.cuh) with streams, cuBLAS/cuBLASLt handles
* `DevXExpanded()` - Returns device pointer to replicated input (single timestep replicated across sequence length)
* `DevY()` - Returns device pointer to full output buffer
* `dev_x_last_` - Pointer to single input timestep (used with SkipReplicate optimization, default and more efficient)
* `dev_y_last_` - Pointer to last timestep in output buffer

**Note on input handling**: Even for single-step models, two input representations exist to allow kernels to choose
between efficiency and compatibility:
- `dev_x_last_`: Single timestep input (n features) - used when SkipReplicate=true (default, more efficient)
- `DevXExpanded()`: Same input replicated s times - used when algorithms need full sequence tensor

See ModelSingleStepBase.cuh for implementation details.

### ModelPersistentBase (Persistent Timing)

[`ModelPersistentBase`](../src/cpp/nvlstm_inf/ModelPersistentBase.cuh) provides the base for persistent timing models where
kernels are launched once during initialization and remain alive throughout the benchmark. CPU-GPU synchronization uses
atomic operations and spinning loops for minimal latency.

**Key characteristics**:
* Aggregation is fused into the kernel - the `Aggregate()` method is a no-op
* Output is always a single scalar `float` value

See the Persistent Kernel Interface section below for details on the parameter structures used by persistent models.

## Model Implementations

Current implementations:

* **ModelSingleStep** ([ModelSingleStep.cuh](../src/cpp/nvlstm_inf/ModelSingleStep.cuh)) - Default single-step implementation using cuBLAS (extends ModelSingleStepBase). Resets hidden/cell states to 0 at each sliding window shift.
* **ModelSingleStep (Streaming)** - Streaming variant of ModelSingleStep selected via `SingleStepStreaming` algorithm. Preserves hidden/cell states across sliding window shifts for continuous inference. cuBLAS-based implementation (extends ModelSingleStepBase).
* **ModelSingleStepGemv** ([ModelSingleStepGemv.cuh](../src/cpp/nvlstm_inf/ModelSingleStepGemv.cuh)) - Persistent timing using custom GEMV kernels (extends ModelPersistentBase)
* **ModelAsSingleStepGemv** ([ModelAsSingleStepGemv.cuh](../src/cpp/nvlstm_inf/ModelAsSingleStepGemv.cuh)) - Advanced persistent timing with pre-launched kernels (extends ModelPersistentBase)
* **PingPongModelPersistent** ([PingPongModelPersistent.cuh](../src/cpp/nvlstm_inf/PingPongModelPersistent.cuh)) - Baseline null model for measurements (extends ModelPersistentBase)

**Note**: Future work will add streaming support to persistent kernel implementations.

## Persistent Kernel Interface

Persistent kernels (ModelSingleStepGemv, ModelAsSingleStepGemv) use parameter structures to pass pointers to
input/output/weights/signals before kernel launch.

### Parameter Structures

Defined in [ss_gemv_impl/kernel_params.h](../src/cpp/nvlstm_inf/ss_gemv_impl/kernel_params.h):

**Base structure** (`Lstm_params_base`):
* Input/output pointers: `gmem_input`, `gmem_h_out`, `gmem_dot_prod_weights`
* Architecture properties: `max_smem_size`, `sm_arch`, `sm_count`
* Synchronization: `host_ld_complete` (CPU-visible signal)
* Configuration: `always_spin`, `use_gdrcopy`

**Extended structure** (`Lstm_bpt_params<L, NUM_INPUTS, NUM_UNITS>`):
* Per-layer arrays (size = L):
  - `gmem_weights[L]` - Weight matrices
  - `gmem_biases[L]` - Bias vectors
  - `gmem_c_in[L]` - Input cell states
  - `gmem_c_out[L]` - Output cell states
  - `gmem_gemm_h[L]` - Hidden states
* Device synchronization: `gmem_ld_complete` (GPU-visible signal)

### Model-Specific Parameter Types

* **lstm_s**: `Lstm_s_bpt_params` (2 layers, 128 inputs, 96 units)
* **lstm_m**: `Lstm_m_tbc_t_params` (3 layers, 192 inputs, 160 units)
* **lstm_l**: `Lstm_l_dev_t_params` (4 layers, 512 inputs, 736 units)

### Parameter Population

Parameters are populated in the `InitParams()` method (ModelAsSingleStepGemv.cu):

1. **Weights and biases**: Point to device memory allocated during initialization
2. **State buffers**: Point to last timestep position in state arrays (for single-step)
3. **Synchronization signals**: Connect to host/device pinned memory for atomic operations
4. **Input pointer**: Points to staged input data location

### Kernel Launch

Kernels are launched once with populated parameters via `run_lstm_s_bpt()`, `run_lstm_m_tbc_t()`, `run_lstm_l_dev_t()`
functions in [gemv_lstm.h](../src/cpp/nvlstm_inf/ss_gemv_impl/gemv_lstm.h).

See [persistent_timing.md](persistent_timing.md) for details on CPU-GPU synchronization using these signals.

## Network Specifications

The Benchmark defines the following networks:

| Model  | Features (`n`) | Timesteps (`s`) | Layers (`l`) | Units / Layer (`h`) |
|--------|----------------|-----------------|--------------|---------------------|
| lstm_s | 128            | 64              | 2            | 96                  |
| lstm_m | 192            | 96              | 3            | 160                 |
| lstm_l | 512            | 128             | 4            | 736                 |          

