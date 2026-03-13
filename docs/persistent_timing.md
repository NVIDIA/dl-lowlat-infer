# Timing of persistent kernels

## Algorithm Outline

We adopt a persistent kernels approach, launching the final timestep kernel only once during preprocessing. "Persistent" means that a kernel remains alive for the lifetime of the application. This improves performance since weights are loaded into shared memory and registers just once at kernel launch. Depending on the problem size, we use a single CUDA block for lstm_s, a thread block cluster (TBC) of 5 or 10 blocks for lstm_m depending on SM arch, and allocate nearly the entire device for lstm_l. All three implementations share the same memory layout and timing conventions. Each kernel outputs a single float that also serves as a signal to the CPU indicating computation has completed.

## Two Timing Modes

The implementation supports two modes controlled by `DEVICE_ALWAYS_SPIN` macro in kernels and `model_config.as_supported` in main.cu:

### Always-Spinning Mode (lstm_s, lstm_m)
When `as_supported=true`:
- Kernel is launched once during initialization and spins indefinitely
- Weights remain in shared memory/registers throughout benchmark lifetime
- Each iteration: CPU signals → GPU computes → GPU signals back
- No kernel launch overhead in timing measurements

### Launch-Per-Iteration Mode (lstm_l)
When `as_supported=false`:
- Persistent kernel is launched at each iteration
- Timing starts AFTER weights are loaded into shared memory/registers
- Kernel still does multiple layers in a single computation without additional weights loading/unloading.
- Required because lstm_l needs to unload the model to perform precompute
- Future releases will implement precompute more efficiently to enable always-spinning mode

## High-Level Timing Workflow

The persistent timing mechanism uses CPU/GPU atomic synchronization primitives to achieve ultra-low latency:

1. **Initialization** (in `PreProcess` method from ModelAsSingleStepGemv.cu):
   - Persistent kernel is launched (once for always-spinning, per-iteration otherwise)
   - GPU signals CPU it's ready (after weights loaded)
   - CPU spins waiting for ready signal

2. **Per-Inference** (in `LstmForward` method from ModelPersistentBase.cuh):
   - CPU starts timer
   - CPU signals GPU to load input and compute
   - No kernel launch in always-spinning mode

3. **Completion** (in `Output` method from ModelPersistentBase.cuh):
   - GPU performs LSTM forward pass
   - GPU signals completion by writing output value
   - CPU stops timer when value received

**Important note**: We start timing before signaling the GPU and stop after obtaining output, measuring true end-to-end latency including synchronization overhead.

## Synchronization Mechanism: Grace Hopper vs PCIe

The synchronization has two variants depending on system architecture.

### Grace Hopper Systems

On Grace Hopper systems with unified memory:
- CPU uses `compare_exchange_strong` to atomically read and reset signals
- GPU uses atomic exchange operations
- Functions: `gpu_cpu_sync_part_1()` and `poll_atomic()` in gemv_lstm_utils.cuh

### PCIe Systems

On PCIe systems, `compare_exchange_strong` is NOT guaranteed to work reliably. Instead:
- A `device_spin_switch` variable alternates between 0 and 1
- Signal value alternates each iteration to prevent race conditions
- Both CPU and GPU toggle the switch after each barrier
- Functions: `gpu_cpu_sync_part_1_no_aex()` and `poll_load()` in gemv_lstm_utils.cuh

System architecture is automatically detected at compile time (see kernel_params.h and util.h):
- PCIe systems (x86_64): `DISABLE_ATOMIC_EXCHANGE=1`
- Grace Hopper (aarch64): `DISABLE_ATOMIC_EXCHANGE=0`

## Device Side Signaling

On the GPU side, three synchronization primitives handle CPU-GPU coordination (all in gemv_lstm_utils.cuh):

### 1. GPU Notifies CPU Ready
`gpu_cpu_sync_part_0()` signals the CPU that the kernel is initialized and weights are loaded. Thread rank 0 writes the ready signal to host memory using atomic store or exchange.

### 2. GPU Waits for CPU Start Signal
The kernel spins waiting for CPU to signal that input is ready:
- Grace Hopper: `gpu_cpu_sync_part_1()` uses `poll_atomic()` with compare-exchange
- PCIe: `gpu_cpu_sync_part_1_no_aex()` uses `poll_load()` with alternating signal values and toggles `device_spin_switch`

See gemv_lstm_bp_impl.cu, gemv_lstm_tbc_impl.cu, and gemv_lstm_dev_impl.cu for usage.

### 3. GPU Signals Completion
`gpu_cpu_sync_part_2()` writes the output value which also serves as completion signal. Thread rank 0 atomically writes the regressed float value to host memory.

All model sizes (lstm_s, lstm_m, lstm_l) use these same primitives with different cooperative group types (block, cluster, or grid).

## Host Side Signaling

The CPU orchestrates timing using atomic spin loops:

### 1. CPU Waits for GPU Ready
In `PreProcess()` method (ModelAsSingleStepGemv.cu), CPU spins until GPU signals ready using `poll_atomic()` from gemv_lstm.h. This waits for the ready signal value, then resets it.

### 2. CPU Signals GPU to Start
In `LstmForward()` method (ModelPersistentBase.cuh), CPU signals GPU to proceed using `SignalReleaseDevice()`. This does NOT launch kernels in always-spinning mode.

On PCIe systems, the signal value is multiplied by the switch value (0 or 1), and the switch is toggled after each call to ensure alternating signal values.

### 3. CPU Waits for Output
In `Output()` method (ModelPersistentBase.cuh), CPU spins using `poll_load()` until GPU writes the result value. The CPU copies the output value from pinned memory.

After output is fetched, timing is stopped in ModelInstance.cu.

## PCIe Switch Management

For PCIe systems only, the `device_spin_switch` variable alternates between 0 and 1:

**Initialization and Management** (ModelPersistentBase.cuh):
- Switch starts at 1 (`SIGNAL_SWITCH_INIT_VALUE`)
- CPU side: `DeviceSignalSwitch()` method returns reference to switch variable
- GPU side: Local `device_spin_switch` variable passed to sync functions
- Reset: Must be reset to initial value when kernels are re-launched (in `InitX()` method)

Both CPU and GPU toggle the switch after each synchronization barrier, ensuring signal values alternate between iterations and preventing race conditions.