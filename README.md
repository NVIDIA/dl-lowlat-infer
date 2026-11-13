# Low Latency inference for sliding window LSTMs

Coding-agent instructions (build constraints, architecture, conventions): see [AGENTS.md](AGENTS.md).

For contribution requirements and commit conventions, see
[CONTRIBUTING.md](CONTRIBUTING.md). The formatting and static-analysis commands
are documented in [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

## Obtaining the Source Code

Note that this project stores some files in [Git Large File System](https://docs.gitlab.com/ee/topics/git/lfs/). Make sure to have a [Git LFS client](https://git-lfs.github.com/) installed when cloning this repository. On Ubuntu you can install the client by running

```shell
sudo apt-get install git-lfs
```

## Building with CMake

The project requires the following build tools:

* a recent version of `gcc` supporting C++20
* [CMake](https://cmake.org/) >= 3.20
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 13.0

Inside your code directory, configure and kick off a build as follows:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ARCH=native -DCMAKE_CUDA_ARCHITECTURES="120a-real" -B build
cmake --build build --parallel
```

### Build Options

Key CMake options:

* `CMAKE_BUILD_TYPE`: Build type (Release or Debug)
* `CPU_ARCH`: Target CPU architecture (default: `icelake-server`, use `native` for current CPU)
* `CMAKE_CUDA_ARCHITECTURES`: Target CUDA architecture (e.g., `"90a-real"`, `"100a-real"`, `"120a-real"`)
* `CMAKE_INSTALL_PREFIX`: Installation directory for the `nvLstmInf` binary
* `MODEL_PARAMS_HEADER`: Path of a header of `#define`s with other model sizes (`LSTM_<S|M|L>_NUM_LAYERS`, `_INPUTS`, `_UNITS`, `_STEPS`, all four of a model together) and kernel configurations; default: the built-in sizes of `ss_gemv_impl/kernel_params.h`
* `TBC_USE_MBARRIER`: Use mbarrier primitives for the cluster `lstm_m` kernel's synchronization (FP32/BF16, or FP16 without the FP16 register kernel); the FP16 register kernel always uses them (default: OFF)
* `USE_FP16_REG_KERNELS`: FP16 `lstm_s` and `lstm_m` (always spinning; `lstm_m` if its CTAs fit) and `lstm_l` (if its CTAs fit the device) with the FP16 register-resident kernels of `ss_gemv_impl/fp16_reg/` (default: ON; OFF keeps the older single-block, cluster and device-wide kernels, `ss_gemv_impl/gemv_lstm_{bp,tbc,dev}_impl.cu`, for FP16 as well; FP32 and BF16 always use them)
* `USE_FP16_REG_L_GRAPH`: Build the FP16 register `lstm_l` kernel in its step's precompute graph (`--lstm-l-in-graph`, `--steps-per-graph`): `AUTO`, `ON` or `OFF` (default: `AUTO`, which builds it on x86 with `USE_FP16_REG_KERNELS` and otherwise leaves it out with a message; `ON` builds it on any platform or fails. It sets the switch signal protocol as `USE_SINGLE_STEP_RESIDENT` does; see `src/cpp/nvlstm_inf/ss_gemv_impl/fp16_reg/README.md`)
* `USE_GDRCOPY`: Build GDRCopy signaling support and its dependency (default: ON)
* `USE_SINGLE_STEP_RESIDENT`: Build the resident SingleStep precompute (`--precompute-resident`) for the FP16 register kernels of the persistent `lstm_s` / `lstm_m` models: `AUTO`, `ON` or `OFF` (default: `AUTO`, which builds it on x86 when the build supports it and otherwise leaves it out with a message; `ON` builds it on any platform or fails. It needs `USE_FP16_REG_KERNELS` and numeric CUDA architectures 90 and newer, and sets the switch signal protocol, `DISABLE_ATOMIC_EXCHANGE=1`, for the build: the default on PCIe; on a coherent system such as Grace Hopper it replaces the atomic exchange and costs the other persistent models some latency, see `src/cpp/nvlstm_inf/single_step/README.md`)

To enable the mbarrier optimization of the cluster `lstm_m` kernel, add `-DTBC_USE_MBARRIER=ON` to the cmake configure command. Use `-DUSE_GDRCOPY=OFF` to build without downloading, linking, or installing GDRCopy; the `--gdrcopy` runtime option is then unavailable.

### Testing with CMake

Tests can be run with `ctest` by using the generated `CTestTestfile.cmake` with

```shell
cd build/ && ctest
```

Note that some tests are timing related and may fail when for example built with `-DCMAKE_BUILD_TYPE=Debug`.

### Installation

You may also specify an installation directory to `CMake` by passing the `CMAKE_INSTALL_PREFIX` option like so:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ARCH=native -DCMAKE_CUDA_ARCHITECTURES="120a-real" -DCMAKE_INSTALL_PREFIX=../../nvLstmInf -B build
```

Then running `cd build && make install -j` will install the `nvLstmInf` executable binary and all relevant configuration files in directory `../../nvLstmInf`.

## Source Code Structure

The source code is contained in the following directories:

* `src/cpp/nvlstm_inf`: CUDA based implementation of LSTM Inference
* `src/python`: Python benchmark orchestration scripts
* `thirdparty`: open source libraries used by the benchmark
  * [`libnpy`](https://github.com/llohse/libnpy)
* `test`: unit tests and benchmarking tools
* `docs`: some additional documentation

### Structure within the Implementation

The CUDA based implementation is organized as follows:

* `main.cu`: The main entry point which parses command line parameters and runs the `Benchmark`.
* `Benchmark.cu`, `Benchmark.cuh`: This class executes the benchmark. It instantiates one or several `ModelInstance`s.
* `ModelInstance.cu`, `ModelInstance.cuh`: An instantiation of an independent model instance in the `Benchmark`. It obtains new input data from a `DataIterator` and collects `InferenceResult`s. Model implementations are expected to comply with the `ModelInterface`.
* `DataIterator.h`: Provides access to new input data.
* `InferenceResult.cpp`, `InferenceResult.h`: Collects inference results with timestamps.
* `ModelInterface.h`: Defines the interface of virtual methods of a model implementation expected in `ModelInstance`.
* `ModelSingleStepBase.cu`, `ModelSingleStepBase.cuh`: Base class for single-step LSTM implementations.
* `ModelSingleStep.cu`, `ModelSingleStep.cuh`: Default single-step LSTM implementation using cuBLAS for matrix operations.
* `single_step/`: the single-step kernels, and `nvLstmSingleStepGemmTune` (`gemm_algo_tune.cpp`), the cublasLt tuner for its GEMM, with its cache format (`lt_algo_cache.h`).
* `ModelConfig.cuh`: Internal representation of model configuration parameters.
* `ModelDef.cpp`,`ModelDef.h`: An internal representation of model weights and the network topology.
* `CudaEnv.cu`, `CudaEnv.cuh`: A class that exposes basic information on the CUDA environment.
* `cuda_helpers.cu`, `cuda_helpers.cuh`: A collection of helpers which wrap low level CUDA into C++17 constructs.
* `cuda_copy.cuh`: Helper routines for copying data efficiently between the host and the GPU.
* `util.cpp`, `util.h`: Various helper routines for data loading and saving, implementation of the random sequence generator, etc.
* `ss_gemv_impl/`: Persistent timing kernel implementations for LSTM and ping pong models
* `ModelPersistentBase.cuh`: Base class for all persistent timing models.
* `PingPongModelPersistent.cuh`: Persistent timing implementation of the ping pong model for baseline measurements.
* `ModelSingleStepGemv.cuh`, `ModelSingleStepGemv.cu`: Persistent timing implementation of the LSTM models using kernels from `ss_gemv_impl/`.
* `ModelAsSingleStepGemv.cuh`, `ModelAsSingleStepGemv.cu`: Persistent timing implementation of the LSTM models using kernels from `ss_gemv_impl/` with more advanced signaling. The latency critical kernels are launched once in the beginning of the benchmark.

## Performance Tuning

Some useful commands:
```bash
# This benchmark is sensitive to CPU clock
sudo cpupower frequency-set --governor performance

# Set max clocks on GH200
sudo nvidia-smi -lgc 1980

# Isolate core IDs through grub
# E.g. to isolate 2 cores from the system
# Edit /etc/default/grub with
GRUB_CMDLINE_LINUX="isolcpus=18,19,54,55"
# Where 54-55 are multithreaded cores, if multithreading is enabled
sudo update-grub
# Then point the benchmark to those cores ids

# Run with numactl on multiple socket system
numactl --membind=0 -- $NVLSTM_BIN_DIR/nvLstmInf ...
# Numa topology:
nvidia-smi topo -m
```

## Sample usage
```bash
# development in container exports
export NVLSTM_DATA_DIR=/app/dl-lowlat-infer/data
export NVLSTM_DIR=/code/dl-lowlat-infer
export NVLSTM_OUTPUT_DIR=$NVLSTM_DIR/output
export NVLSTM_BIN_DIR=$NVLSTM_DIR/build/src/cpp/nvlstm_inf
export NVLSTM_PYTHON_DIR=$NVLSTM_DIR/src/python

# Generate ONNX models for all three sizes (Optional)
python $NVLSTM_PYTHON_DIR/create_onnx_model.py --output-dir=$NVLSTM_DATA_DIR

# Convert lstm_s ONNX model to NumPy arrays (Optional). The weights will be dumped into
# $NVLSTM_DATA_DIR/models/lstm_s directory
python $NVLSTM_PYTHON_DIR/convert_model.py $NVLSTM_DATA_DIR/lstm_s.onnx $NVLSTM_DATA_DIR  lstm_s

# Generate lstm_s model weights
python $NVLSTM_PYTHON_DIR/generate_model.py --num-steps=64 --num-units=96 --num-inputs=128  --num-layers=2 --model=lstm_s --output-dir=$NVLSTM_DATA_DIR

# Generate lstm_m model weights
python $NVLSTM_PYTHON_DIR/generate_model.py --num-steps=96 --num-units=160 --num-inputs=192  --num-layers=3 --model=lstm_m --output-dir=$NVLSTM_DATA_DIR

# Generate lstm_l model weights
python $NVLSTM_PYTHON_DIR/generate_model.py --num-steps=128 --num-units=736 --num-inputs=512  --num-layers=4 --model=lstm_l --output-dir=$NVLSTM_DATA_DIR

# Generate input distribution
python $NVLSTM_PYTHON_DIR/generate_data.py --num-inputs 128 -o $NVLSTM_DATA_DIR/lstm_s.npy
python $NVLSTM_PYTHON_DIR/generate_data.py --num-inputs 192 -o $NVLSTM_DATA_DIR/lstm_m.npy
python $NVLSTM_PYTHON_DIR/generate_data.py --num-inputs 512 -o $NVLSTM_DATA_DIR/lstm_l.npy

# Generate the reference using cuBLAS based SingleStep implementation for a single model instance for 30 seconds:
# required parameters:                                    <model> <model_dir> <data_file> <nr_of_sec_to_run>
$NVLSTM_BIN_DIR/nvLstmInf --precision=32 --algo=SingleStep lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 30 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf --precision=32 --algo=SingleStep lstm_m $NVLSTM_DATA_DIR/lstm_m $NVLSTM_DATA_DIR/lstm_m.npy 30 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf --precision=32 --algo=SingleStep lstm_l $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 30 $NVLSTM_OUTPUT_DIR

cp $NVLSTM_OUTPUT_DIR/lstm_s.1.1.npy $NVLSTM_OUTPUT_DIR/lstm_s.1.1.ref.npy
cp $NVLSTM_OUTPUT_DIR/lstm_m.1.1.npy $NVLSTM_OUTPUT_DIR/lstm_m.1.1.ref.npy
cp $NVLSTM_OUTPUT_DIR/lstm_l.1.1.npy $NVLSTM_OUTPUT_DIR/lstm_l.1.1.ref.npy

# Run the inference
# We need three CPU threads to run the following configurations
# main thread, time measurement reaper thread, one green context thread
# required parameters:    <model> <model_dir> <data_file> <nr_of_sec_to_run>
$NVLSTM_BIN_DIR/nvLstmInf lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 30 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf lstm_m $NVLSTM_DATA_DIR/lstm_m $NVLSTM_DATA_DIR/lstm_m.npy 30 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf lstm_l $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 30 $NVLSTM_OUTPUT_DIR
# Each run prints the mean, P50 and P99 latency and the throughput.
# The runs above take these paths by default where they are built (on x86 by default) and work, and print which they
# took, or why not:
# - GDRCopy, on x86 for the FP16 SingleStepGemv and ping-pong models (see "Running with GDRCopy").
# - The resident SingleStep precompute (USE_SINGLE_STEP_RESIDENT): FP16 lstm_s / lstm_m run their precompute as one
#   resident kernel per instance (one model per instance). Each instance's kernel keeps its CTAs resident in the SMs the
#   inference kernels leave, with one SM to spare; a configuration that needs more runs the cuBLAS precompute.
#   --precompute-step-splits=N spreads its pass over more CTAs.
# - The FP16 register lstm_l kernel at the root of each step's precompute graph (USE_FP16_REG_L_GRAPH), with 8 steps per
#   graph (--steps-per-graph=N, 1 or even): more throughput, the same timed step.
# Each is turned off with =0 and required with =1 (or no value: an error where it cannot run):
# --gdrcopy[=0|1], --precompute-resident[=0|1], --lstm-l-in-graph[=0|1], --steps-per-graph=N|auto. Without them:
$NVLSTM_BIN_DIR/nvLstmInf --gdrcopy=0 --precompute-resident=0 --lstm-l-in-graph=0 lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 30 $NVLSTM_OUTPUT_DIR
# The persistent models (--algo=SingleStepGemv) capture their precompute graph on one stream;
# --precompute-multi-stream restores the forked capture (the input copy and the recurrent GEMM on their own streams).

# Check the output:
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.npy
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_m.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_m.1.1.npy
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_l.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_l.1.1.npy

# Run Ping Pong model (ping-pong benchmark) with --ping-pong flag
# Weights and bias are generated internally (no pre-generation needed)
# The ping-pong kernel runs no model (it returns the sum of the input) and waits for one host signal per step:
# the floor of the host-GPU round trip

# Ping Pong model for lstm_s, lstm_m, lstm_l
$NVLSTM_BIN_DIR/nvLstmInf --ping-pong --warm-up=0 lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 2 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf --ping-pong --warm-up=0 lstm_m $NVLSTM_DATA_DIR/lstm_m $NVLSTM_DATA_DIR/lstm_m.npy 2 $NVLSTM_OUTPUT_DIR
$NVLSTM_BIN_DIR/nvLstmInf --ping-pong --warm-up=0 lstm_l $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 2 $NVLSTM_OUTPUT_DIR

# Check the output:
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.npy

```

## Running multiple instances on the same GPU

We need two CPU threads for the main thread and the time measurement reaper thread. 1 CPU thread per additional inference instance. E.g. with 8 instances we would need 10 CPU threads to run. It is good to isolate respective CPU cores and make sure those are separate physical cores.

```bash
# Running 8 inference instances of lstm_s
# Each instance will allocate 2 SMs per green context
$NVLSTM_BIN_DIR/nvLstmInf --cpuset=0,1,2,3,4,5,6,7,8,9 --num-instances=8 lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 2 $NVLSTM_OUTPUT_DIR

# Running 4 inference instances of lstm_m
# Each instance will allocate up to 16 SMs per green context
$NVLSTM_BIN_DIR/nvLstmInf --cpuset=0,1,2,3,4,5 --num-instances=4 lstm_m $NVLSTM_DATA_DIR/lstm_m $NVLSTM_DATA_DIR/lstm_m.npy 2 $NVLSTM_OUTPUT_DIR
```

lstm_l would occupy the whole GPU, so we run only a single instance.

## Tuning the precompute GEMM

In half precision the SingleStep precompute (also the precompute of the persistent `--algo=SingleStepGemv` models)
runs the input GEMM of layers >= 1, `C[4U x steps] = W[4U x U] * X[U x steps]` (U = units), through cublasLt with the
heuristic's first pick. `nvLstmSingleStepGemmTune` times many cublasLt algorithms for that shape and writes the
winner to a cache file when it is at least 2 % faster than that pick (`--min-gain`); `nvLstmInf --lt-algo-cache=FILE`
then uses it. Without the option nothing changes.

```bash
# Tune for a model size on the GPU the benchmark runs on (--model=lstm_s|lstm_m|lstm_l: the build's sizes; or --units=U --steps=T)
$NVLSTM_BIN_DIR/nvLstmSingleStepGemmTune --model=lstm_l --exhaustive --cache=lt_cache.txt

# Run with the cache; one stderr line per GEMM shape says whether a tuned algorithm is used
$NVLSTM_BIN_DIR/nvLstmInf --precision=16 --algo=SingleStepGemv --lt-algo-cache=lt_cache.txt lstm_l \
  $NVLSTM_DATA_DIR/lstm_l $NVLSTM_DATA_DIR/lstm_l.npy 30 $NVLSTM_OUTPUT_DIR
```

Entries are keyed by the GPU, the number of SMs the GEMM runs on, the cublasLt version and the GEMM shape. The
persistent `lstm_s` / `lstm_m` models run their precompute on a green context of the SMs the inference kernels leave
free, which depends on the number of instances: the `[lt-algo-cache]` line prints that SM count, and
`nvLstmSingleStepGemmTune --sms=N` tunes for it. An entry that does not match, or that cublasLt rejects for the
model's matrix layouts, leaves the heuristic's pick in place.

## Building and running with `docker`

The benchmark can be built and run inside a `docker` container. This requires installation of the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The provided `Makefile` automates image builds and runs.

The Docker stages serve different purposes:

- `develop` provides a development environment; it does not build the project.
- `release` builds and installs `nvLstmInf` into the image.
- `models` provides generated model weights and input data; it does not build the project.
- `test` builds the project with tests enabled; `test_run` runs CTest.

To build images without running a container:

```bash
make -C docker develop_build
make -C docker CUDA_ARCHS=120a-real release_build
make -C docker CUDA_ARCHS=120a-real test_build
```

In the easiest case, one can build and run the docker container as follows.

```bash
# Build and run the docker image interactively
make -C docker CUDA_ARCHS=120a-real release_run
```

Supply the CUDA architecture appropriate for your GPU whenever the selected stage builds the project. The above command will drop you into a bash shell inside of the container. The application is located in `/app/dl-lowlat-infer`. From there you can run the benchmark for example like so:

```bash
./nvLstmInf --precision=32 --algo=SingleStep lstm_s data/lstm_s data/lstm_s.npy 30
```

The current source directory is mounted at `/code/dl-lowlat-infer` for development, release, and models runs. The release image runs the binary baked into `/app/dl-lowlat-infer`; changes in the mounted source directory do not change that binary. The mount is useful for development and for saving benchmark output. When a container must write to the mounted directory, add `LOCAL_USER=1` so it uses your host UID and GID. The `test_run` target instead runs from the image build tree and does not mount the source directory. For example:

```bash
make -C docker CUDA_ARCHS=120a-real LOCAL_USER=1 release_run
export NVLSTM_OUTPUT_DIR=/code/dl-lowlat-infer/output
mkdir -p $NVLSTM_OUTPUT_DIR
./nvLstmInf --precision=32 --algo=SingleStep lstm_s data/lstm_s data/lstm_s.npy 30 $NVLSTM_OUTPUT_DIR
```

For interactive development, you can also build and run the `develop` stage of the dockerfile:

```bash
make -C docker LOCAL_USER=1 develop_run
```

You can then build the source code inside the container with `cmake` as described in the [Building with CMake](#building-with-cmake) section.

Run targets rebuild their image by default. To reuse an existing image, set `BUILD_BEFORE_RUN=0`:

```bash
make -C docker BUILD_BEFORE_RUN=0 LOCAL_USER=1 develop_run
```

Running the `models` stage will add access to model weights and input data:

```bash
make -C docker LOCAL_USER=1 models_run
```

The models are stored in `/app/dl-lowlat-infer/data` inside the container.

If you want to run tests inside the container, you can use the `test_run` target:

```bash
make -C docker CUDA_ARCHS=120a-real test_run
```

## Running with GDRCopy

Polling a device on a flag located in a pinned host buffer can be quite expensive. [GDRCopy](https://github.com/NVIDIA/gdrcopy) provides a low-latency alternative by creating a CPU mapping of GPU memory using GPUDirect RDMA. This allows CPU-driven memory copies with minimal overhead — particularly beneficial in low-latency scenarios where small data transfers are small and frequent. With `--gdrcopy` the host writes both the step's signal and, on PCIe systems for the persistent `lstm_s` / `lstm_m` models and the ping-pong model, the step's input into device memory, so their kernels read neither over PCIe. On Grace Hopper they read the input over the coherent C2C link as before; `-DGDRCOPY_DEVICE_INPUT=0|1` in `CMAKE_CUDA_FLAGS` overrides the default.

In our experiments, using GDRCopy has yielded speedups of up to **0.5 µs** on PCIe-based systems.

The build uses GDRCopy v2.6, which maps device memory in one of two ways:
- **The `gdrdrv` kernel module**, if it is loaded (`/dev/gdrdrv`). Loading it needs root; see below.
- **Without the module, GDRCopy's DMA-BUF backend**, which needs no root but a driver supporting CUDA 13.3 or newer: the CUDA version `nvidia-smi` shows in its header ("CUDA Version" or, on newer drivers, "CUDA UMD Version"), not the CUDA toolkit the build used.

On x86, `nvLstmInf` uses GDRCopy by default for the FP16 SingleStepGemv and ping-pong models where it works: it checks first (it maps, writes and reads back a device page) and prints the backend it uses, e.g. `GDRCopy:      Yes (gdrdrv; default)`, or `Yes (DMA-BUF, driver CUDA 13.4; default)` on a driver supporting CUDA 13.4; or, if neither way works, it runs without GDRCopy and prints why (e.g. that the module could not be opened and the driver supports only CUDA 13.2). `--gdrcopy=0` turns it off; `--gdrcopy` (or `=1`) requires it, on any platform and model, and stops with the reason where it does not work.

**With a driver supporting CUDA 13.3 or newer, GDRCopy needs nothing else:** no kernel module, no root, and in Docker no `--privileged` and no `--device /dev/gdrdrv`. The DMA-BUF backend goes through the CUDA driver, which `--gpus` already gives the container (`make -C docker ... develop_run` as it is). Using the project without GDRCopy remains fully supported as described in previous sections. Where GDRCopy works, x86 runs use it by default; elsewhere, or to make sure, add the `--gdrcopy` flag to your command:

```bash
$NVLSTM_BIN_DIR/nvLstmInf lstm_s \
  $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 30 --gdrcopy
```

### Drivers older than CUDA 13.3: the `gdrdrv` module

With an older driver, GDRCopy works only through the `gdrdrv` kernel module. It must be compiled and loaded on the host system (root) before starting the container; kernel modules cannot be loaded inside containers since they share the host's kernel. The `gdrdrv` directory contains helper scripts for this purpose:
- `build_gdrcopy_driver.sh` — Compiles the GDRCopy kernel module for your current kernel version (defaults to GDRCopy v2.6 if `GDRCOPY_TAG` is not set)
- `insmod.sh` — Loads the compiled module into the running kernel (requires root privileges)

```bash
cd gdrdrv
./build_gdrcopy_driver.sh
./insmod.sh
```

Then pass the device to the container with `--device /dev/gdrdrv` through the `DOCKER_RUN_ARGS` variable of the Makefile. Pass it only where the host has the module loaded: without `/dev/gdrdrv` on the host, `docker run` fails, while GDRCopy would have worked through DMA-BUF on a driver supporting CUDA 13.3 or newer. A `/dev/gdrdrv` that does not open (e.g. left from another module version) does no harm: GDRCopy then falls back to DMA-BUF where the driver supports it.
