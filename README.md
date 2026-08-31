# Low Latency inference for sliding window LSTMs

Coding-agent instructions (build constraints, architecture, conventions): see [AGENTS.md](AGENTS.md). Claude Code loads the same content via [CLAUDE.md](CLAUDE.md).

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
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ARCH=native -DCMAKE_CUDA_ARCHITECTURES="90a-real" -B build
cmake --build build --parallel
```

### Build Options

Key CMake options:

* `CMAKE_BUILD_TYPE`: Build type (Release or Debug)
* `CPU_ARCH`: Target CPU architecture (default: `icelake-server`, use `native` for current CPU)
* `CMAKE_CUDA_ARCHITECTURES`: Target CUDA architecture (e.g., `"90a-real"`, `"100a-real"`, `"120a-real"`)
* `CMAKE_INSTALL_PREFIX`: Installation directory for the `nvLstmInf` binary
* `TBC_USE_MBARRIER`: Use mbarrier primitives for thread block cluster synchronization (default: OFF)

To enable mbarrier optimization, add `-DTBC_USE_MBARRIER=ON` to the cmake configure command.

### Testing with CMake

Tests can be run with `ctest` by using the generated `CTestTestfile.cmake` with

```shell
cd build/ && ctest
```

Note that some tests are timing related and may fail when for example built with `-DCMAKE_BUILD_TYPE=Debug`.

### Installation

You may also specify an installation directory to `CMake` by passing the `CMAKE_INSTALL_PREFIX` option like so:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ARCH=native -DCMAKE_CUDA_ARCHITECTURES="90a-real" -DCMAKE_INSTALL_PREFIX=../../nvLstmInf -B build
```

Then running `cd build && make install -j` will install the `nvLstmInf` executable binary and all relevant configuration files in directory `../../nvLstmInf`. 

## Source Code Structure

The source code is contained in the following directories:

* `src/cpp/nvLstmInf`: CUDA based implementation of LSTM Inference
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
* `ModelConfig.cuh`: Internal representation of model configuration parameters.
* `ModelDef.cpp`,`ModelDef.h`: An internal representation of model weights and the network topology.
* `CudaEnv.cu`, `CudaEnv.cuh`: A class that exposes basic information on the CUDA environment.
* `cuda_helpers.cu`, `cuda_helpers.cuh`: A collection of helpers which wrap low level CUDA into C++17 constructs.
* `cuda_copy.cuh`: Helper routines for copying data efficiently between the host and the GPU.
* `util.cpp`, `util.h`: Various helper routines for data loading and saving, implementation of the random sequence generator, etc. 
* `ss_gemv_impl\`: Persistent timing kernel implementations for LSTM and ping pong models
* `ModelPersistentBase.cuh`: Base class for all persistent timing models.
* `PingPongModelPersistent.cuh`: Persistent timing implementation of the ping pong model for baseline measurements.
* `ModelSingleStepGemv.cuh`, `ModelSingleStepGemv.cu`: Persistent timing implementation of the LSTM models using kernels from `ss_gemv_impl\`.
* `ModelAsSingleStepGemv.cuh`, `ModelAsSingleStepGemv.cu`: Persistent timing implementation of the LSTM models using kernels from `ss_gemv_impl\` with mode advanced signaling. The latency critical kernels are launched once in the beginning of the benchmark.

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

# Generate the reference using cuBLAS based SingleStep implementation for a single model instance for 2 seconds:
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

# Check the output:
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_s.1.1.npy
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_m.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_m.1.1.npy
python $NVLSTM_PYTHON_DIR/check.py --ref=$NVLSTM_OUTPUT_DIR/lstm_l.1.1.ref.npy --out=$NVLSTM_OUTPUT_DIR/lstm_l.1.1.npy

# Run Ping Pong model (ping-pong benchmark) with --ping-pong flag
# Weights and bias are generated internally (no pre-generation needed)

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
# Running 8 inference instances of LSTM S
# Each instance will allocate 2 SMs per green context
$NVLSTM_BIN_DIR/nvLstmInf --cpuset=0,1,2,3,4,5,6,7,8,9 --num-instances=8 lstm_s $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 2 $NVLSTM_OUTPUT_DIR

# Running 4 inference instances of LSTM M
# Each instance will alocate up to 16 SMs per green context
$NVLSTM_BIN_DIR/nvLstmInf --cpuset=0,1,2,3,4,5 --num-instances=4 lstm_m $NVLSTM_DATA_DIR/lstm_m $NVLSTM_DATA_DIR/lstm_m.npy 2 $NVLSTM_OUTPUT_DIR
```

lstm_l would occupy the whole GPU, so we run only a single instance.

## Building and running with `docker`

The benchmark can be built and run inside a `docker` container. This requires installation of the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). For full automation of the build and push process, `Makefile` targets are provided.

In the easiest case, one can build and run the docker container as follows.

```bash
# Build and run the docker image interactively
make -C docker CUDA_ARCHS=90a-real release_run
```

Please make sure to supply the correct CUDA architecture for your GPU with the `CUDA_ARCHS` argument. The above command will drop you into a bash shell inside of the container. The application is located in `/app/dl-lowlat-infer`. From there you can run the benchmark for example like so:

```bash
./nvLstmInf --precision=32 --algo=SingleStep lstm_s data/lstm_s data/lstm_s.npy 30
```

The current source directory is mounted into the container at `/code/dl-lowlat-infer`. If you want to record the output of the benchmark, you can create a directory in the source tree and supply it to the benchmark driver. However, make sure to supply the `LOCAL_USER=1` argument to `make` so that the container user has write permissions to the mounted directory. For example:

```bash
make -C docker CUDA_ARCHS=90a-real LOCAL_USER=1 release_run
export NVLSTM_OUTPUT_DIR=/code/dl-lowlat-infer/output
mkdir -p $NVLSTM_OUTPUT_DIR
./nvLstmInf --precision=32 --algo=SingleStep lstm_s data/lstm_s data/lstm_s.npy 30 $NVLSTM_OUTPUT_DIR
```

For interactive development, you can also build and run the `develop` stage of the dockerfile:

```bash
make -C docker CUDA_ARCHS=90a-real LOCAL_USER=1 develop_run
```

You can then build the source code inside the container with `cmake` as described in the [Building with CMake](#building-with-cmake) section.

Running the `models` stage will add access to model weights and input data:

```bash
make -C docker CUDA_ARCHS=90a-real LOCAL_USER=1 models_run
```

The models are stored in `/app/dl-lowlat-infer/data` inside the container.

If you want to run tests inside the container, you can use the `test_run` target:

```bash
make -C docker CUDA_ARCHS=90a-real test_run
```

## Running with GDRCopy

Polling a device on a flag located in a pinned host buffer can be quite expensive. [GDRCopy](https://github.com/NVIDIA/gdrcopy) provides a low-latency alternative by creating a CPU mapping of GPU memory using GPUDirect RDMA. This allows CPU-driven memory copies with minimal overhead — particularly beneficial in low-latency scenarios where small data transfers are small and frequent.

In our experiments, using GDRCopy has yielded speedups of up to **0.5 µs** on PCIe-based systems.  

**Important:** The GDRCopy kernel module must be compiled and loaded on the host system before starting the container. Kernel modules cannot be loaded inside containers since they share the host's kernel.

The `gdrdrv` directory contains helper scripts for this purpose:
- `build_gdrcopy_driver.sh` — Compiles the GDRCopy kernel module for your current kernel version (defaults to GDRCopy v2.5.1 if `GDRCOPY_TAG` is not set)
- `insmod.sh` — Loads the compiled module into the running kernel (requires root privileges)

To set up GDRCopy support:

```bash
cd gdrdrv
./build_gdrcopy_driver.sh
./insmod.sh
```

Once the module is loaded on the host, you can start the container (with `--privileged` flag, or with `--device /dev/gdrdrv` passed throguh `DOCKER_RUN_ARGS` variable in the Makefile), and GDRCopy operations will be supported. Using the project without GDRCopy remains fully supported as described in previous sections. If the GDRCopy kernel module has been successfully loaded, you can enable GDRCopy by adding the `--gdrcopy` flag to your command: 

```bash $NVLSTM_BIN_DIR/nvLstmInf lstm_s \
   $NVLSTM_DATA_DIR/lstm_s $NVLSTM_DATA_DIR/lstm_s.npy 30 \
   $NVLSTM_OUTPUT_DIR --gdrcopy```