# AGENTS.md

Canonical agent instructions for this repository (Cursor, Codex, Claude Code via `CLAUDE.md` import, and other coding agents).

## Project

Low-latency single-step streaming LSTM (and related) inference on NVIDIA GPUs: CUDA, cuBLAS/cuBLASLt, and custom persistent-timing kernels. Batch size is always 1 (one timestep per inference call).

Model sizes (features / steps / layers / units): `lstm_s` 128/64/2/96, `lstm_m` 192/96/3/160, `lstm_l` 512/128/4/736.

## Build, test, Docker

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ARCH=native -DCMAKE_CUDA_ARCHITECTURES="90a-real" -B build
cmake --build build --parallel
cd build/ && ctest
```

Key CMake options: `CMAKE_BUILD_TYPE`, `CPU_ARCH` (default `icelake-server`; use `native` locally), `CMAKE_CUDA_ARCHITECTURES` (e.g. `90a-real`, `100a-real`, `120a-real`), `CMAKE_INSTALL_PREFIX`, `BUILD_TESTS`, `USE_NVTX`, `USE_PCIE_OUTBOUND_ATOMICS`. Some timing tests fail in Debug.

Docker (needs NVIDIA Container Toolkit; `LOCAL_USER=1` for mounted-volume permissions):

```bash
make -C docker CUDA_ARCHS=90a-real LOCAL_USER=1 develop_run
make -C docker CUDA_ARCHS=90a-real release_run
make -C docker CUDA_ARCHS=90a-real LOCAL_USER=1 models_run
make -C docker CUDA_ARCHS=90a-real test_run
```

Requires Git LFS to clone (`sudo apt-get install git-lfs` on Ubuntu). Runtime stack: CUDA 13.0+, cuBLAS, OpenMP 4.5+; optional cuBLASLt and GDRCopy. Tests use Google Test (CUTLASS only in `cutlass_test.cu`). No cuDNN.

## Formatting

- C++: `clang-format -i <files>` with the repo `.clang-format`
- Python: `black src/python/*.py`

## Hard constraints

- Batch size must be 1; streaming one timestep per call
- `--algo=SingleStepGemv` requires `--precision=16` (FP16)
- LSTM gate order in `.npy` storage is **iofc** (not Wikipedia fioc / PyTorch ifco); reorder when converting formats
- Commits must be signed off: `git commit -s` (DCO 1.1)

## Architecture

Layered flow: **Benchmark** → **ModelInstance** (DataIterator + InferenceResult) → **ModelInterface** implementations.

Notable models: `ModelSingleStep` (cuBLAS), `ModelSingleStepGemv` / `ModelAsSingleStepGemv` (persistent kernels in `ss_gemv_impl/`), `PingPongModelPersistent` (baseline). Persistent kernels stay alive after init (S: one block; M: TBC; L: full device); CPU–GPU sync via atomics/spin.

CLI `--algo`: `SingleStep`, `SingleStepStreaming`, `SingleStepGemv`, or PingPong via `--ping-pong`. Config lives in `ModelConfig.cuh` (`data_type`, `algorithm`, `pre_compute`, `use_cublas_lt`, `as_supported`, `use_gdrcopy`).

## Key paths

Under `src/cpp/nvlstm_inf/` (and related): `main.cu`, `Benchmark.*`, `ModelInstance.*`, `ModelInterface.h`, `ModelConfig.cuh`, `ModelDef.*`, `CudaEnv.*`, `ss_gemv_impl/`, `gdrcopy_util.*`. Python helpers in `src/python/`.

## Where to look next

Long playbooks are **not** repeated here. See [README.md](README.md) for generate/run/validate recipes, multi-instance (`--cpuset` / `--num-instances`), performance tuning (CPU governor, GPU clocks, NUMA), Docker details, and GDRCopy.
