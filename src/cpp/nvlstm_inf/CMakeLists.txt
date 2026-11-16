# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(
    nvlstminf_src OBJECT
    cuda_error_util.cpp
    util.cpp
    Benchmark.cu
    cuda_helpers.cu
    CudaEnv.cu
    InferenceResult.cpp
    ModelDef.cpp
    ModelInstance.cu
    ModelSingleStepBase.cu
    ModelSingleStep.cu
    ModelSingleStepGemv.cu
    ModelAsSingleStepGemv.cu
    ss_gemv_impl/gemv_lstm_bp_impl.cu
    ss_gemv_impl/gemv_lstm_tbc_impl.cu
    ss_gemv_impl/gemv_lstm_dev_impl.cu)

target_link_libraries(nvlstminf_src INTERFACE cublas cublasLt cuda)
target_compile_definitions(nvlstminf_src PUBLIC GDRCOPY_AVAILABLE=$<BOOL:${USE_GDRCOPY}>)
if(USE_GDRCOPY)
    target_sources(nvlstminf_src PRIVATE gdrcopy_util.cpp)
    target_link_libraries(nvlstminf_src PUBLIC gdrcopy)
endif()
target_compile_definitions(nvlstminf_src PUBLIC USE_FP16_REG_KERNELS=$<BOOL:${USE_FP16_REG_KERNELS}>)
if(USE_FP16_REG_KERNELS)
    target_sources(nvlstminf_src PRIVATE ss_gemv_impl/fp16_reg/gemv_lstm_fp16_reg_impl.cu)
endif()
target_compile_definitions(nvlstminf_src PUBLIC USE_SINGLE_STEP_RESIDENT=$<BOOL:${USE_SINGLE_STEP_RESIDENT}>)
target_compile_definitions(nvlstminf_src PUBLIC USE_FP16_REG_L_GRAPH=$<BOOL:${USE_FP16_REG_L_GRAPH}>)
if(USE_SINGLE_STEP_RESIDENT)
    target_sources(nvlstminf_src PRIVATE single_step/single_step_resident.cu)
endif()
target_include_directories(
    nvlstminf_src
    PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty/include" "${CUDAToolkit_INCLUDE_DIRS}"
    PUBLIC "${PROJECT_SOURCE_DIR}/src/cpp")

if(USE_NVTX)
    target_link_libraries(nvlstminf_src PRIVATE nvtx3-cpp)
endif()

add_executable(nvLstmInf main.cu)
target_link_libraries(nvLstmInf nvlstminf_src)

if(USE_GDRCOPY)
    # Set RPATH for the optional GDRCopy shared library.
    set_target_properties(nvLstmInf PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/_deps/gdrcopy-build/lib"
                                               INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/../gdrcopy/lib")
endif()

# Tuner for the cublasLt algorithm of the SingleStep precompute GEMM.  It writes a cache file of tuned algorithms
# (format in single_step/lt_algo_cache.h) that nvLstmInf reads with --lt-algo-cache; its --model sizes are the build's
# (ss_gemv_impl/kernel_params.h).  It does not link the model library.
add_executable(nvLstmSingleStepGemmTune single_step/gemm_algo_tune.cpp cuda_error_util.cpp)
target_link_libraries(nvLstmSingleStepGemmTune PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
# <cuda/cmath>: CUDA 13 keeps CCCL under include/cccl, which nvcc adds on its own but the host compiler does not
find_path(
    CCCL_INCLUDE_DIR cuda/cmath
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl
    NO_DEFAULT_PATH REQUIRED)
target_include_directories(nvLstmSingleStepGemmTune PRIVATE "${CUDAToolkit_INCLUDE_DIRS}" "${CCCL_INCLUDE_DIR}"
                                                            "${PROJECT_SOURCE_DIR}/src/cpp")

install(TARGETS nvLstmInf nvLstmSingleStepGemmTune RUNTIME DESTINATION .)
