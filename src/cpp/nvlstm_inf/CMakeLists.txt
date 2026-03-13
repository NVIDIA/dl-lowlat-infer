# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(nvlstminf_src OBJECT
        cuda_error_util.cpp
        gdrcopy_util.cpp
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
        ss_gemv_impl/gemv_lstm_dev_impl.cu
        )

target_link_libraries(nvlstminf_src INTERFACE cublas cublasLt cuda)
target_link_libraries(nvlstminf_src PUBLIC gdrcopy)
# disable for now
#target_compile_definitions(nvlstminf_src PUBLIC USE_GDRCOPY=1)
target_include_directories(
        nvlstminf_src 
        PRIVATE 
                "${PROJECT_SOURCE_DIR}/thirdparty/include"
                "${CUDAToolkit_INCLUDE_DIRS}"
        PUBLIC
                "${PROJECT_SOURCE_DIR}/src/cpp"
)

if(USE_NVTX)
	target_link_libraries(nvlstminf_src PRIVATE nvtx3-cpp)
endif()

add_executable(nvLstmInf main.cu)
target_link_libraries(nvLstmInf nvlstminf_src)

# Set RPATH for gdrcopy shared library
set_target_properties(nvLstmInf PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/_deps/gdrcopy-build/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/../gdrcopy/lib"
)

install(TARGETS nvLstmInf
		RUNTIME DESTINATION .)

