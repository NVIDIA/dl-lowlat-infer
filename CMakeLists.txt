# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.20)

option(BUILD_TESTS "Build Google tests" ON)
option(USE_GDRCOPY "Build with optional GDRCopy signaling support" ON)
option(USE_FP16_REG_KERNELS "FP16 lstm_s, lstm_m and lstm_l with the FP16 register-resident kernels" ON)
set(USE_SINGLE_STEP_RESIDENT
    AUTO
    CACHE STRING
          "Resident SingleStep precompute for the FP16 register kernels (--precompute-resident): AUTO, ON or OFF")
set_property(CACHE USE_SINGLE_STEP_RESIDENT PROPERTY STRINGS AUTO ON OFF)
set(USE_FP16_REG_L_GRAPH
    AUTO
    CACHE STRING "FP16 register lstm_l kernel in its step's precompute graph (--lstm-l-in-graph): AUTO, ON or OFF")
set_property(CACHE USE_FP16_REG_L_GRAPH PROPERTY STRINGS AUTO ON OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90a-real;100a-real;120a-real")
endif()

if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

if(NOT DEFINED CPU_ARCH)
    set(CPU_ARCH icelake-server)
endif()

message(STATUS "Cuda architectures: " ${CMAKE_CUDA_ARCHITECTURES})

project(nvLstmInf LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Optional features with an AUTO / ON / OFF option (default AUTO): AUTO builds the feature on x86 when the build
# supports it (unsupported: empty, else why not) and leaves it out otherwise, with a message; ON builds it on any
# platform or fails; OFF leaves it out. Sets the option's variable to ON or OFF.
function(auto_option name description unsupported)
    string(TOUPPER "${${name}}" choice)
    if(choice STREQUAL "AUTO")
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            set(off_reason "on by default on x86 only, not ${CMAKE_SYSTEM_PROCESSOR}; -D${name}=ON builds it")
        else()
            set(off_reason "${unsupported}")
        endif()
        if(off_reason)
            message(STATUS "${description}: off (${off_reason})")
            set(${name}
                OFF
                PARENT_SCOPE)
        else()
            set(${name}
                ON
                PARENT_SCOPE)
        endif()
    elseif(choice MATCHES "^(ON|TRUE|YES|1)$")
        if(unsupported)
            message(FATAL_ERROR "${name} ${unsupported}")
        endif()
        set(${name}
            ON
            PARENT_SCOPE)
    elseif(choice MATCHES "^(OFF|FALSE|NO|0)$")
        set(${name}
            OFF
            PARENT_SCOPE)
    else()
        message(FATAL_ERROR "${name} must be AUTO, ON or OFF, not ${${name}}")
    endif()
endfunction()

# The resident SingleStep precompute (src/cpp/nvlstm_inf/single_step/README.md): the FP16 register kernels, sm_90 and
# newer (TMA, mbarrier).
set(single_step_resident_unsupported "")
if(NOT USE_FP16_REG_KERNELS)
    set(single_step_resident_unsupported "requires USE_FP16_REG_KERNELS")
endif()
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
    string(REGEX MATCH "^[0-9]+" arch_number "${arch}")
    if(NOT arch_number)
        set(single_step_resident_unsupported "needs numeric CUDA architectures (e.g. 120a-real), not ${arch}")
    elseif(arch_number LESS 90)
        set(single_step_resident_unsupported "requires CUDA architectures 90 and newer, not ${arch}")
    endif()
endforeach()
auto_option(USE_SINGLE_STEP_RESIDENT "Resident SingleStep precompute" "${single_step_resident_unsupported}")

# The FP16 register lstm_l kernel in its step's precompute graph (src/cpp/nvlstm_inf/ss_gemv_impl/fp16_reg/README.md).
set(fp16_reg_l_graph_unsupported "")
if(NOT USE_FP16_REG_KERNELS)
    set(fp16_reg_l_graph_unsupported "requires USE_FP16_REG_KERNELS")
endif()
auto_option(USE_FP16_REG_L_GRAPH "FP16 register lstm_l kernel in the precompute graph"
            "${fp16_reg_l_graph_unsupported}")

# Both need the switch signal protocol, which x86 (PCIe) builds use already and which they set on Grace Hopper too (no
# path for the atomic exchange yet).
if(USE_SINGLE_STEP_RESIDENT OR USE_FP16_REG_L_GRAPH)
    add_compile_definitions(DISABLE_ATOMIC_EXCHANGE=1 SIGNAL_SWITCH_INIT_VALUE=1)
endif()
if(USE_SINGLE_STEP_RESIDENT)
    message(STATUS "Resident SingleStep precompute: on (switch signal protocol)")
endif()
if(USE_FP16_REG_L_GRAPH)
    message(STATUS "FP16 register lstm_l kernel in the precompute graph: on (switch signal protocol)")
endif()

find_package(OpenMP 4.5 REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -march=${CPU_ARCH}")
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --compiler-options -Wall,-fopenmp,-march=${CPU_ARCH} -Xptxas=--warn-on-spills --use_fast_math --generate-line-info -std=c++20"
)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug")
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit 13.0)

if(USE_GDRCOPY)
    include(gdrcopy)
endif()

option(USE_NVTX "Whether or not to use nvtx" OFF)
if(USE_NVTX)
    # NVTX 3 downloaded from GitHub
    include(CPM)
    cpmaddpackage(
        NAME
        NVTX
        GITHUB_REPOSITORY
        NVIDIA/NVTX
        GIT_TAG
        v3..0-c-cpp
        GIT_SHALLOW
        TRUE)

    if(TARGET nvtx3-cpp)
        get_target_property(NVTX3_VERSION nvtx3-cpp VERSION)
        get_target_property(NVTX3_IMPORTED nvtx3-cpp IMPORTED)
        message(STATUS "nvtx3-cpp found: ${NVTX3_VERSION} (imported: ${NVTX3_IMPORTED})")
    else()
        message(ERROR "nvtx3-cpp not found")
    endif()
    add_definitions("-DUSE_NVTX")
endif()

# nvidia-smi -q | grep -i atomic | grep -i outbound experimental, we do not use it in actual code
option(USE_PCIE_OUTBOUND_ATOMICS "Indicates that the target system supports outbound PCIe atomics." OFF)
if(USE_PCIE_OUTBOUND_ATOMICS)
    message(STATUS "Enabling PCIe Outbound Atomics")
    add_definitions("-DUSE_PCIE_OUTBOUND_ATOMICS")
endif()

option(TBC_USE_MBARRIER "Use mbarrier pattern for TBC inter-layer sync" OFF)
if(TBC_USE_MBARRIER)
    message(STATUS "TBC mbarrier optimization: Enabled")
    add_definitions("-DTBC_USE_MBARRIER")
endif()

# Model sizes and kernel configurations other than the built-in ones: a header of #defines
# (ss_gemv_impl/kernel_params.h)
set(MODEL_PARAMS_HEADER
    ""
    CACHE FILEPATH "Header setting other model sizes and kernel configurations")
if(MODEL_PARAMS_HEADER)
    message(STATUS "Model sizes and kernel configurations from ${MODEL_PARAMS_HEADER}")
    add_compile_definitions(MODEL_PARAMS_HEADER="${MODEL_PARAMS_HEADER}")
endif()

add_subdirectory(src/cpp/nvlstm_inf)
add_subdirectory(src/python)

# Testing
if(BUILD_TESTS)
    enable_testing()
    message(STATUS "Building tests")
    add_subdirectory(test/cpp/nvlstm_inf)
else()
    message(STATUS "Not building tests")
endif()
