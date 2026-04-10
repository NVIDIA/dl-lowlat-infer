# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.20)

option(BUILD_TESTS "Build Google tests" ON)

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

find_package(OpenMP 4.5 REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -fopenmp -march=${CPU_ARCH}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall,-fopenmp,-march=${CPU_ARCH} -Xptxas=--warn-on-spills --use_fast_math --generate-line-info -std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug")
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit 13.0)

include(gdrcopy)

option(USE_NVTX "Whether or not to use nvtx" OFF)
if(USE_NVTX)
    # NVTX 3 downloaded from GitHub
    include(CPM)
    CPMAddPackage(
            NAME NVTX
            GITHUB_REPOSITORY NVIDIA/NVTX
            GIT_TAG v3..0-c-cpp
            GIT_SHALLOW TRUE)

    if (TARGET nvtx3-cpp)
        get_target_property(NVTX3_VERSION nvtx3-cpp VERSION)
        get_target_property(NVTX3_IMPORTED nvtx3-cpp IMPORTED)
        message(STATUS "nvtx3-cpp found: ${NVTX3_VERSION} (imported: ${NVTX3_IMPORTED})")
    else()
        message(ERROR "nvtx3-cpp not found")
    endif()
    add_definitions("-DUSE_NVTX")
endif()

# nvidia-smi -q | grep -i atomic | grep -i outbound
# experimental, we do not use it in actual code
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

add_subdirectory(src/cpp/nvlstm_inf)
add_subdirectory(src/python)
#add_subdirectory(configs)

# Testing
if(BUILD_TESTS) 
    enable_testing()
    message(STATUS "Building tests")
    add_subdirectory(test/cpp/nvlstm_inf)
else()
    message(STATUS "Not building tests")
endif()
