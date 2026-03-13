# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

include(gtest)
include(GoogleTest)
include(cutlass)

SET(RES_DIR ${PROJECT_SOURCE_DIR}/test/resources)

function(add_gtest test_name test_src)
    add_executable(
            ${test_name}
            ${test_src}
    )

    target_link_libraries(
            ${test_name}
            nvlstminf_src
            gtest_main
    )

    target_include_directories(${test_name} SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include)
    target_compile_definitions(${test_name} PUBLIC TEST_RESOURCES_DIR="${RES_DIR}")
    
    # Set RPATH for gdrcopy shared library
    set_target_properties(${test_name} PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/_deps/gdrcopy-build/lib"
    )

    gtest_discover_tests(${test_name}
        PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=LAZY" DISCOVERY_MODE PRE_TEST)
endfunction()

add_gtest(benchmark_test benchmark_test.cu)
add_gtest(data_test data_test.cu)
add_gtest(copy_test copy_test.cu)
add_gtest(cuda_test cuda_test.cu)
add_gtest(cutlass_test cutlass_test.cu)
add_gtest(inference_result_test inference_result_test.cpp)
add_gtest(model_def_test model_def_test.cpp)
add_gtest(model_test model_test.cu)
add_gtest(model_instance_test model_instance_test.cu)
add_gtest(ping_pong_model_test ping_pong_model_test.cu)
add_gtest(util_test util_test.cpp)