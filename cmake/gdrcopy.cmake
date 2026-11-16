include(FetchContent)

# Use FetchContent to download gdrcopy source to _deps/
FetchContent_Declare(
    gdrcopy
    GIT_REPOSITORY https://github.com/NVIDIA/gdrcopy.git
    GIT_TAG v2.6)

FetchContent_GetProperties(gdrcopy)
if(NOT gdrcopy_POPULATED)
    FetchContent_Populate(gdrcopy)
    # gdrcopy uses Makefile, not CMake, so we can't use add_subdirectory
endif()

# Set up paths - source in _deps, build output also in _deps
set(GDRCOPY_SOURCE_DIR ${gdrcopy_SOURCE_DIR})
set(GDRCOPY_BUILD_DIR ${CMAKE_BINARY_DIR}/_deps/gdrcopy-build)
set(GDRCOPY_INCLUDE_DIR ${GDRCOPY_BUILD_DIR}/include)
set(GDRCOPY_LIB_DIR ${GDRCOPY_BUILD_DIR}/lib)
set(GDRCOPY_LIBRARY ${GDRCOPY_LIB_DIR}/libgdrapi.so)

# Create build directory
file(MAKE_DIRECTORY ${GDRCOPY_BUILD_DIR})

# Build gdrcopy at configure time using execute_process
message(STATUS "Building gdrcopy from ${GDRCOPY_SOURCE_DIR}")
find_program(MAKE_EXECUTABLE NAMES make REQUIRED)

# Allow parallel builds
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 1)
endif()

# The library only: nvLstmInf links libgdrapi and uses none of GDRCopy's test and benchmark programs, which do not link
# with every toolchain (objects without -fPIC against a default-PIE g++, e.g. g++ 16 on aarch64; the test suite does not
# compile with clang). lib_install below, not install, which would build them again.
execute_process(
    COMMAND ${MAKE_EXECUTABLE} -j${NPROC} prefix=${GDRCOPY_BUILD_DIR} config lib
    WORKING_DIRECTORY ${GDRCOPY_SOURCE_DIR}
    RESULT_VARIABLE BUILD_RESULT
    OUTPUT_VARIABLE BUILD_OUTPUT
    ERROR_VARIABLE BUILD_ERROR)

if(NOT BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "gdrcopy build failed:\nOutput:\n${BUILD_OUTPUT}\nError:\n${BUILD_ERROR}")
else()
    message(STATUS "gdrcopy build output:\n${BUILD_OUTPUT}")
endif()

execute_process(
    COMMAND ${MAKE_EXECUTABLE} prefix=${GDRCOPY_BUILD_DIR} lib_install
    WORKING_DIRECTORY ${GDRCOPY_SOURCE_DIR}
    RESULT_VARIABLE INSTALL_RESULT)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "gdrcopy install failed")
endif()

# Verify library was built
if(NOT EXISTS ${GDRCOPY_LIBRARY})
    message(FATAL_ERROR "gdrcopy library not found at ${GDRCOPY_LIBRARY}")
endif()

message(STATUS "gdrcopy built successfully: ${GDRCOPY_LIBRARY}")

# Create INTERFACE library target (works best with shared libs built at configure time)
add_library(gdrcopy INTERFACE)
target_include_directories(gdrcopy INTERFACE ${GDRCOPY_INCLUDE_DIR})
target_link_libraries(gdrcopy INTERFACE ${GDRCOPY_LIBRARY})
target_link_directories(gdrcopy INTERFACE ${GDRCOPY_LIB_DIR})

# Install gdrcopy relative to CMAKE_INSTALL_PREFIX Default: "../gdrcopy" -> installs to
# ${CMAKE_INSTALL_PREFIX}/../gdrcopy Example: if CMAKE_INSTALL_PREFIX=/app/dl-lowlat-infer -> gdrcopy goes to
# /app/gdrcopy
set(GDRCOPY_INSTALL_DIR
    "../gdrcopy"
    CACHE PATH "Installation directory for gdrcopy (relative to CMAKE_INSTALL_PREFIX)")

# Determine absolute install path for display and RPATH
if(IS_ABSOLUTE ${GDRCOPY_INSTALL_DIR})
    set(GDRCOPY_INSTALL_PATH ${GDRCOPY_INSTALL_DIR})
else()
    # Resolve relative path (handles ../)
    get_filename_component(GDRCOPY_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${GDRCOPY_INSTALL_DIR}" ABSOLUTE)
endif()

# Export for use in other CMakeLists.txt files
set(GDRCOPY_INSTALL_LIB_PATH ${GDRCOPY_INSTALL_PATH}/lib)

install(
    DIRECTORY ${GDRCOPY_BUILD_DIR}/
    DESTINATION ${GDRCOPY_INSTALL_DIR}
    USE_SOURCE_PERMISSIONS)

message(STATUS "GDRCopy will be installed to: ${GDRCOPY_INSTALL_PATH}")
