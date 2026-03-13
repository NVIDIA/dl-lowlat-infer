include(FetchContent)

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        v2.10.0
)

FetchContent_GetProperties(googletest)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()