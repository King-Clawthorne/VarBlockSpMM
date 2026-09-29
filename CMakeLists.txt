cmake_minimum_required(VERSION 3.25)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

if(MSVC)
  message(FATAL_ERROR
    "VarBlockSpMM requires host C++26 and CUDA C++23. The MSVC/NVCC toolchain "
    "cannot compile CUDA C++23; configure this project with Linux/WSL2, GCC "
    "or Clang, and CUDA Toolkit 13.3 or newer.")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# CMake otherwise inherits the C++26 requirement for the CUDA language.
# Keep its CUDA baseline at C++20, then override it with NVCC's C++23 flag
# where the non-MSVC host compiler supports that mode.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 13.3 REQUIRED)

# Build the public library from host-side storage, planning, generation, and
# CUDA kernel implementation sources.
add_library(varblockspmm
  src/vbsr.cpp
  src/generator.cpp
  src/row_owned.cu
  src/plan.cpp)
add_library(VarBlockSpMM::varblockspmm ALIAS varblockspmm)
# Export the public header location and CUDA runtime dependency to consumers.
target_include_directories(varblockspmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# CUDA sources share this target with host sources. Keep the host dialect
# scoped to C++ translation units rather than propagating it to CUDA.
set_property(TARGET varblockspmm PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
target_link_libraries(varblockspmm PUBLIC CUDA::cudart)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
# CMake does not yet identify CUDA C++23 for NVCC. Select it directly for
# device compilation and use C++26 for GCC's host pass of the .cu file.
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++23>)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++2c>)

# Keep compiler warnings scoped to host C++ translation units.
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
