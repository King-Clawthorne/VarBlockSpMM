cmake_minimum_required(VERSION 3.25)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

# Require standard C++20 modes for both host and CUDA translation units.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 12.6 REQUIRED)

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
target_compile_features(varblockspmm PUBLIC cxx_std_20)
target_link_libraries(varblockspmm PUBLIC CUDA::cudart)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Keep compiler warning options scoped to the language and toolchain in use.
if(MSVC)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4996>)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()
