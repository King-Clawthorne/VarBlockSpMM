cmake_minimum_required(VERSION 3.25)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 12.6 REQUIRED)

add_library(varblockspmm
  src/format/vbsr.cpp
  src/format/generator.cpp
  src/kernels/row_owned.cu
  src/plan/plan.cpp)
add_library(VarBlockSpMM::varblockspmm ALIAS varblockspmm)
target_include_directories(varblockspmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(varblockspmm PUBLIC cxx_std_20)
target_link_libraries(varblockspmm PUBLIC CUDA::cudart)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

if(MSVC)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4996>)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()
