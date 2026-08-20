cmake_minimum_required(VERSION 3.25)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

# CUDA 13.4 does not expose a CUDA26 dialect; device translation units use
# its newest supported standard while host C++ remains strictly C++26.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
find_package(CUDAToolkit REQUIRED)

function(enable_cxx26 target)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/std:c++latest>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++2c>)
  endif()
endfunction()

add_library(varblockspmm
  src/format/vbsr.cpp
  src/format/generator.cpp
  src/kernels/row_owned.cu
  src/baseline/cusparse_baseline.cu
  src/baseline/slot_split.cu
  src/plan/plan.cpp)
target_include_directories(varblockspmm PUBLIC include)
target_link_libraries(varblockspmm PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
enable_cxx26(varblockspmm)
set_target_properties(varblockspmm PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=/wd4996>)
if(MSVC)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(vbsr_tests tests/test_main.cpp)
target_link_libraries(vbsr_tests PRIVATE varblockspmm)
enable_cxx26(vbsr_tests)
add_executable(vbsr_benchmark bench/benchmark.cpp)
target_link_libraries(vbsr_benchmark PRIVATE varblockspmm)
enable_cxx26(vbsr_benchmark)
enable_testing()
add_test(NAME vbsr_tests COMMAND vbsr_tests)
install(TARGETS varblockspmm ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
