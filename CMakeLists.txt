cmake_minimum_required(VERSION 3.24)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

add_library(varblockspmm
  src/format/vbsr.cpp
  src/format/generator.cpp
  src/kernels/row_owned.cu
  src/baseline/cusparse_baseline.cu
  src/baseline/slot_split.cu
  src/plan/plan.cpp)
target_include_directories(varblockspmm PUBLIC include)
target_link_libraries(varblockspmm PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(varblockspmm PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=/wd4996>)
if(MSVC)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(vbsr_tests tests/test_main.cpp)
target_link_libraries(vbsr_tests PRIVATE varblockspmm)
add_executable(vbsr_benchmark bench/benchmark.cpp)
target_link_libraries(vbsr_benchmark PRIVATE varblockspmm)
enable_testing()
add_test(NAME vbsr_tests COMMAND vbsr_tests)
install(TARGETS varblockspmm ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
