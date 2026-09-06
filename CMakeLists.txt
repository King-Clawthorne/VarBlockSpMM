cmake_minimum_required(VERSION 3.25)
project(VarBlockSpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# CUDA 12.6 includes both grouped GEMM and generic BSR SpMM. Measurements use 13.4.
find_package(CUDAToolkit 12.6 REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
option(VARBLOCKSPMM_BUILD_BENCHMARKS "Build research benchmark executables" ON)

add_library(varblockspmm
  src/format/vbsr.cpp
  src/format/generator.cpp
  src/kernels/row_owned.cu
  src/baseline/cusparse_baseline.cu
  src/baseline/slot_split.cu
  src/plan/plan.cpp)
add_library(VarBlockSpMM::varblockspmm ALIAS varblockspmm)
target_include_directories(varblockspmm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(varblockspmm PUBLIC cxx_std_20)
target_link_libraries(varblockspmm PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(varblockspmm PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
if(MSVC)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4996>)
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(varblockspmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

if(BUILD_TESTING)
add_executable(vbsr_tests tests/test_main.cpp)
target_link_libraries(vbsr_tests PRIVATE varblockspmm)
add_test(NAME vbsr_tests COMMAND vbsr_tests)
endif()

if(VARBLOCKSPMM_BUILD_BENCHMARKS)
add_executable(vbsr_benchmark bench/benchmark.cpp)
target_link_libraries(vbsr_benchmark PRIVATE varblockspmm)
add_library(vbsr_benchmark_support STATIC
  bench/support.cpp
  bench/application_data.cpp
  bench/dense.cpp
  bench/compact_csr.cpp)
target_link_libraries(vbsr_benchmark_support PUBLIC varblockspmm)
add_executable(vbsr_audit bench/audit.cpp)
target_link_libraries(vbsr_audit PRIVATE vbsr_benchmark_support)
add_executable(vbsr_application_audit bench/application.cpp)
target_link_libraries(vbsr_application_audit PRIVATE vbsr_benchmark_support)
add_executable(vbsr_kernel_audit bench/kernel.cpp)
target_link_libraries(vbsr_kernel_audit PRIVATE vbsr_benchmark_support)
add_executable(vbsr_ablation bench/ablation.cpp)
target_link_libraries(vbsr_ablation PRIVATE vbsr_benchmark_support)
endif()

configure_package_config_file(cmake/VarBlockSpMMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/VarBlockSpMMConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VarBlockSpMM)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/VarBlockSpMMConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(TARGETS varblockspmm EXPORT VarBlockSpMMTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(EXPORT VarBlockSpMMTargets NAMESPACE VarBlockSpMM::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VarBlockSpMM)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VarBlockSpMMConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/VarBlockSpMMConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VarBlockSpMM)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/VarBlockSpMM)
