# Compile the upstream variable-batch SGEMM dispatcher and its runtime support.
# This avoids unrelated LAPACK/Fortran solvers. No MAGMA GEMM kernel is modified.
set(MAGMA_HAVE_CUDA ON)
set(MAGMA_CUDA_ARCH "native")
set(MAGMA_CUDA_ARCH_MIN 1200)
configure_file(${VARBLOCKSPMM_MAGMA_SOURCE}/include/magma_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/magma-config/magma_config.h)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/magma-config/magma_mangling_cmake.h "/* No Fortran routines in the GEMM comparison. */\n")
add_library(vbsr_magma STATIC
  ${VARBLOCKSPMM_MAGMA_SOURCE}/magmablas/sgemm_vbatched_core.cu
  ${VARBLOCKSPMM_MAGMA_SOURCE}/interface_cuda/interface.cpp
  ${VARBLOCKSPMM_MAGMA_SOURCE}/interface_cuda/alloc.cpp
  ${VARBLOCKSPMM_MAGMA_SOURCE}/interface_cuda/error.cpp
  ${VARBLOCKSPMM_MAGMA_SOURCE}/control/constants.cpp)
target_sources(vbsr_magma PRIVATE ${VARBLOCKSPMM_MAGMA_SOURCE}/control/auxiliary.cpp)
if(WIN32)
  target_sources(vbsr_magma PRIVATE ${VARBLOCKSPMM_MAGMA_SOURCE}/control/magma_winthread.cpp)
endif()
target_include_directories(vbsr_magma PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/magma-config
  ${VARBLOCKSPMM_MAGMA_SOURCE}/include ${VARBLOCKSPMM_MAGMA_SOURCE}/control
  ${VARBLOCKSPMM_MAGMA_SOURCE}/magmablas)
target_compile_definitions(vbsr_magma PUBLIC ADD_)
target_link_libraries(vbsr_magma PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
if(MSVC)
  target_compile_options(vbsr_magma PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Gy>)
endif()
add_library(vbsr_magma_comparison STATIC bench/magma.cu)
target_link_libraries(vbsr_magma_comparison PUBLIC vbsr_magma vbsr_benchmark_support)
target_include_directories(vbsr_magma_comparison PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bench)
add_executable(vbsr_relevance bench/relevance.cpp)
target_link_libraries(vbsr_relevance PRIVATE vbsr_magma_comparison)
add_executable(vbsr_transport bench/transport.cpp)
target_link_libraries(vbsr_transport PRIVATE vbsr_magma_comparison)
add_executable(vbsr_updates bench/updates.cpp)
target_link_libraries(vbsr_updates PRIVATE vbsr_benchmark_support)
if(BUILD_TESTING)
  target_compile_definitions(vbsr_tests PRIVATE VBSR_ENABLE_MAGMA_TEST)
  target_link_libraries(vbsr_tests PRIVATE vbsr_magma_comparison)
  # Both cases exceed the timing harness's two-million-value sampling cutoff.
  add_test(NAME vbsr_relevance_full_uniform COMMAND vbsr_relevance
    512 8 64 32 1 2 1 none random 1 --verify-only)
  add_test(NAME vbsr_relevance_full_mixed COMMAND vbsr_relevance
    512 16 32 -1 2 2 2 none random 1 --verify-only)
  set_tests_properties(vbsr_relevance_full_uniform vbsr_relevance_full_mixed
    PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE)
endif()
