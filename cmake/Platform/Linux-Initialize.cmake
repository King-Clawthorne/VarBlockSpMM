# CMake 4.2 knows CUDA C++23, but its NVIDIA CUDA compiler module does not
# register the standard for NVCC. CUDA Toolkit 13.3 added device-side C++23
# support, so provide the missing CMake dialect mapping for supported toolkits.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA23_STANDARD__HAS_FULL_SUPPORT ON)
endif()
