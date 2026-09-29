# CMake 4.2's NVIDIA CUDA module does not register CUDA C++23, even though
# NVCC 13.3 and newer support the dialect. Add the missing standard mapping
# before project() enables CUDA so CUDA_STANDARD 23 is handled natively.
set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
set(CMAKE_CUDA23_STANDARD__HAS_FULL_SUPPORT ON)
