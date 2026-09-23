#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "vbsr.hpp"

namespace vbsr {
namespace {

template <int RHS>
__global__ void row_owned_scalar(DeviceMatrix matrix, const float* __restrict__ input,
                                 float* __restrict__ output) {

  const int block_row = blockIdx.x;
  const int row_height = matrix.row_size[block_row];
  const int grid_stride = blockDim.x * gridDim.y;

  for (int work_index = blockIdx.y * blockDim.x + threadIdx.x; work_index < row_height * RHS;
       work_index += grid_stride) {
    const int local_row = work_index % row_height;
    const int rhs_column = work_index / row_height;
    float accumulator = 0.0f;

#pragma unroll 1
    for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1];
         ++block_index) {
      const int block_column = matrix.block_col[block_index];
      const int column_width = matrix.col_size[block_column];
      const float* block_values = matrix.values + matrix.value_off[block_index] + local_row;
      const float* input_column =
          input + matrix.col_scalar_off[block_column] + int64_t(rhs_column) * matrix.scalar_cols;

#pragma unroll 4
      for (int local_column = 0; local_column < column_width; ++local_column) {
        accumulator =
            fmaf(block_values[local_column * row_height], input_column[local_column], accumulator);
      }
    }

    output[matrix.row_scalar_off[block_row] + local_row +
           int64_t(rhs_column) * matrix.scalar_rows] = accumulator;
  }
}

template <int RHS>
__device__ void stage_input_async(DeviceMatrix matrix, const float* __restrict__ input,
                                  int block_index, float* shared_input) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;

  const int block_column = matrix.block_col[block_index];
  const int column_width = matrix.col_size[block_column];
  const int element_count = RHS * column_width;

  for (int element = threadIdx.x; element < element_count; element += blockDim.x) {
    const int rhs_column = element / column_width;
    const int local_column = element - rhs_column * column_width;
    const float* source = input + matrix.col_scalar_off[block_column] + local_column +
                          int64_t(rhs_column) * matrix.scalar_cols;
    __pipeline_memcpy_async(shared_input + rhs_column * shared_stride + local_column, source,
                            sizeof(float));
  }
  __pipeline_commit();
}

template <int RHS, int VectorWidth, int Threads, bool IndirectRows>
__global__ void
row_owned_single_buffered(DeviceMatrix matrix, const int32_t* __restrict__ row_order,
                          const float* __restrict__ input, float* __restrict__ output) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;
  __shared__ float shared_input[RHS * shared_stride];

  const int block_row = IndirectRows ? row_order[blockIdx.x] : int(blockIdx.x);
  const int row_height = matrix.row_size[block_row];
  const int rhs_groups = RHS / VectorWidth;
  const int work_index = threadIdx.x;
  const bool active = work_index < row_height * rhs_groups;
  const int local_row = work_index % row_height;
  const int first_rhs_column = (work_index / row_height) * VectorWidth;
  const int block_end = matrix.row_ptr[block_row + 1];
  float accumulators[VectorWidth] = {};

#pragma unroll 1
  for (int block_index = matrix.row_ptr[block_row]; block_index < block_end; ++block_index) {
    const int block_column = matrix.block_col[block_index];
    const int column_width = matrix.col_size[block_column];
    const int local_column = threadIdx.x % max_block_width;

    for (int rhs_column = threadIdx.x / max_block_width; rhs_column < RHS;
         rhs_column += Threads / max_block_width) {
      if (local_column < column_width) {
        shared_input[rhs_column * shared_stride + local_column] =
            input[matrix.col_scalar_off[block_column] + local_column +
                  int64_t(rhs_column) * matrix.scalar_cols];
      }
    }
    __syncthreads();

    if (active) {
      const float* block_values = matrix.values + matrix.value_off[block_index] + local_row;
#pragma unroll 4
      for (int column = 0; column < column_width; ++column) {
        const float matrix_value = block_values[column * row_height];
#pragma unroll
        for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
          accumulators[vector_index] =
              fmaf(matrix_value,
                   shared_input[(first_rhs_column + vector_index) * shared_stride + column],
                   accumulators[vector_index]);
        }
      }
    }
    if (block_index + 1 < block_end) {
      __syncthreads();
    }
  }

  if (active) {
#pragma unroll
    for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
      output[matrix.row_scalar_off[block_row] + local_row +
             int64_t(first_rhs_column + vector_index) * matrix.scalar_rows] =
          accumulators[vector_index];
    }
  }
}

template <int RHS, int VectorWidth, int Threads>
__global__ __launch_bounds__(Threads) void row_owned_double_buffered(
    DeviceMatrix matrix, const int32_t* __restrict__ row_order, const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;
  __shared__ float shared_input[2][RHS * shared_stride];

  const int block_row = row_order[blockIdx.x];
  const int row_height = matrix.row_size[block_row];
  const int rhs_groups = RHS / VectorWidth;
  const int work_index = threadIdx.x;
  const bool active = work_index < row_height * rhs_groups;
  const int local_row = work_index % row_height;
  const int first_rhs_column = (work_index / row_height) * VectorWidth;
  const int block_begin = matrix.row_ptr[block_row];
  const int block_end = matrix.row_ptr[block_row + 1];
  float accumulators[VectorWidth] = {};

  if (block_begin < block_end) {
    stage_input_async<RHS>(matrix, input, block_begin, shared_input[0]);
  }

#pragma unroll 1
  for (int block_index = block_begin; block_index < block_end; ++block_index) {
    const int current_buffer = (block_index - block_begin) & 1;
    const int next_block = block_index + 1;

    __pipeline_wait_prior(0);
    __syncthreads();
    if (next_block < block_end) {
      stage_input_async<RHS>(matrix, input, next_block, shared_input[current_buffer ^ 1]);
    }

    const int block_column = matrix.block_col[block_index];
    const int column_width = matrix.col_size[block_column];

    if (active) {
      const float* block_values = matrix.values + matrix.value_off[block_index] + local_row;
#pragma unroll 4
      for (int column = 0; column < column_width; ++column) {
        const float matrix_value = block_values[column * row_height];
#pragma unroll
        for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
          accumulators[vector_index] =
              fmaf(matrix_value,
                   shared_input[current_buffer]
                               [(first_rhs_column + vector_index) * shared_stride + column],
                   accumulators[vector_index]);
        }
      }
    }
  }

  if (active) {
#pragma unroll
    for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
      output[matrix.row_scalar_off[block_row] + local_row +
             int64_t(first_rhs_column + vector_index) * matrix.scalar_rows] =
          accumulators[vector_index];
    }
  }
}

template <int RHS>
void launch_scalar(DeviceMatrix matrix, const float* input, float* output, cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int blocks_per_row = (64 * RHS + threads - 1) / threads;
  row_owned_scalar<RHS>
      <<<dim3(matrix.block_rows, blocks_per_row), threads, 0, stream>>>(matrix, input, output);
}

template <int RHS, int VectorWidth>
void launch_single_buffered(DeviceMatrix matrix, const float* input, float* output,
                            cudaStream_t stream);

template <int RHS, int VectorWidth>
void launch_shape_dispatched(DeviceMatrix matrix, const int32_t* row_shape_order,
                             int small_row_count, int large_row_count, const float* input,
                             float* output, cudaStream_t stream) {
  static_assert(RHS % VectorWidth == 0);
  static_assert(RHS / VectorWidth == 4);

  if (small_row_count != 0 && large_row_count != 0 &&
      matrix.nnzb < int64_t(8) * matrix.block_rows) {
    launch_single_buffered<RHS, VectorWidth>(matrix, input, output, stream);
    return;
  }

  if (small_row_count != 0) {
    row_owned_single_buffered<RHS, VectorWidth, 128, true>
        <<<small_row_count, 128, 0, stream>>>(matrix, row_shape_order, input, output);
  }
  if (large_row_count != 0) {
    row_owned_double_buffered<RHS, VectorWidth, 256><<<large_row_count, 256, 0, stream>>>(
        matrix, row_shape_order + small_row_count, input, output);
  }
}

template <int RHS, int VectorWidth>
void launch_single_buffered(DeviceMatrix matrix, const float* input, float* output,
                            cudaStream_t stream) {
  row_owned_single_buffered<RHS, VectorWidth, 256, false>
      <<<matrix.block_rows, 256, 0, stream>>>(matrix, nullptr, input, output);
}

void check_kernel_launch() {
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

}
void launch_row_owned(const DeviceMatrix& matrix, const int32_t* row_shape_order,
                      int small_row_count, int large_row_count, const float* input, float* output,
                      int rhs_width, cudaStream_t stream) {

  switch (rhs_width) {
  case 8:

    row_owned_single_buffered<8, 8, 64, false>
        <<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
    break;
  case 16:
    row_owned_single_buffered<16, 16, 64, false>
        <<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
    break;
  case 32:
    launch_shape_dispatched<32, 8>(matrix, row_shape_order, small_row_count, large_row_count, input,
                                   output, stream);
    break;
  case 64:

    launch_single_buffered<64, 16>(matrix, input, output, stream);
    break;
  default:
    throw std::invalid_argument("unsupported rhs width");
  }
  check_kernel_launch();
}

}
