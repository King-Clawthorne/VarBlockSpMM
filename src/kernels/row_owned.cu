#include <cuda_runtime.h>

#include <stdexcept>

#include "varblockspmm/vbsr.hpp"

namespace vbsr {
namespace {

template <int RHS>
__global__ void row_owned_scalar(DeviceMatrix matrix, const float* __restrict__ input,
                                 float* __restrict__ output) {
  // One x-grid block owns one block row. The y-grid and threads stride over
  // every (local row, RHS column) output element, so no atomics are required.
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

template <int RHS, int VectorWidth>
__global__ void row_owned_ilp(DeviceMatrix matrix, const float* __restrict__ input,
                              float* __restrict__ output) {
  // Each thread reuses one A value across VectorWidth independent RHS columns.
  // This increases instruction-level parallelism and reduces repeated A loads.
  const int block_row = blockIdx.x;
  const int row_height = matrix.row_size[block_row];
  const int rhs_groups = RHS / VectorWidth;
  const int grid_stride = blockDim.x * gridDim.y;

  for (int work_index = blockIdx.y * blockDim.x + threadIdx.x; work_index < row_height * rhs_groups;
       work_index += grid_stride) {
    const int local_row = work_index % row_height;
    const int first_rhs_column = (work_index / row_height) * VectorWidth;
    float accumulators[VectorWidth] = {};

#pragma unroll 1
    for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1];
         ++block_index) {
      const int block_column = matrix.block_col[block_index];
      const int column_width = matrix.col_size[block_column];
      const float* block_values = matrix.values + matrix.value_off[block_index] + local_row;
      const float* first_input_column = input + matrix.col_scalar_off[block_column] +
                                        int64_t(first_rhs_column) * matrix.scalar_cols;

#pragma unroll 4
      for (int local_column = 0; local_column < column_width; ++local_column) {
        const float matrix_value = block_values[local_column * row_height];
#pragma unroll
        for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
          const auto input_index = local_column + int64_t(vector_index) * matrix.scalar_cols;
          accumulators[vector_index] =
              fmaf(matrix_value, first_input_column[input_index], accumulators[vector_index]);
        }
      }
    }

#pragma unroll
    for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
      output[matrix.row_scalar_off[block_row] + local_row +
             int64_t(first_rhs_column + vector_index) * matrix.scalar_rows] =
          accumulators[vector_index];
    }
  }
}

template <int RHS, int VectorWidth>
__global__ void row_owned_staged(DeviceMatrix matrix, const float* __restrict__ input,
                                 float* __restrict__ output) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;
  __shared__ float shared_input[RHS * shared_stride];

  const int block_row = blockIdx.x;
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

    // Four 64-thread cohorts cooperatively load coalesced columns. Padding the
    // shared stride avoids conflicts when a warp spans multiple RHS groups.
    for (int rhs_column = threadIdx.x / max_block_width; rhs_column < RHS; rhs_column += 4) {
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

template <int RHS>
void launch_scalar(DeviceMatrix matrix, const float* input, float* output, cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int blocks_per_row = (64 * RHS + threads - 1) / threads;
  row_owned_scalar<RHS>
      <<<dim3(matrix.block_rows, blocks_per_row), threads, 0, stream>>>(matrix, input, output);
}

template <int RHS, int VectorWidth>
void launch_ilp(DeviceMatrix matrix, const float* input, float* output, cudaStream_t stream) {
  static_assert(RHS % VectorWidth == 0);
  constexpr int threads = 256;
  constexpr int blocks_per_row = (64 * (RHS / VectorWidth) + threads - 1) / threads;
  row_owned_ilp<RHS, VectorWidth>
      <<<dim3(matrix.block_rows, blocks_per_row), threads, 0, stream>>>(matrix, input, output);
}

template <int RHS, int VectorWidth>
void launch_staged(DeviceMatrix matrix, const float* input, float* output, cudaStream_t stream) {
  static_assert(RHS % VectorWidth == 0);
  static_assert(64 * (RHS / VectorWidth) <= 256);
  constexpr int threads = 256;
  row_owned_staged<RHS, VectorWidth>
      <<<matrix.block_rows, threads, 0, stream>>>(matrix, input, output);
}

void check_kernel_launch() {
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

} // namespace

void launch_row_owned_scalar(DeviceMatrix matrix, const float* input, float* output, int rhs_width,
                             cudaStream_t stream) {
  switch (rhs_width) {
  case 8:
    launch_scalar<8>(matrix, input, output, stream);
    break;
  case 16:
    launch_scalar<16>(matrix, input, output, stream);
    break;
  case 32:
    launch_scalar<32>(matrix, input, output, stream);
    break;
  case 64:
    launch_scalar<64>(matrix, input, output, stream);
    break;
  default:
    throw std::invalid_argument("unsupported rhs width");
  }
  check_kernel_launch();
}

void launch_row_owned(DeviceMatrix matrix, const float* input, float* output, int rhs_width,
                      cudaStream_t stream) {
  // Wider panels amortize each A load across more independent output columns.
  // The selected widths retain enough threads per row to cover latency; going
  // wider than these measured points loses more parallelism than it saves.
  switch (rhs_width) {
  case 8:
    launch_scalar<8>(matrix, input, output, stream);
    break;
  case 16:
    launch_ilp<16, 8>(matrix, input, output, stream);
    break;
  case 32:
    launch_staged<32, 8>(matrix, input, output, stream);
    break;
  case 64:
    launch_staged<64, 16>(matrix, input, output, stream);
    break;
  default:
    throw std::invalid_argument("unsupported rhs width");
  }
  check_kernel_launch();
}

} // namespace vbsr
