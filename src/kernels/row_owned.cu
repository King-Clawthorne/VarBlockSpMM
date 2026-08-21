#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

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

template <int RHS>
__device__ void stage_input_async(DeviceMatrix matrix, const float* __restrict__ input,
                                  int block_index, float* shared_input) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;

  const int block_column = matrix.block_col[block_index];
  const int column_width = matrix.col_size[block_column];
  const int element_count = RHS * column_width;

  // Four-byte copies preserve the 65-float shared stride that avoids bank
  // conflicts for short rows. The copies bypass the register file and become
  // visible after the pipeline wait and following CTA barrier.
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
__global__ void row_owned_single_buffered(
    DeviceMatrix matrix, const int32_t* __restrict__ row_order, const float* __restrict__ input,
    float* __restrict__ output) {
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

// Keep the release kernel's direct block-row mapping for RHS 64. Besides
// avoiding an unnecessary row-list load, preserving this compact variant
// retains its measured register allocation and occupancy.
template <int RHS, int VectorWidth>
__global__ void row_owned_single_buffered_direct(DeviceMatrix matrix,
                                                  const float* __restrict__ input,
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

    // Waiting at the start of the iteration also prevents a fast thread from
    // reusing the previous tile's buffer before slower threads finish reading
    // it. The next copy is then in flight during the current block's FMAs.
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
void launch_ilp(DeviceMatrix matrix, const float* input, float* output, cudaStream_t stream) {
  static_assert(RHS % VectorWidth == 0);
  constexpr int threads = 256;
  constexpr int blocks_per_row = (64 * (RHS / VectorWidth) + threads - 1) / threads;
  row_owned_ilp<RHS, VectorWidth>
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

  // At low mean degree the second launch costs more than shape separation
  // saves for mixed-height matrices. Use the original one-CTA-per-row path in
  // that regime; homogeneous matrices still get the fitting one-launch path.
  if (small_row_count != 0 && large_row_count != 0 &&
      matrix.nnzb < 8 * matrix.block_rows) {
    launch_single_buffered<RHS, VectorWidth>(matrix, input, output, stream);
    return;
  }

  if (small_row_count != 0) {
    row_owned_single_buffered<RHS, VectorWidth, 128, true>
        <<<small_row_count, 128, 0, stream>>>(matrix, row_shape_order, input, output);
  }
  if (large_row_count != 0) {
    row_owned_double_buffered<RHS, VectorWidth, 256>
        <<<large_row_count, 256, 0, stream>>>(matrix, row_shape_order + small_row_count, input,
                                              output);
  }
}

template <int RHS, int VectorWidth>
void launch_single_buffered(DeviceMatrix matrix, const float* input, float* output,
                            cudaStream_t stream) {
  row_owned_single_buffered_direct<RHS, VectorWidth>
      <<<matrix.block_rows, 256, 0, stream>>>(matrix, input, output);
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

void launch_row_owned(DeviceMatrix matrix, const int32_t* row_shape_order, int small_row_count,
                      int large_row_count, const float* input, float* output, int rhs_width,
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
    launch_shape_dispatched<32, 8>(matrix, row_shape_order, small_row_count, large_row_count, input,
                                   output, stream);
    break;
  case 64:
    // Two complete RHS-64 buffers reduce occupancy enough to outweigh copy /
    // compute overlap on the release GPU, so retain the measured single tile.
    launch_single_buffered<64, 16>(matrix, input, output, stream);
    break;
  default:
    throw std::invalid_argument("unsupported rhs width");
  }
  check_kernel_launch();
}

} // namespace vbsr
