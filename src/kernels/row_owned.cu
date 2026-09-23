#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "varblockspmm/vbsr.hpp"

namespace vbsr {
namespace {

// A warp covers RowsPerWarp adjacent rows and splits each dot product across
// the remaining lanes. The final warp reduction keeps complete row ownership.
template <int RHS, int RowsPerWarp, int Threads>
__global__ void row_owned_warp_reduced(DeviceMatrix matrix, const float* __restrict__ input,
                                       float* __restrict__ output) {
  constexpr int stripe_count = 32 / RowsPerWarp;
  constexpr int rows_per_cta = Threads / 32 * RowsPerWarp;
  constexpr int stride = 65;
  __shared__ float tile[RHS * stride];
  const int row = blockIdx.x;
  const int height = matrix.row_size[row];
  const int lane = threadIdx.x % 32;
  const int stripe = lane / RowsPerWarp;
  for (int base = 0; base < height; base += rows_per_cta) {
    const int local_row = base + (threadIdx.x / 32) * RowsPerWarp + lane % RowsPerWarp;
    const bool active = local_row < height;
    float sums[RHS] = {};
    for (int block = matrix.row_ptr[row]; block < matrix.row_ptr[row + 1]; ++block) {
      const int col = matrix.block_col[block];
      const int width = matrix.col_size[col];
      for (int i = threadIdx.x; i < RHS * 64; i += Threads) {
        const int k = i % 64;
        const int q = i / 64;
        if (k < width)
          tile[q * stride + k] = input[matrix.col_scalar_off[col] + k + int64_t(q) * matrix.scalar_cols];
      }
      __syncthreads();
      if (active) {
        const float* values = matrix.values + matrix.value_off[block] + local_row;
#pragma unroll 2
        for (int k = stripe; k < width; k += stripe_count) {
          const float a = values[k * height];
#pragma unroll
          for (int q = 0; q < RHS; ++q) sums[q] = fmaf(a, tile[q * stride + k], sums[q]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < RHS; ++q) {
#pragma unroll
      for (int delta = RowsPerWarp; delta < 32; delta *= 2)
        sums[q] += __shfl_xor_sync(0xffffffffu, sums[q], delta);
      if (active && stripe == 0)
        output[matrix.row_scalar_off[row] + local_row + int64_t(q) * matrix.scalar_rows] = sums[q];
    }
  }
}

template <int VectorWidth, int Threads>
__global__ void row_owned_rhs8_tiled(DeviceMatrix matrix, const float* __restrict__ input,
                                     float* __restrict__ output) {
  const int row = blockIdx.x;
  const int height = matrix.row_size[row];
  // Groups of eight contiguous lanes follow the format's minimum row extent.
  for (int tile = blockIdx.y * Threads + threadIdx.x; tile < height * (8 / VectorWidth);
       tile += Threads * gridDim.y) {
    const int local_row = (tile / (64 / VectorWidth)) * 8 + tile % 8;
    const int first_rhs = ((tile / 8) % (8 / VectorWidth)) * VectorWidth;
    float sums[VectorWidth] = {};
    for (int block = matrix.row_ptr[row]; block < matrix.row_ptr[row + 1]; ++block) {
      const int column = matrix.block_col[block];
      const int width = matrix.col_size[column];
      const float* a = matrix.values + matrix.value_off[block] + local_row;
      const float* b = input + matrix.col_scalar_off[column] + int64_t(first_rhs) * matrix.scalar_cols;
#pragma unroll 4
      for (int k = 0; k < width; ++k) {
        const float value = a[k * height];
#pragma unroll
        for (int v = 0; v < VectorWidth; ++v)
          sums[v] = fmaf(value, b[k + int64_t(v) * matrix.scalar_cols], sums[v]);
      }
    }
#pragma unroll
    for (int v = 0; v < VectorWidth; ++v)
      output[matrix.row_scalar_off[row] + local_row + int64_t(first_rhs + v) * matrix.scalar_rows] = sums[v];
  }
}

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
  if (small_row_count != 0 && large_row_count != 0 && matrix.nnzb < int64_t(8) * matrix.block_rows) {
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

// Experimental controls keep execution policies explicit. They are never selected
// by the public Plan dispatch. RHS-8 variants form a 2x2x2 factorial design.
void launch_ablation(DeviceMatrix matrix, const int32_t* order, const float* input,
                     float* output, int rhs, int variant, cudaStream_t stream) {
  if ((rhs == 8 && (variant == 8 || variant == 9)) ||
      (rhs == 16 && (variant == 5 || variant == 6)) ||
      ((rhs == 8 || rhs == 16) && variant == 200)) {
    if (variant == 200) {
      if (rhs == 16) launch_ilp<16, 8>(matrix, input, output, stream);
      else if (matrix.block_rows < 512) launch_scalar<8>(matrix, input, output, stream);
      else row_owned_rhs8_tiled<2, 256><<<matrix.block_rows, 256, 0, stream>>>(matrix, input, output);
    } else if (rhs == 8) {
      if (variant == 8) row_owned_ilp<8, 8><<<matrix.block_rows, 64, 0, stream>>>(matrix, input, output);
      else row_owned_single_buffered<8, 8, 64, false><<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
    } else {
      if (variant == 5) row_owned_ilp<16, 16><<<matrix.block_rows, 64, 0, stream>>>(matrix, input, output);
      else row_owned_single_buffered<16, 16, 64, false><<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
    }
    check_kernel_launch();
    return;
  }
  if (variant >= 100 && (rhs == 8 || rhs == 16)) {
#define NARROW_CASE(N) \
    case N: \
      switch (variant) { \
      case 100: launch_single_buffered<N, N / 2>(matrix, input, output, stream); break; \
      case 101: row_owned_warp_reduced<N, 8, 256><<<matrix.block_rows, 256, 0, stream>>>(matrix, input, output); break; \
      case 102: row_owned_warp_reduced<N, 16, 128><<<matrix.block_rows, 128, 0, stream>>>(matrix, input, output); break; \
      case 103: row_owned_warp_reduced<N, 8, 128><<<matrix.block_rows, 128, 0, stream>>>(matrix, input, output); break; \
      case 104: row_owned_warp_reduced<N, 4, 256><<<matrix.block_rows, 256, 0, stream>>>(matrix, input, output); break; \
      case 105: row_owned_single_buffered<N, N / 2, 128, false><<<matrix.block_rows, 128, 0, stream>>>(matrix, nullptr, input, output); break; \
      case 106: row_owned_single_buffered<N, N / 4, 256, false><<<matrix.block_rows, 256, 0, stream>>>(matrix, nullptr, input, output); break; \
      case 107: row_owned_single_buffered<N, N, 64, false><<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output); break; \
      default: throw std::invalid_argument("invalid tuning variant"); \
      } break
    switch (rhs) { NARROW_CASE(8); NARROW_CASE(16); }
#undef NARROW_CASE
    check_kernel_launch();
    return;
  }
  if (rhs == 8 && variant >= 0 && variant < 8) {
    const int ctas = (variant & 1) + 1;
    const bool reuse = variant & 2;
    const bool tiled = variant & 4;
    const dim3 grid(matrix.block_rows, ctas);
    if (tiled) {
      if (reuse) row_owned_rhs8_tiled<2, 256><<<grid, 256, 0, stream>>>(matrix, input, output);
      else row_owned_rhs8_tiled<1, 256><<<grid, 256, 0, stream>>>(matrix, input, output);
    } else {
      if (reuse) row_owned_ilp<8, 2><<<grid, 256, 0, stream>>>(matrix, input, output);
      else row_owned_ilp<8, 1><<<grid, 256, 0, stream>>>(matrix, input, output);
    }
  } else {
    // v0/v1 isolate accumulator width within the global-load family.
    // v1/v2 isolate staging with the same CTA size, output mapping, and width.
    // v3/v4 isolate the buffer schedule with the same indirect row order.
#define WIDE_CASE(N, V) \
    case N: \
      switch (variant) { \
      case 0: row_owned_ilp<N, 4><<<matrix.block_rows, 256, 0, stream>>>(matrix, input, output); break; \
      case 1: launch_ilp<N, V>(matrix, input, output, stream); break; \
      case 2: launch_single_buffered<N, V>(matrix, input, output, stream); break; \
      case 3: row_owned_single_buffered<N, V, 256, true><<<matrix.block_rows, 256, 0, stream>>>(matrix, order, input, output); break; \
      case 4: row_owned_double_buffered<N, V, 256><<<matrix.block_rows, 256, 0, stream>>>(matrix, order, input, output); break; \
      default: throw std::invalid_argument("invalid ablation variant"); \
      } break
    switch (rhs) {
      WIDE_CASE(16, 8);
      WIDE_CASE(32, 8);
      WIDE_CASE(64, 16);
    default: throw std::invalid_argument("invalid ablation RHS");
    }
#undef WIDE_CASE
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
    // One thread owns a complete panel row. Sixty-four threads cover every
    // supported height while input staging shares B across the active rows.
    row_owned_single_buffered<8, 8, 64, false><<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
    break;
  case 16:
    row_owned_single_buffered<16, 16, 64, false><<<matrix.block_rows, 64, 0, stream>>>(matrix, nullptr, input, output);
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
