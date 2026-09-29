#include "vbsr.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <ranges>
#include <stdexcept>
#include <string>

namespace vbsr {
namespace {
// Convert CUDA runtime failures into exceptions at the public API boundary.
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

// Allocate a device array and copy a host vector into it. On copy failure,
// release the allocation before propagating the CUDA error.
template <class T> T* copy_to_device(const std::vector<T>& source) {
  T* destination = nullptr;
  check_cuda(cudaMalloc(&destination, source.size() * sizeof(T)));

  try {
    check_cuda(
        cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
  } catch (...) {
    cudaFree(destination);
    throw;
  }

  return destination;
}

}

// Validate the complete host representation, including its value payload.
void HostMatrix::validate() const { validate_structure(values.size()); }

// Check the CSR block structure, scalar offsets, block shapes, and packed
// payload offsets. This routine is also used to validate device metadata
// after the metadata arrays have been copied to temporary host vectors.
void HostMatrix::validate_structure(size_t value_count) const {

  if (block_rows <= 0 || block_cols <= 0)
    throw std::invalid_argument("positive block dimensions required");
  if (row_ptr.size() != size_t(block_rows) + 1 || row_size.size() != size_t(block_rows) ||
      col_size.size() != size_t(block_cols)) {
    throw std::invalid_argument("metadata size mismatch");
  }

  if (row_scalar_off.size() != size_t(block_rows) + 1 ||
      col_scalar_off.size() != size_t(block_cols) + 1) {
    throw std::invalid_argument("scalar offsets size mismatch");
  }

  if (row_scalar_off.front() != 0 || col_scalar_off.front() != 0) {
    throw std::invalid_argument("scalar offsets must start at zero");
  }

  if (row_ptr.front() != 0 || row_ptr.back() < 0 || size_t(row_ptr.back()) != block_col.size()) {
    throw std::invalid_argument("row_ptr does not match block columns");
  }

  if (value_off.size() != block_col.size() + 1 || value_off.front() != 0 || value_off.back() < 0 ||
      size_t(value_off.back()) != value_count) {
    throw std::invalid_argument("value_off does not match packed values");
  }

  // Concatenate both block-size arrays and enumerate them in one pass. Reset
  // the prefix sum when the traversal moves from row dimensions to columns.
  int64_t expected_offset = 0;
  const auto dimensions = std::views::concat(row_size, col_size);
  for (const auto& [dimension, block_size] : std::views::enumerate(dimensions)) {
    const bool is_row_dimension = dimension < block_rows;
    const auto block_index = is_row_dimension ? dimension : dimension - block_rows;
    const auto& scalar_offsets = is_row_dimension ? row_scalar_off : col_scalar_off;

    if (block_index == 0)
      expected_offset = 0;
    expected_offset += block_size;

    const bool invalid_row_structure =
        is_row_dimension && row_ptr[block_index] > row_ptr[block_index + 1];
    if (invalid_row_structure || block_size < 8 || block_size > 64 || block_size % 8 != 0 ||
        scalar_offsets[block_index + 1] != expected_offset) {
      throw std::invalid_argument(is_row_dimension ? "invalid block-row metadata"
                                                   : "invalid block-column metadata");
    }
  }

  int64_t expected_value_offset = 0;
  for (int block_row = 0; block_row < block_rows; ++block_row) {
    for (int block_index = row_ptr[block_row]; block_index < row_ptr[block_row + 1];
         ++block_index) {
      const int block_column = block_col[block_index];
      if (block_column < 0 || block_column >= block_cols) {
        throw std::invalid_argument("block column out of range");
      }

      if (block_index > row_ptr[block_row] && block_col[block_index - 1] >= block_column) {
        throw std::invalid_argument("block columns must be sorted and unique in each row");
      }

      const int64_t expected_block_values = int64_t(row_size[block_row]) * col_size[block_column];
      expected_value_offset += expected_block_values;
      if (value_off[block_index + 1] != expected_value_offset) {
        throw std::invalid_argument("value_off does not match block payload sizes");
      }
    }
  }
}

// Own device copies of the matrix arrays and prepare a stable ordering that
// groups rows by shape for the specialized RHS=32 launch path.
Matrix::Matrix(const HostMatrix& host) {
  host.validate();
  block_rows_ = host.block_rows;
  block_cols_ = host.block_cols;
  nnzb_ = int(host.block_col.size());
  scalar_rows_ = host.scalar_rows();
  scalar_cols_ = host.scalar_cols();
  value_count_ = host.values.size();

  auto row_shape_order =
      std::views::iota(0, host.block_rows) | std::ranges::to<std::vector<int32_t>>();
  const auto large_begin = std::ranges::stable_partition(
      row_shape_order, [&](int block_row) { return host.row_size[block_row] <= 16; });
  small_row_count_ = int(large_begin.begin() - row_shape_order.begin());
  large_row_count_ = host.block_rows - small_row_count_;

  try {
    row_ptr_ = copy_to_device(host.row_ptr);
    block_col_ = copy_to_device(host.block_col);
    row_size_ = copy_to_device(host.row_size);
    col_size_ = copy_to_device(host.col_size);
    row_scalar_off_ = copy_to_device(host.row_scalar_off);
    col_scalar_off_ = copy_to_device(host.col_scalar_off);
    value_off_ = copy_to_device(host.value_off);
    values_ = copy_to_device(host.values);
    row_shape_order_ = copy_to_device(row_shape_order);
    check_cuda(cudaStreamSynchronize(nullptr));
  } catch (...) {
    release();
    throw;
  }
}

// Free every allocation owned by this object. cudaFree accepts null pointers,
// which keeps this safe for partially constructed and moved-from objects.
void Matrix::release() {
  cudaFree(row_ptr_);
  cudaFree(block_col_);
  cudaFree(row_size_);
  cudaFree(col_size_);
  cudaFree(row_scalar_off_);
  cudaFree(col_scalar_off_);
  cudaFree(value_off_);
  cudaFree(values_);
  cudaFree(row_shape_order_);

  row_ptr_ = nullptr;
  block_col_ = nullptr;
  row_size_ = nullptr;
  col_size_ = nullptr;
  row_scalar_off_ = nullptr;
  col_scalar_off_ = nullptr;
  value_off_ = nullptr;
  values_ = nullptr;
  row_shape_order_ = nullptr;
}

// Destruction releases the device-side matrix storage.
Matrix::~Matrix() { release(); }

// Transfer device ownership and leave `other` in a destructible empty state.
Matrix::Matrix(Matrix&& other) noexcept { *this = std::move(other); }

Matrix& Matrix::operator=(Matrix&& other) noexcept {
  if (this != &other) {
    release();
    block_rows_ = other.block_rows_;
    block_cols_ = other.block_cols_;
    nnzb_ = other.nnzb_;
    small_row_count_ = other.small_row_count_;
    large_row_count_ = other.large_row_count_;
    scalar_rows_ = other.scalar_rows_;
    scalar_cols_ = other.scalar_cols_;
    value_count_ = other.value_count_;
    row_ptr_ = other.row_ptr_;
    block_col_ = other.block_col_;
    row_size_ = other.row_size_;
    col_size_ = other.col_size_;
    row_scalar_off_ = other.row_scalar_off_;
    col_scalar_off_ = other.col_scalar_off_;
    value_off_ = other.value_off_;
    values_ = other.values_;
    row_shape_order_ = other.row_shape_order_;

    other.row_ptr_ = nullptr;
    other.block_col_ = nullptr;
    other.row_size_ = nullptr;
    other.col_size_ = nullptr;
    other.row_scalar_off_ = nullptr;
    other.col_scalar_off_ = nullptr;
    other.value_off_ = nullptr;
    other.values_ = nullptr;
    other.row_shape_order_ = nullptr;
    other.small_row_count_ = 0;
    other.large_row_count_ = 0;
    other.value_count_ = 0;
  }

  return *this;
}

// Expose the owned arrays without transferring ownership.
DeviceMatrix Matrix::device_view() const {
  return {block_rows_,     block_cols_, nnzb_,     scalar_rows_, scalar_cols_,
          row_ptr_,        block_col_,  row_size_, col_size_,    row_scalar_off_,
          col_scalar_off_, value_off_,  values_};
}

// Replace only the numeric payload, preserving the matrix's sparsity pattern.
void Matrix::update_values(const float* source, size_t count, cudaStream_t stream) {
  if (count != value_count_ || (count && (!source || !values_)))
    throw std::invalid_argument("value update size or pointer mismatch");
  if (count && source != values_)
    check_cuda(
        cudaMemcpyAsync(values_, source, count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

// Sum the packed values and all device metadata arrays. The final value
// offset stores the payload length, so read it from device metadata.
size_t Matrix::storage_bytes() const {
  int64_t count = 0;
  check_cuda(cudaMemcpy(&count, value_off_ + nnzb_, sizeof(count), cudaMemcpyDeviceToHost));
  return size_t(count) * sizeof(float) +
         (size_t(block_rows_) * 3 + 1 + block_cols_ + nnzb_) * sizeof(int32_t) +
         (size_t(block_rows_) + block_cols_ + nnzb_ + 3) * sizeof(int64_t);
}

// Reference implementation of Y = A * X using double-precision accumulation.
// The input and output are column-major with `rhs_width` dense columns.
std::vector<float> cpu_reference(const HostMatrix& matrix, std::span<const float> input,
                                 int rhs_width) {
  if (input.size() != size_t(matrix.scalar_cols() * rhs_width)) {
    throw std::invalid_argument("input matrix has incorrect size");
  }

  std::vector<float> output(matrix.scalar_rows() * rhs_width, 0.0f);
  for (int block_row = 0; block_row < matrix.block_rows; ++block_row) {
    const int row_height = matrix.row_size[block_row];
    for (int local_row = 0; local_row < row_height; ++local_row) {
      for (int rhs_column = 0; rhs_column < rhs_width; ++rhs_column) {
        double sum = 0.0;
        for (int block_index = matrix.row_ptr[block_row];
             block_index < matrix.row_ptr[block_row + 1]; ++block_index) {
          const int block_column = matrix.block_col[block_index];
          const int column_width = matrix.col_size[block_column];
          for (int local_column = 0; local_column < column_width; ++local_column) {
            const auto value_index =
                matrix.value_off[block_index] + local_row + local_column * row_height;
            const auto input_index = matrix.col_scalar_off[block_column] + local_column +
                                     int64_t(rhs_column) * matrix.scalar_cols();
            sum += double(matrix.values[value_index]) * input[input_index];
          }
        }

        const auto output_index = matrix.row_scalar_off[block_row] + local_row +
                                  int64_t(rhs_column) * matrix.scalar_rows();
        output[output_index] = float(sum);
      }
    }
  }

  return output;
}

// Preserve the original vector overload while sharing the span implementation.
std::vector<float> cpu_reference(const HostMatrix& matrix, const std::vector<float>& input,
                                 int rhs_width) {
  return cpu_reference(matrix, std::span<const float>{input}, rhs_width);
}
}
