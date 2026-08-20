#include "varblockspmm/vbsr.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace vbsr {
namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <class T> T* copy_to_device(const std::vector<T>& source) {
  T* destination = nullptr;
  check_cuda(cudaMalloc(&destination, source.size() * sizeof(T)));
  check_cuda(
      cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
  return destination;
}

} // namespace

void HostMatrix::validate() const {
  // Validate the structure from coarse dimensions down to individual block
  // payloads.
  if (block_rows <= 0 || block_cols <= 0) {
    throw std::invalid_argument("positive block dimensions required");
  }
  if (row_ptr.size() != size_t(block_rows + 1) || row_size.size() != size_t(block_rows) ||
      col_size.size() != size_t(block_cols)) {
    throw std::invalid_argument("metadata size mismatch");
  }
  if (row_scalar_off.size() != size_t(block_rows + 1) ||
      col_scalar_off.size() != size_t(block_cols + 1)) {
    throw std::invalid_argument("scalar offsets size mismatch");
  }
  if (row_ptr.front() != 0 || row_ptr.back() < 0 || size_t(row_ptr.back()) != block_col.size()) {
    throw std::invalid_argument("bad row_ptr");
  }
  if (value_off.size() != block_col.size() + 1 || value_off.front() != 0 ||
      size_t(value_off.back()) != values.size()) {
    throw std::invalid_argument("bad value offsets");
  }

  for (int block_row = 0; block_row < block_rows; ++block_row) {
    if (row_ptr[block_row] > row_ptr[block_row + 1] || row_size[block_row] <= 0 ||
        row_scalar_off[block_row + 1] - row_scalar_off[block_row] != row_size[block_row]) {
      throw std::invalid_argument("bad block row");
    }
  }
  for (int block_column = 0; block_column < block_cols; ++block_column) {
    if (col_size[block_column] <= 0 ||
        col_scalar_off[block_column + 1] - col_scalar_off[block_column] != col_size[block_column]) {
      throw std::invalid_argument("bad block column");
    }
  }

  // row_ptr is monotonic, so one forward cursor identifies the owner of every
  // block.
  int block_row = 0;
  for (size_t block_index = 0; block_index < block_col.size(); ++block_index) {
    while (row_ptr[block_row + 1] <= static_cast<int>(block_index)) {
      ++block_row;
    }
    const int block_column = block_col[block_index];
    if (block_column < 0 || block_column >= block_cols) {
      throw std::invalid_argument("block column out of range");
    }
    const int64_t expected_value_count = int64_t(row_size[block_row]) * col_size[block_column];
    if (value_off[block_index + 1] - value_off[block_index] != expected_value_count) {
      throw std::invalid_argument("block payload mismatch");
    }
  }
}

Matrix::Matrix(const HostMatrix& host) {
  host.validate();
  block_rows_ = host.block_rows;
  block_cols_ = host.block_cols;
  nnzb_ = int(host.block_col.size());
  rows_ = host.scalar_rows();
  cols_ = host.scalar_cols();
  row_ptr_ = copy_to_device(host.row_ptr);
  block_col_ = copy_to_device(host.block_col);
  row_size_ = copy_to_device(host.row_size);
  col_size_ = copy_to_device(host.col_size);
  row_off_ = copy_to_device(host.row_scalar_off);
  col_off_ = copy_to_device(host.col_scalar_off);
  value_off_ = copy_to_device(host.value_off);
  values_ = copy_to_device(host.values);
}
void Matrix::release() {
  cudaFree(row_ptr_);
  cudaFree(block_col_);
  cudaFree(row_size_);
  cudaFree(col_size_);
  cudaFree(row_off_);
  cudaFree(col_off_);
  cudaFree(value_off_);
  cudaFree(values_);
  row_ptr_ = nullptr;
}
Matrix::~Matrix() { release(); }
Matrix::Matrix(Matrix&& other) noexcept { *this = std::move(other); }
Matrix& Matrix::operator=(Matrix&& other) noexcept {
  if (this != &other) {
    release();
    block_rows_ = other.block_rows_;
    block_cols_ = other.block_cols_;
    nnzb_ = other.nnzb_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    row_ptr_ = other.row_ptr_;
    block_col_ = other.block_col_;
    row_size_ = other.row_size_;
    col_size_ = other.col_size_;
    row_off_ = other.row_off_;
    col_off_ = other.col_off_;
    value_off_ = other.value_off_;
    values_ = other.values_;

    other.row_ptr_ = nullptr;
    other.block_col_ = nullptr;
    other.row_size_ = nullptr;
    other.col_size_ = nullptr;
    other.row_off_ = nullptr;
    other.col_off_ = nullptr;
    other.value_off_ = nullptr;
    other.values_ = nullptr;
  }
  return *this;
}
DeviceMatrix Matrix::device_view() const {
  return {block_rows_, block_cols_, nnzb_,    rows_,    cols_,      row_ptr_, block_col_,
          row_size_,   col_size_,   row_off_, col_off_, value_off_, values_};
}
std::vector<float> cpu_reference(const HostMatrix& matrix, const std::vector<float>& input,
                                 int rhs_width) {
  if (input.size() != size_t(matrix.scalar_cols() * rhs_width)) {
    throw std::invalid_argument("B size");
  }

  // The reference mirrors the packed block layout but accumulates each dot
  // product in double.
  std::vector<float> output(matrix.scalar_rows() * rhs_width, 0.0f);
  for (int block_row = 0; block_row < matrix.block_rows; ++block_row) {
    const int row_height = matrix.row_size[block_row];
    for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1];
         ++block_index) {
      const int block_column = matrix.block_col[block_index];
      const int column_width = matrix.col_size[block_column];

      for (int local_row = 0; local_row < row_height; ++local_row) {
        for (int rhs_column = 0; rhs_column < rhs_width; ++rhs_column) {
          double sum = 0.0;
          for (int local_column = 0; local_column < column_width; ++local_column) {
            const auto value_index =
                matrix.value_off[block_index] + local_row + local_column * row_height;
            const auto input_index = matrix.col_scalar_off[block_column] + local_column +
                                     int64_t(rhs_column) * matrix.scalar_cols();
            sum += double(matrix.values[value_index]) * input[input_index];
          }
          const auto output_index = matrix.row_scalar_off[block_row] + local_row +
                                    int64_t(rhs_column) * matrix.scalar_rows();
          output[output_index] += float(sum);
        }
      }
    }
  }
  return output;
}
} // namespace vbsr
