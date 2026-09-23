#include "vbsr.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace vbsr {
  namespace {
    void check_cuda(cudaError_t status) {
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }

    template <class T> T* copy_to_device(const std::vector<T>& source) {
      T* destination = nullptr;
      check_cuda(cudaMalloc(&destination, source.size() * sizeof(T)));

      try {
        check_cuda( cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
      } catch (...) {
        cudaFree(destination);
        throw;
      }

      return destination;
    }

  }

  void HostMatrix::validate() const { validate_structure(values.size()); }
  void HostMatrix::validate_structure(size_t value_count) const {

    if (block_rows <= 0 || block_cols <= 0) throw std::invalid_argument("positive block dimensions required");
    if (row_ptr.size() != size_t(block_rows) + 1 || row_size.size() != size_t(block_rows) || col_size.size() != size_t(block_cols)) {
      throw std::invalid_argument("metadata size mismatch");
    }

    if (row_scalar_off.size() != size_t(block_rows) + 1 || col_scalar_off.size() != size_t(block_cols) + 1) {
      throw std::invalid_argument("scalar offsets size mismatch");
    }

    if (row_scalar_off.front() != 0 || col_scalar_off.front() != 0) {
      throw std::invalid_argument("scalar offsets must start at zero");
    }

    if (row_ptr.front() != 0 || row_ptr.back() < 0 || size_t(row_ptr.back()) != block_col.size()) {
      throw std::invalid_argument("row_ptr does not match block columns");
    }

    if (value_off.size() != block_col.size() + 1 || value_off.front() != 0 || value_off.back() < 0 || size_t(value_off.back()) != value_count) {
      throw std::invalid_argument("value_off does not match packed values");
    }

    int64_t expected_row_offset = 0;
    for (int block_row = 0; block_row < block_rows; ++block_row) {
      expected_row_offset += row_size[block_row];
      if (row_ptr[block_row] > row_ptr[block_row + 1] || row_size[block_row] < 8 ||
          row_size[block_row] > 64 || row_size[block_row] % 8 != 0 ||
          row_scalar_off[block_row + 1] != expected_row_offset) {
        throw std::invalid_argument("invalid block-row metadata");
      }
    }

    int64_t expected_col_offset = 0;
    for (int block_column = 0; block_column < block_cols; ++block_column) {
      expected_col_offset += col_size[block_column];
      if (col_size[block_column] < 8 || col_size[block_column] > 64 ||
          col_size[block_column] % 8 != 0 ||
          col_scalar_off[block_column + 1] != expected_col_offset) {
        throw std::invalid_argument("invalid block-column metadata");
      }
    }

    int64_t expected_value_offset = 0;
    for (int block_row = 0; block_row < block_rows; ++block_row) {
      for (int block_index = row_ptr[block_row]; block_index < row_ptr[block_row + 1]; ++block_index) {
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

  Matrix::Matrix(const HostMatrix& host) {
    host.validate();
    block_rows_ = host.block_rows;
    block_cols_ = host.block_cols;
    nnzb_ = int(host.block_col.size());
    scalar_rows_ = host.scalar_rows();
    scalar_cols_ = host.scalar_cols();
    value_count_ = host.values.size();

    std::vector<int32_t> row_shape_order(host.block_rows);
    std::iota(row_shape_order.begin(), row_shape_order.end(), int32_t{0});
    const auto large_begin = std::stable_partition(row_shape_order.begin(), row_shape_order.end(), [&](int32_t block_row) { return host.row_size[block_row] <= 16; });
    small_row_count_ = int(large_begin - row_shape_order.begin());
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

  Matrix::~Matrix() { release(); }
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

  DeviceMatrix Matrix::device_view() const {
    return {block_rows_,     block_cols_, nnzb_,     scalar_rows_, scalar_cols_,
            row_ptr_,        block_col_,  row_size_, col_size_,    row_scalar_off_,
            col_scalar_off_, value_off_,  values_};
  }

  void Matrix::update_values(const float* source, size_t count, cudaStream_t stream) {
    if (count != value_count_ || (count && (!source || !values_))) throw std::invalid_argument("value update size or pointer mismatch");
    if (count && source != values_) check_cuda(cudaMemcpyAsync(values_, source, count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }

  size_t Matrix::storage_bytes() const {
    int64_t count = 0;
    check_cuda(cudaMemcpy(&count, value_off_ + nnzb_, sizeof(count), cudaMemcpyDeviceToHost));
    return size_t(count) * sizeof(float) +
          (size_t(block_rows_) * 3 + 1 + block_cols_ + nnzb_) * sizeof(int32_t) +
          (size_t(block_rows_) + block_cols_ + nnzb_ + 3) * sizeof(int64_t);
  }

  std::vector<float> cpu_reference(const HostMatrix& matrix, const std::vector<float>& input, int rhs_width) {
    if (input.size() != size_t(matrix.scalar_cols() * rhs_width)) {
      throw std::invalid_argument("input matrix has incorrect size");
    }

    std::vector<float> output(matrix.scalar_rows() * rhs_width, 0.0f);
    for (int block_row = 0; block_row < matrix.block_rows; ++block_row) {
      const int row_height = matrix.row_size[block_row];
      for (int local_row = 0; local_row < row_height; ++local_row) {
        for (int rhs_column = 0; rhs_column < rhs_width; ++rhs_column) {
          double sum = 0.0;
          for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1]; ++block_index) {
            const int block_column = matrix.block_col[block_index];
            const int column_width = matrix.col_size[block_column];
            for (int local_column = 0; local_column < column_width; ++local_column) {
              const auto value_index = matrix.value_off[block_index] + local_row + local_column * row_height;
              const auto input_index = matrix.col_scalar_off[block_column] + local_column + int64_t(rhs_column) * matrix.scalar_cols();
              sum += double(matrix.values[value_index]) * input[input_index];
            }
          }

          const auto output_index = matrix.row_scalar_off[block_row] + local_row + int64_t(rhs_column) * matrix.scalar_rows();
          output[output_index] = float(sum);
        }
      }
    }

    return output;
  }
}
