#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vbsr {

struct HostMatrix {
  int block_rows{}, block_cols{};
  std::vector<int32_t> row_ptr, block_col, row_size, col_size;
  std::vector<int64_t> row_scalar_off, col_scalar_off, value_off;
  std::vector<float> values;

  int64_t scalar_rows() const { return row_scalar_off.empty() ? 0 : row_scalar_off.back(); }

  int64_t scalar_cols() const { return col_scalar_off.empty() ? 0 : col_scalar_off.back(); }

  void validate() const;

  void validate_structure(size_t value_count) const;
};

struct DeviceMatrix {
  int block_rows{}, block_cols{}, nnzb{};
  int64_t scalar_rows{}, scalar_cols{};
  const int32_t *row_ptr{}, *block_col{}, *row_size{}, *col_size{};
  const int64_t *row_scalar_off{}, *col_scalar_off{}, *value_off{};
  const float* values{};
};

class Matrix {
public:
  explicit Matrix(const HostMatrix& host);
  ~Matrix();
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) noexcept;
  Matrix& operator=(Matrix&&) noexcept;

  DeviceMatrix device_view() const;
  int64_t scalar_rows() const { return rows_; }
  int64_t scalar_cols() const { return cols_; }
  size_t storage_bytes() const;

  void update_values(const float* device_values, size_t value_count, cudaStream_t stream = 0);

private:
  friend class Plan;
  int block_rows_{}, block_cols_{}, nnzb_{};
  int small_row_count_{}, large_row_count_{};
  int64_t rows_{}, cols_{};
  int32_t *row_ptr_{}, *block_col_{}, *row_size_{}, *col_size_{}, *row_shape_order_{};
  int64_t *row_off_{}, *col_off_{}, *value_off_{};
  float* values_{};
  size_t value_count_{};
  void release();
};

enum class Distribution { Uniform, LowVariance, HighVariance, Bimodal };

struct GeneratorOptions {
  int block_rows = 1024, block_cols = 1024, degree = 4;
  int rhs_width = 32;
  Distribution distribution = Distribution::HighVariance;
  bool local_columns = true;
  uint64_t seed = 1;
};

HostMatrix generate(const GeneratorOptions&);

std::vector<float> cpu_reference(const HostMatrix&, const std::vector<float>& B, int rhs);

enum class Kernel { Auto, RowOwned };

struct PlanOptions {
  int rhs_width = 32;
  Kernel kernel = Kernel::Auto;
};

class Plan {
public:
  Plan(const Matrix&, PlanOptions);

  Plan(DeviceMatrix, PlanOptions, cudaStream_t stream = 0);

  void execute(const float* B, float* C, cudaStream_t stream = 0) const;

  int launch_count() const;

private:
  DeviceMatrix a_{};
  const int32_t* row_shape_order_{};
  int small_row_count_{}, large_row_count_{};
  PlanOptions options_{};
  std::shared_ptr<int32_t> owned_row_shape_order_;
};

void launch_row_owned(const DeviceMatrix&, const int32_t* row_shape_order, int small_row_count,
                      int large_row_count, const float*, float*, int rhs, cudaStream_t);
} // namespace vbsr
