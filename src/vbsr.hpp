#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <vector>

namespace vbsr {
/// Host-side representation of a variable-block sparse matrix.
///
/// Block payloads are stored contiguously in column-major order. The offset
/// arrays describe scalar coordinates and packed value positions; `row_ptr`
/// and `block_col` use the usual compressed sparse row layout at block level.
struct HostMatrix {
  int block_rows{};
  int block_cols{};
  std::vector<int32_t> row_ptr;
  std::vector<int32_t> block_col;
  std::vector<int32_t> row_size;
  std::vector<int32_t> col_size;
  std::vector<int64_t> row_scalar_off;
  std::vector<int64_t> col_scalar_off;
  std::vector<int64_t> value_off;
  std::vector<float> values;

  /// Returns the total number of scalar rows, or zero when offsets are absent.
  int64_t scalar_rows() const { return row_scalar_off.empty() ? 0 : row_scalar_off.back(); }
  /// Returns the total number of scalar columns, or zero when offsets are absent.
  int64_t scalar_cols() const { return col_scalar_off.empty() ? 0 : col_scalar_off.back(); }

  /// Validates metadata and the host value payload, throwing on malformed input.
  void validate() const;
  /// Validates metadata against a payload length without requiring host values.
  void validate_structure(size_t value_count) const;
};

/// Non-owning view of matrix arrays resident on a CUDA device.
struct DeviceMatrix {
  int block_rows{};
  int block_cols{};
  int nnzb{};
  int64_t scalar_rows{};
  int64_t scalar_cols{};
  const int32_t* row_ptr{};
  const int32_t* block_col{};
  const int32_t* row_size{};
  const int32_t* col_size{};
  const int64_t* row_scalar_off{};
  const int64_t* col_scalar_off{};
  const int64_t* value_off{};
  const float* values{};
};

class Matrix {
public:
  /// Validates and copies a host matrix and its metadata to the current device.
  explicit Matrix(const HostMatrix& host);
  /// Releases the device allocations owned by this matrix.
  ~Matrix();
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) noexcept;
  Matrix& operator=(Matrix&&) noexcept;

  /// Returns a non-owning device view; the matrix must outlive its consumers.
  DeviceMatrix device_view() const;
  int64_t scalar_rows() const { return scalar_rows_; }
  int64_t scalar_cols() const { return scalar_cols_; }
  /// Returns the logical device storage occupied by matrix arrays, in bytes.
  size_t storage_bytes() const;

  /// Copies replacement values device-to-device on `stream`.
  /// `device_values` must address a device buffer containing exactly the
  /// same number of values as the matrix payload.
  void update_values(const float* device_values, size_t value_count, cudaStream_t stream = 0);

private:
  friend class Plan;
  int block_rows_{};
  int block_cols_{};
  int nnzb_{};
  int small_row_count_{};
  int large_row_count_{};
  int64_t scalar_rows_{};
  int64_t scalar_cols_{};
  int32_t* row_ptr_{};
  int32_t* block_col_{};
  int32_t* row_size_{};
  int32_t* col_size_{};
  int32_t* row_shape_order_{};
  int64_t* row_scalar_off_{};
  int64_t* col_scalar_off_{};
  int64_t* value_off_{};
  float* values_{};
  size_t value_count_{};
  void release();
};

/// Computes Y = A * X on the CPU; X and Y use column-major scalar matrices.
std::vector<float> cpu_reference(const HostMatrix& matrix, std::span<const float> input,
                                 int rhs_width);
/// Vector overload retained for callers of the original API.
std::vector<float> cpu_reference(const HostMatrix& matrix, const std::vector<float>& input,
                                 int rhs_width);
/// Kernel selection policy used by a plan.
enum class Kernel { Auto, RowOwned };

/// Execution configuration for a sparse-dense multiplication plan.
struct PlanOptions {
  int rhs_width = 32;
  Kernel kernel = Kernel::Auto;
};

class Plan {
public:
  /// Creates a plan using the row ordering owned by `matrix`.
  Plan(const Matrix&, PlanOptions);
  /// Validates a device matrix view and creates an owned row-shape ordering.
  /// Metadata is copied to the host during validation; values remain on device.
  Plan(DeviceMatrix matrix, PlanOptions options, cudaStream_t stream = 0);
  /// Enqueues Y = A * X on `stream`. Input and output are device pointers.
  void execute(const float* input, float* output, cudaStream_t stream = 0) const;
  /// Returns the number of CUDA kernel launches selected for this plan.
  int launch_count() const;

private:
  DeviceMatrix matrix_{};
  const int32_t* row_shape_order_{};
  int small_row_count_{};
  int large_row_count_{};
  PlanOptions options_{};
  std::shared_ptr<int32_t> owned_row_order_;
};

/// Dispatches the row-owned CUDA implementation for the requested RHS width.
void launch_row_owned(const DeviceMatrix& matrix, const int32_t* row_order, int small_row_count,
                      int large_row_count, const float* input, float* output, int rhs_width,
                      cudaStream_t stream);
}
