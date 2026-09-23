#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace vbsr {
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

    int64_t scalar_rows() const { return row_scalar_off.empty() ? 0 : row_scalar_off.back(); }
    int64_t scalar_cols() const { return col_scalar_off.empty() ? 0 : col_scalar_off.back(); }

    void validate() const;
    void validate_structure(size_t value_count) const;
  };

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
      explicit Matrix(const HostMatrix& host);
      ~Matrix();
      Matrix(const Matrix&) = delete;
      Matrix& operator=(const Matrix&) = delete;
      Matrix(Matrix&&) noexcept;
      Matrix& operator=(Matrix&&) noexcept;

      DeviceMatrix device_view() const;
      int64_t scalar_rows() const { return scalar_rows_; }
      int64_t scalar_cols() const { return scalar_cols_; }
      size_t storage_bytes() const;

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

  enum class Distribution { Uniform, LowVariance, HighVariance, Bimodal };

  struct GeneratorOptions {
    int block_rows = 1024;
    int block_cols = 1024;
    int degree = 4;
    int rhs_width = 32;
    Distribution distribution = Distribution::HighVariance;
    bool local_columns = true;
    uint64_t seed = 1;
  };

  HostMatrix generate(const GeneratorOptions&);
  std::vector<float> cpu_reference(const HostMatrix& matrix, const std::vector<float>& input, int rhs_width);
  enum class Kernel { Auto, RowOwned };

  struct PlanOptions {
    int rhs_width = 32;
    Kernel kernel = Kernel::Auto;
  };

  class Plan {
    public:
      Plan(const Matrix&, PlanOptions);
      Plan(DeviceMatrix matrix, PlanOptions options, cudaStream_t stream = 0);
      void execute(const float* input, float* output, cudaStream_t stream = 0) const;
      int launch_count() const;

    private:
      DeviceMatrix matrix_{};
      const int32_t* row_shape_order_{};
      int small_row_count_{};
      int large_row_count_{};
      PlanOptions options_{};
      std::shared_ptr<int32_t> owned_row_order_;
  };

  void launch_row_owned(const DeviceMatrix& matrix, const int32_t* row_order, int small_row_count,
                        int large_row_count, const float* input, float* output, int rhs_width,
                        cudaStream_t stream);
}
