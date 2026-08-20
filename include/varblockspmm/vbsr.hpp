#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vbsr {
/** Host-side variable-block sparse-row matrix.
 *
 * Blocks and dense matrices use column-major storage. `row_ptr` indexes the
 * nonzero blocks belonging to each block row, while `value_off` indexes each
 * block's packed scalar payload. Scalar offset arrays translate block indices
 * into rows and columns of the unblocked matrix.
 */
struct HostMatrix {
  int block_rows{}, block_cols{};
  std::vector<int32_t> row_ptr, block_col, row_size, col_size;
  std::vector<int64_t> row_scalar_off, col_scalar_off, value_off;
  std::vector<float> values;
  /** Returns the number of scalar rows represented by all block rows. */
  int64_t scalar_rows() const { return row_scalar_off.empty() ? 0 : row_scalar_off.back(); }
  /** Returns the number of scalar columns represented by all block columns. */
  int64_t scalar_cols() const { return col_scalar_off.empty() ? 0 : col_scalar_off.back(); }

  /** Throws `std::invalid_argument` if metadata or packed values are
   * inconsistent. */
  void validate() const;
};

/** Non-owning device view of a variable-block sparse matrix. */
struct DeviceMatrix {
  int block_rows{}, block_cols{}, nnzb{};
  int64_t scalar_rows{}, scalar_cols{};
  const int32_t *row_ptr{}, *block_col{}, *row_size{}, *col_size{};
  const int64_t *row_scalar_off{}, *col_scalar_off{}, *value_off{};
  const float* values{};
};

/** Owns an immutable copy of a validated matrix in CUDA device memory. */
class Matrix {
public:
  /** Copies `host` and all of its metadata to the current CUDA device. */
  explicit Matrix(const HostMatrix& host);
  ~Matrix();
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) noexcept;
  Matrix& operator=(Matrix&&) noexcept;
  /** Returns a non-owning view valid for this object's lifetime. */
  DeviceMatrix device_view() const;
  int64_t scalar_rows() const { return rows_; }
  int64_t scalar_cols() const { return cols_; }

private:
  int block_rows_{}, block_cols_{}, nnzb_{};
  int64_t rows_{}, cols_{};
  int32_t *row_ptr_{}, *block_col_{}, *row_size_{}, *col_size_{};
  int64_t *row_off_{}, *col_off_{}, *value_off_{};
  float* values_{};
  void release();
};

/** Distribution used when generating block heights and widths. */
enum class Distribution { Uniform, LowVariance, HighVariance, Bimodal };

/** Parameters for deterministic synthetic matrix generation. */
struct GeneratorOptions {
  int block_rows = 1024, block_cols = 1024, degree = 4;
  int rhs_width = 32;
  Distribution distribution = Distribution::HighVariance;
  bool local_columns = true;
  uint64_t seed = 1;
};

/** Generates a validated matrix from `options`. Equal seeds produce equal
 * matrices. */
HostMatrix generate(const GeneratorOptions&);

/** Computes `C = A * B` on the CPU using double-precision accumulation.
 *
 * `B` and the returned `C` are column-major. `B` must contain
 * `matrix.scalar_cols() * rhs_width` elements.
 */
std::vector<float> cpu_reference(const HostMatrix&, const std::vector<float>& B, int rhs);

/** Direct-kernel selection. Split-row remains reserved and is rejected in
 * version 1.0. */
enum class Kernel { Auto, RowOwned, SplitRow };

/** Configuration for a reusable direct execution plan. */
struct PlanOptions {
  int rhs_width = 32;
  Kernel kernel = Kernel::Auto;
};

/** Lightweight non-owning execution plan for the row-owned CUDA kernel.
 *
 * The source `Matrix` must outlive the plan. Calls enqueue work asynchronously
 * on the supplied stream.
 */
class Plan {
public:
  Plan(const Matrix&, PlanOptions);

  /** Enqueues `C = A * B`; both dense matrices are column-major device buffers.
   */
  void execute(const float* B, float* C, cudaStream_t stream = 0) const;

private:
  DeviceMatrix a_{};
  PlanOptions options_{};
};

/** Persistent cuSPARSE baseline using an expanded scalar CSR representation. */
class ScalarCsrPlan {
public:
  ScalarCsrPlan(const HostMatrix&, int rhs_width);
  ~ScalarCsrPlan();
  ScalarCsrPlan(const ScalarCsrPlan&) = delete;
  /** Enqueues the baseline multiplication on `stream`. */
  void execute(const float* B, float* C, cudaStream_t stream = 0);
  /** Returns persistent temporary storage allocated after the first execution.
   */
  size_t workspace_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Persistent cuBLAS baseline that groups equally shaped blocks into GEMM
 * batches. */
class GroupedGemmPlan {
public:
  GroupedGemmPlan(const HostMatrix&, int rhs_width);
  ~GroupedGemmPlan();
  GroupedGemmPlan(const GroupedGemmPlan&) = delete;
  /** Enqueues all grouped GEMM batches on `stream`. */
  void execute(const float* B, float* C, cudaStream_t stream = 0);
  /** Returns the number of sequential block slots (and therefore launches). */
  int launch_count() const;
  size_t workspace_bytes() const { return 0; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Launches the optimized row-owned kernel, using width-specific ILP and
 * shared-memory input staging when beneficial. */
void launch_row_owned(DeviceMatrix, const float*, float*, int rhs, cudaStream_t);
/** Launches the scalar row-owned kernel for comparison and profiling. */
void launch_row_owned_scalar(DeviceMatrix, const float*, float*, int rhs, cudaStream_t);

/** One-shot, synchronizing cuSPARSE baseline. Prefer `ScalarCsrPlan` for
 * repeated work. */
void cusparse_scalar_baseline(const HostMatrix&, const float* dB, float* dC, int rhs, cudaStream_t);
/** One-shot, synchronizing grouped-cuBLAS baseline. Prefer `GroupedGemmPlan`
 * for reuse. */
void slot_split_baseline(const HostMatrix&, const float* dB, float* dC, int rhs, cudaStream_t);
} // namespace vbsr
