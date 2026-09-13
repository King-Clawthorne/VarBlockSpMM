#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vbsr {

/** Numerical contract.
 *
 * Each output entry is a sum of products, evaluated as FP32 multiply-add with
 * FP32 accumulation. There is no double accumulation, no compensated
 * summation, and no reassociation guarantee: the order in which a block row's
 * contributions are accumulated is unspecified and may change between kernels,
 * panel widths, and releases.
 *
 * The device sources are compiled with `--use_fast_math`, which implies
 * `-ftz=true`. Flush-to-zero applies per instruction, to that instruction's
 * inputs and to its rounded result. It does not apply to the product term
 * inside a fused multiply-add: that product is computed exactly and added to
 * the accumulator before a single rounding, so `fma(FLT_MIN, 0.5, FLT_MIN)`
 * retains `1.5 * FLT_MIN`. Three consequences follow, and none of them can be
 * excluded by inspecting the caller's inputs alone.
 *
 * Intermediate underflow: any instruction input, or any accumulator value an
 * instruction produces, that lands in the subnormal range is replaced by zero.
 * Contributions that would have summed to a normal result can be lost when
 * they pass through a subnormal accumulator. Two contributions of
 * `FLT_MIN * 0.5` have normal operands and an exact sum of `FLT_MIN`, and this
 * library returns zero for them, because each multiply-add produces a
 * subnormal result from a zero accumulator. The largest subnormal times 2^126
 * returns zero for the other reason: the operand itself is flushed on input.
 *
 * Intermediate overflow: a partial sum above `FLT_MAX` becomes an infinity and
 * propagates, even when the exact sum is finite. Because the accumulation
 * order is unspecified, callers cannot rely on a later cancelling term
 * rescuing such a sum.
 *
 * Cancellation: accumulated error grows with the number of terms and with
 * cancellation between them, as in any uncompensated FP32 summation.
 *
 * What is verified, rather than guaranteed: on the tested workloads results
 * agree with a reference that accumulates in double and rounds once, within a
 * maximum absolute error at most 5e-4 and relative L2 error at most 5e-5, and
 * `tests/test_numeric_range.cpp` pins each behavior above, including the
 * multi-contribution cases. This library is not a drop-in numerical
 * substitute for an IEEE 754 gradual-underflow implementation. Callers whose
 * values or intermediates approach the subnormal range should rescale, or
 * rebuild without `--use_fast_math`.
 */

/** Host-side variable-block sparse-row matrix.
 *
 * Blocks and dense matrices use column-major storage. `row_ptr` indexes the
 * nonzero blocks belonging to each block row, while `value_off` indexes each
 * block's packed scalar payload. Block columns must be sorted and unique within
 * each row. Scalar offset arrays translate block indices
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
  /** Validates metadata against a payload count without accessing values. */
  void validate_structure(size_t value_count) const;
};

/** Non-owning device view of a variable-block sparse matrix. */
struct DeviceMatrix {
  int block_rows{}, block_cols{}, nnzb{};
  int64_t scalar_rows{}, scalar_cols{};
  const int32_t *row_ptr{}, *block_col{}, *row_size{}, *col_size{};
  const int64_t *row_scalar_off{}, *col_scalar_off{}, *value_off{};
  const float* values{};
};

/** Owns fixed matrix structure and mutable values in CUDA device memory. */
class Matrix {
public:
  /** Copies `host` and all of its metadata to the current CUDA device. */
  explicit Matrix(const HostMatrix& host);
  ~Matrix();
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) noexcept;
  Matrix& operator=(Matrix&&) noexcept;
  /** Returns a non-owning view, invalidated by destruction or move assignment. */
  DeviceMatrix device_view() const;
  int64_t scalar_rows() const { return rows_; }
  int64_t scalar_cols() const { return cols_; }
  size_t storage_bytes() const;
  /** Enqueues a value-only device-to-device copy. Existing plans remain valid.
   * Source has exactly value_count elements and must remain live until the copy
   * completes. Source and destination must be disjoint or identical.
   * Order updates and executions on one stream, or use CUDA events.
   */
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
 * The source allocation must outlive the plan and its queued work. Do not move
 * assign the owning Matrix while a plan refers to its old allocation.
 * Calls enqueue work asynchronously
 * on the supplied stream.
 */
class Plan {
public:
  Plan(const Matrix&, PlanOptions);
  /** Borrows existing device allocations. Construction synchronizes stream and
   * validates a host copy of metadata only. Values stay on the GPU. The caller
   * guarantees buffer extents, device accessibility, and allocation lifetime.
   * Structure must remain unchanged until this plan and its queued work finish.
   * Values may be produced or updated on the execution stream. Rebuild the plan
   * after structural changes. The plan owns only its row classification array.
   */
  Plan(DeviceMatrix, PlanOptions, cudaStream_t stream = 0);

  /** Enqueues `C = A * B`; both dense matrices are non-overlapping, sufficiently
   * sized column-major device buffers on the matrix's CUDA device.
   */
  void execute(const float* B, float* C, cudaStream_t stream = 0) const;
  /** Returns the number of kernel launches used by one execution. */
  int launch_count() const;

private:
  DeviceMatrix a_{};
  const int32_t* row_shape_order_{};
  int small_row_count_{}, large_row_count_{};
  PlanOptions options_{};
  std::shared_ptr<int32_t> owned_row_shape_order_;
};

/** Persistent cuSPARSE baseline using an expanded scalar CSR representation. */
class ScalarCsrPlan {
public:
  /** Algorithm 0 preserves the historical default. Algorithms 1 to 3 select
   * explicit CSR variants. Fixed BSR subdivides blocks into 8-by-8 or 32-by-32
   * tiles without padding, and rejects partitions not divisible by that size.
   * Scalar CSR requires the expanded entry count and columns to fit INT32_MAX.
   * Preprocessing, when requested, happens on the first execute call.
   */
  ScalarCsrPlan(const HostMatrix&, int rhs_width, int algorithm = 0, bool preprocess = false,
                bool fixed_bsr = false, int bsr_block_size = 32);
  ~ScalarCsrPlan();
  ScalarCsrPlan(const ScalarCsrPlan&) = delete;
  /** Enqueues the baseline multiplication on `stream`. One plan must be ordered
   * on one stream or externally synchronized across streams and host threads. */
  void execute(const float* B, float* C, cudaStream_t stream = 0);
  /** Returns persistent temporary storage allocated after the first execution.
   */
  size_t workspace_bytes() const;
  size_t storage_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Persistent cuBLAS baseline that groups equally shaped blocks into GEMM
 * batches. */
class GroupedGemmPlan {
public:
  /** The default preserves the historical pointer-refresh baseline.
   * Cached mode reuses pointer arrays for unchanged base addresses and skips
   * the full output clear when no block row is empty. Calls on a plan must
   * be ordered on one stream or externally synchronized across streams.
   */
  GroupedGemmPlan(const HostMatrix&, int rhs_width, bool cache_pointers = false);
  ~GroupedGemmPlan();
  GroupedGemmPlan(const GroupedGemmPlan&) = delete;
  /** Enqueues all grouped GEMM batches on `stream`. */
  void execute(const float* B, float* C, cudaStream_t stream = 0);
  /** Returns sequential grouped GEMM calls, excluding pointer-refresh kernels
   * and any internal library launches. Leading dimensions must fit INT32_MAX. */
  int launch_count() const;
  size_t workspace_bytes() const { return 0; }
  size_t storage_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Launches the optimized row-owned kernel, using width-specific ILP and
 * shared-memory input staging when beneficial. */
void launch_row_owned(DeviceMatrix, const int32_t* row_shape_order, int small_row_count,
                      int large_row_count, const float*, float*, int rhs, cudaStream_t);
/** Launches the scalar row-owned kernel for comparison and profiling. */
void launch_row_owned_scalar(DeviceMatrix, const float*, float*, int rhs, cudaStream_t);

/** One-shot, synchronizing cuSPARSE baseline. Prefer `ScalarCsrPlan` for
 * repeated work. */
void cusparse_scalar_baseline(const HostMatrix&, const float* dB, float* dC, int rhs, cudaStream_t);
/** One-shot, synchronizing grouped-cuBLAS baseline. Prefer `GroupedGemmPlan`
 * for reuse. */
void slot_split_baseline(const HostMatrix&, const float* dB, float* dC, int rhs, cudaStream_t);
} // namespace vbsr
