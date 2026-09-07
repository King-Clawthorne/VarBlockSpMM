#include "dg_specialized.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbsr::bench {
namespace {

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS error in the specialized transport comparator");
}

// One strided batched SGEMM. The operator stride is zero, so every element in
// the batch multiplies the same cached dense block.
struct Run {
  const float* operator_values{};
  int rows{}, inner{}, count{};
  int64_t input_offset{}, output_offset{}, input_stride{}, output_stride{};
  float beta{};
};

} // namespace

struct DgSpecializedPlan::Impl {
  int rhs{};
  int64_t rows{}, cols{};
  int distinct{};
  size_t operator_bytes{}, assembled_bytes{};
  bool zero_output{};
  float* operators{};
  cublasHandle_t handle{};
  std::vector<Run> runs;
  ~Impl() {
    if (handle)
      cublasDestroy(handle);
    cudaFree(operators);
  }
};

DgSpecializedPlan::DgSpecializedPlan(const HostMatrix& a, int rhs) : impl_(new Impl) {
  a.validate();
  if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
    throw std::invalid_argument("invalid rhs width");
  if (a.scalar_rows() > INT32_MAX || a.scalar_cols() > INT32_MAX)
    throw std::invalid_argument("specialized comparator leading dimensions must fit INT32_MAX");
  impl_->rhs = rhs;
  impl_->rows = a.scalar_rows();
  impl_->cols = a.scalar_cols();
  impl_->assembled_bytes = a.values.size() * sizeof(float);

  // Deduplicate the dense blocks by shape and exact payload. A transport
  // operator with repeating orders collapses to a handful of distinct blocks.
  std::map<std::pair<int, std::string>, int> unique;
  std::vector<int> operator_of_block(a.block_col.size());
  std::vector<int64_t> operator_offset;
  std::vector<float> operator_values;
  int max_degree = 0;
  bool has_empty_rows = false;
  for (int row = 0; row < a.block_rows; row++) {
    max_degree = std::max(max_degree, a.row_ptr[row + 1] - a.row_ptr[row]);
    has_empty_rows |= a.row_ptr[row + 1] == a.row_ptr[row];
    for (int p = a.row_ptr[row]; p < a.row_ptr[row + 1]; p++) {
      const int height = a.row_size[row], width = a.col_size[a.block_col[p]];
      const size_t count = size_t(height) * width;
      const float* payload = a.values.data() + a.value_off[p];
      std::string bytes(reinterpret_cast<const char*>(payload), count * sizeof(float));
      auto key = std::make_pair(height, std::move(bytes));
      auto found = unique.find(key);
      if (found == unique.end()) {
        found = unique.emplace(std::move(key), int(operator_offset.size())).first;
        operator_offset.push_back(int64_t(operator_values.size()));
        operator_values.insert(operator_values.end(), payload, payload + count);
      }
      operator_of_block[p] = found->second;
    }
  }
  impl_->distinct = int(operator_offset.size());
  impl_->operator_bytes = operator_values.size() * sizeof(float);
  if (operator_values.empty())
    throw std::invalid_argument("specialized comparator requires at least one block");
  // Refuse to stand in for a general product. Without heavy reuse this
  // comparator would simply be a slower way to run the same GEMMs.
  if (int64_t(operator_values.size()) * 4 > int64_t(a.values.size()))
    throw std::invalid_argument(
        "specialized comparator requires repeated block operators, and this matrix has too few");
  cuda_check(cudaMalloc(&impl_->operators, operator_values.size() * sizeof(float)));
  cuda_check(cudaMemcpy(impl_->operators, operator_values.data(),
                        operator_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  blas_check(cublasCreate(&impl_->handle));
  // Keep FP32 arithmetic. The comparator must not win on reduced precision.
  blas_check(cublasSetMathMode(impl_->handle, CUBLAS_DEFAULT_MATH));

  impl_->zero_output = has_empty_rows;
  // Slot k holds the kth block of every row, so all slot-0 blocks together
  // cover each nonempty output row exactly once and may overwrite it.
  for (int slot = 0; slot < max_degree; slot++) {
    // Each entry is one block of this slot, as (input offset, output offset,
    // height, width), collected in increasing block-row order.
    struct Item {
      int64_t input_offset, output_offset;
      int height, width;
    };
    std::map<int, std::vector<Item>> groups;
    for (int row = 0; row < a.block_rows; row++) {
      const int p = a.row_ptr[row] + slot;
      if (p >= a.row_ptr[row + 1])
        continue;
      groups[operator_of_block[p]].push_back(Item{a.col_scalar_off[a.block_col[p]],
                                                  a.row_scalar_off[row], a.row_size[row],
                                                  a.col_size[a.block_col[p]]});
    }
    for (const auto& [id, items] : groups) {
      // Split each group into maximal runs of constant input and output stride.
      // A periodic wrap breaks one arithmetic sequence, which becomes its own run.
      size_t start = 0;
      while (start < items.size()) {
        size_t end = start + 1;
        int64_t input_stride = 0, output_stride = 0;
        if (end < items.size()) {
          input_stride = items[end].input_offset - items[start].input_offset;
          output_stride = items[end].output_offset - items[start].output_offset;
          while (end + 1 < items.size() &&
                 items[end + 1].input_offset - items[end].input_offset == input_stride &&
                 items[end + 1].output_offset - items[end].output_offset == output_stride)
            end++;
          end++;
        }
        Run run;
        run.operator_values = impl_->operators + operator_offset[id];
        run.rows = items[start].height;
        run.inner = items[start].width;
        run.count = int(end - start);
        run.input_offset = items[start].input_offset;
        run.output_offset = items[start].output_offset;
        run.input_stride = input_stride;
        run.output_stride = output_stride;
        run.beta = slot ? 1.f : 0.f;
        impl_->runs.push_back(run);
        start = end;
      }
    }
  }
}

DgSpecializedPlan::~DgSpecializedPlan() = default;
int DgSpecializedPlan::distinct_operators() const { return impl_->distinct; }
int DgSpecializedPlan::launch_count() const { return int(impl_->runs.size()); }
size_t DgSpecializedPlan::storage_bytes() const { return impl_->operator_bytes; }
size_t DgSpecializedPlan::assembled_bytes() const { return impl_->assembled_bytes; }

void DgSpecializedPlan::execute(const float* input, float* output, cudaStream_t stream) {
  blas_check(cublasSetStream(impl_->handle, stream));
  if (impl_->zero_output)
    cuda_check(cudaMemsetAsync(output, 0, size_t(impl_->rows) * impl_->rhs * sizeof(float), stream));
  const float alpha = 1.f;
  for (const auto& run : impl_->runs) {
    const float beta = impl_->zero_output ? 1.f : run.beta;
    blas_check(cublasSgemmStridedBatched(
        impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N, run.rows, impl_->rhs, run.inner, &alpha,
        run.operator_values, run.rows, 0, input + run.input_offset, int(impl_->cols),
        run.input_stride, &beta, output + run.output_offset, int(impl_->rows), run.output_stride,
        run.count));
  }
}

} // namespace vbsr::bench
