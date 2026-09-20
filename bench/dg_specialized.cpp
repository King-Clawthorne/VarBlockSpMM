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

// One batched SGEMM over the elements that share a cached operator. The A
// pointer array repeats that operator, which is ordinary batched usage: only
// the output matrices have to be distinct, and they are.
struct Group {
  const float** matrix_a{};
  const float** matrix_b{};
  float** output{};
  int rows{}, inner{}, count{};
  float beta{};
};

} // namespace

struct DgSpecializedPlan::Impl {
  int rhs{};
  int64_t rows{}, cols{};
  int distinct{};
  size_t operator_bytes{}, assembled_bytes{}, pointer_bytes{};
  bool zero_output{};
  float* output{};
  float* operators{};
  const float** pointers{};
  cublasHandle_t handle{};
  std::vector<Group> groups;
  ~Impl() {
    if (handle)
      cublasDestroy(handle);
    cudaFree(operators);
    cudaFree(pointers);
  }
};

DgSpecializedPlan::DgSpecializedPlan(const HostMatrix& a, int rhs, const float* input, float* output)
    : impl_(new Impl) {
  a.validate();
  if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
    throw std::invalid_argument("invalid rhs width");
  if (a.scalar_rows() > INT32_MAX || a.scalar_cols() > INT32_MAX)
    throw std::invalid_argument("specialized comparator leading dimensions must fit INT32_MAX");
  if (input == nullptr || output == nullptr)
    throw std::invalid_argument("specialized comparator requires both device panels");
  impl_->rhs = rhs;
  impl_->rows = a.scalar_rows();
  impl_->cols = a.scalar_cols();
  impl_->output = output;
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
  struct Plan {
    int rows, inner;
    float beta;
    std::vector<const float*> matrix_a, matrix_b;
    std::vector<float*> output;
  };
  std::vector<Plan> plans;
  for (int slot = 0; slot < max_degree; slot++) {
    std::map<int, Plan> groups;
    for (int row = 0; row < a.block_rows; row++) {
      const int p = a.row_ptr[row] + slot;
      if (p >= a.row_ptr[row + 1])
        continue;
      const int id = operator_of_block[p];
      auto& group = groups[id];
      group.rows = a.row_size[row];
      group.inner = a.col_size[a.block_col[p]];
      group.beta = slot ? 1.f : 0.f;
      group.matrix_a.push_back(impl_->operators + operator_offset[id]);
      group.matrix_b.push_back(input + a.col_scalar_off[a.block_col[p]]);
      group.output.push_back(output + a.row_scalar_off[row]);
    }
    for (auto& [id, group] : groups)
      plans.push_back(std::move(group));
  }

  // One device array holds every pointer, so the batch arrays stay contiguous
  // and are uploaded once. They never change, because the panels are fixed.
  std::vector<const float*> flat;
  for (const auto& plan : plans) {
    flat.insert(flat.end(), plan.matrix_a.begin(), plan.matrix_a.end());
    flat.insert(flat.end(), plan.matrix_b.begin(), plan.matrix_b.end());
    for (float* pointer : plan.output)
      flat.push_back(pointer);
  }
  impl_->pointer_bytes = flat.size() * sizeof(const float*);
  cuda_check(cudaMalloc(&impl_->pointers, impl_->pointer_bytes));
  cuda_check(cudaMemcpy(impl_->pointers, flat.data(), impl_->pointer_bytes, cudaMemcpyHostToDevice));
  size_t cursor = 0;
  for (const auto& plan : plans) {
    Group group;
    group.rows = plan.rows;
    group.inner = plan.inner;
    group.count = int(plan.matrix_a.size());
    group.beta = plan.beta;
    group.matrix_a = impl_->pointers + cursor;
    group.matrix_b = impl_->pointers + cursor + group.count;
    group.output = const_cast<float**>(impl_->pointers + cursor + 2 * size_t(group.count));
    cursor += 3 * size_t(group.count);
    impl_->groups.push_back(group);
  }
  // Finish default-stream uploads before callers use a nonblocking stream.
  cuda_check(cudaStreamSynchronize(nullptr));
}

DgSpecializedPlan::~DgSpecializedPlan() = default;
int DgSpecializedPlan::distinct_operators() const { return impl_->distinct; }
int DgSpecializedPlan::launch_count() const { return int(impl_->groups.size()); }
size_t DgSpecializedPlan::storage_bytes() const {
  return impl_->operator_bytes + impl_->pointer_bytes;
}
size_t DgSpecializedPlan::operator_bytes() const { return impl_->operator_bytes; }
size_t DgSpecializedPlan::assembled_bytes() const { return impl_->assembled_bytes; }

void DgSpecializedPlan::execute(cudaStream_t stream) {
  blas_check(cublasSetStream(impl_->handle, stream));
  if (impl_->zero_output)
    cuda_check(cudaMemsetAsync(impl_->output, 0,
                               size_t(impl_->rows) * impl_->rhs * sizeof(float), stream));
  const float alpha = 1.f;
  for (const auto& group : impl_->groups) {
    const float beta = group.beta;
    blas_check(cublasSgemmBatched(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N, group.rows, impl_->rhs,
                                  group.inner, &alpha, group.matrix_a, group.rows, group.matrix_b,
                                  int(impl_->cols), &beta, group.output, int(impl_->rows),
                                  group.count));
  }
}

} // namespace vbsr::bench
