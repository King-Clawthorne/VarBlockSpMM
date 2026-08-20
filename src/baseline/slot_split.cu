#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "varblockspmm/vbsr.hpp"

namespace vbsr {
static void cuda_check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
static void blas_check(cublasStatus_t e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS failure: " + std::to_string(int(e)));
}

struct GroupedGemmPlan::Impl {
  struct Slot {
    std::vector<cublasOperation_t> matrix_a_operations;
    std::vector<cublasOperation_t> matrix_b_operations;
    std::vector<int> row_counts;
    std::vector<int> column_counts;
    std::vector<int> inner_dimensions;
    std::vector<int> matrix_a_strides;
    std::vector<int> matrix_b_strides;
    std::vector<int> output_strides;
    std::vector<int> group_sizes;
    std::vector<float> alphas;
    std::vector<float> betas;
    std::vector<const float*> matrix_a_pointers;
    std::vector<int64_t> input_offsets;
    std::vector<int64_t> output_offsets;
    const float** device_matrix_a_pointers{};
    const float** device_matrix_b_pointers{};
    float** device_output_pointers{};
  };
  int rhs{};
  int64_t rows{}, cols{};
  float* values{};
  cublasHandle_t handle{};
  std::vector<Slot> slots;
  ~Impl() {
    for (auto& slot : slots) {
      cudaFree(slot.device_matrix_a_pointers);
      cudaFree(slot.device_matrix_b_pointers);
      cudaFree(slot.device_output_pointers);
    }
    if (handle)
      cublasDestroy(handle);
    cudaFree(values);
  }
};

GroupedGemmPlan::GroupedGemmPlan(const HostMatrix& a, int rhs) : impl_(new Impl) {
  a.validate();
  if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
    throw std::invalid_argument("invalid rhs width");
  impl_->rhs = rhs;
  impl_->rows = a.scalar_rows();
  impl_->cols = a.scalar_cols();
  cuda_check(cudaMalloc(&impl_->values, a.values.size() * sizeof(float)));
  cuda_check(cudaMemcpy(impl_->values, a.values.data(), a.values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  blas_check(cublasCreate(&impl_->handle));
  int max_degree = 0;
  for (int i = 0; i < a.block_rows; i++)
    max_degree = std::max(max_degree, a.row_ptr[i + 1] - a.row_ptr[i]);
  impl_->slots.resize(max_degree);
  // Slot k contains the kth block from every row that has one. Executing slots
  // in order lets the first overwrite C and later slots accumulate into it.
  for (int slot = 0; slot < max_degree; slot++) {
    using Key = std::pair<int, int>;
    std::map<Key, std::vector<std::pair<int, int>>> groups;
    for (int i = 0; i < a.block_rows; i++) {
      int p = a.row_ptr[i] + slot;
      if (p < a.row_ptr[i + 1])
        groups[{a.row_size[i], a.col_size[a.block_col[p]]}].push_back({i, p});
    }
    auto& slot_data = impl_->slots[slot];
    // Group blocks by (row height, column width), as required by grouped GEMM.
    for (const auto& entry : groups) {
      const int row_height = entry.first.first;
      const int column_width = entry.first.second;

      slot_data.matrix_a_operations.push_back(CUBLAS_OP_N);
      slot_data.matrix_b_operations.push_back(CUBLAS_OP_N);
      slot_data.row_counts.push_back(row_height);
      slot_data.column_counts.push_back(rhs);
      slot_data.inner_dimensions.push_back(column_width);
      slot_data.matrix_a_strides.push_back(row_height);
      slot_data.matrix_b_strides.push_back(int(a.scalar_cols()));
      slot_data.output_strides.push_back(int(a.scalar_rows()));
      slot_data.alphas.push_back(1.0f);
      slot_data.betas.push_back(slot ? 1.0f : 0.0f);
      slot_data.group_sizes.push_back(int(entry.second.size()));

      for (auto [block_row, block_index] : entry.second) {
        const int block_column = a.block_col[block_index];
        slot_data.matrix_a_pointers.push_back(impl_->values + a.value_off[block_index]);
        slot_data.input_offsets.push_back(a.col_scalar_off[block_column]);
        slot_data.output_offsets.push_back(a.row_scalar_off[block_row]);
      }
    }
    const size_t pointer_bytes = slot_data.matrix_a_pointers.size() * sizeof(float*);
    cuda_check(cudaMalloc(&slot_data.device_matrix_a_pointers, pointer_bytes));
    cuda_check(cudaMalloc(&slot_data.device_matrix_b_pointers, pointer_bytes));
    cuda_check(cudaMalloc(&slot_data.device_output_pointers, pointer_bytes));
    cuda_check(cudaMemcpy(slot_data.device_matrix_a_pointers, slot_data.matrix_a_pointers.data(),
                          pointer_bytes, cudaMemcpyHostToDevice));
  }
}

GroupedGemmPlan::~GroupedGemmPlan() = default;
int GroupedGemmPlan::launch_count() const { return int(impl_->slots.size()); }
void GroupedGemmPlan::execute(const float* B, float* C, cudaStream_t stream) {
  blas_check(cublasSetStream(impl_->handle, stream));
  cuda_check(cudaMemsetAsync(C, 0, impl_->rows * impl_->rhs * sizeof(float), stream));
  for (auto& slot : impl_->slots) {
    // A pointers are persistent. B and C pointers depend on this call's base
    // addresses.
    std::vector<const float*> input_pointers(slot.matrix_a_pointers.size());
    std::vector<float*> output_pointers(slot.matrix_a_pointers.size());
    for (size_t index = 0; index < input_pointers.size(); ++index) {
      input_pointers[index] = B + slot.input_offsets[index];
      output_pointers[index] = C + slot.output_offsets[index];
    }
    const size_t pointer_bytes = input_pointers.size() * sizeof(float*);
    cuda_check(cudaMemcpyAsync(slot.device_matrix_b_pointers, input_pointers.data(), pointer_bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(slot.device_output_pointers, output_pointers.data(), pointer_bytes,
                               cudaMemcpyHostToDevice, stream));
    blas_check(cublasSgemmGroupedBatched(
        impl_->handle, slot.matrix_a_operations.data(), slot.matrix_b_operations.data(),
        slot.row_counts.data(), slot.column_counts.data(), slot.inner_dimensions.data(),
        slot.alphas.data(), slot.device_matrix_a_pointers, slot.matrix_a_strides.data(),
        slot.device_matrix_b_pointers, slot.matrix_b_strides.data(), slot.betas.data(),
        slot.device_output_pointers, slot.output_strides.data(), int(slot.group_sizes.size()),
        slot.group_sizes.data()));
  }
}

void slot_split_baseline(const HostMatrix& a, const float* B, float* C, int rhs,
                         cudaStream_t stream) {
  GroupedGemmPlan p(a, rhs);
  p.execute(B, C, stream);
  cuda_check(cudaStreamSynchronize(stream));
}
} // namespace vbsr
