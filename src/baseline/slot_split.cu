#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "varblockspmm/vbsr.hpp"

namespace vbsr {
__global__ void refresh_grouped_pointers(const float* input, float* output,
                                         const int64_t* input_offsets, const int64_t* output_offsets,
                                         const float** inputs, float** outputs, size_t count) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    inputs[index] = input + input_offsets[index];
    outputs[index] = output + output_offsets[index];
  }
}
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
    int64_t* device_input_offsets{};
    int64_t* device_output_offsets{};
  };
  int rhs{};
  bool cache_pointers = false;
  bool has_empty_rows = false;
  const float* cached_input = nullptr;
  float* cached_output = nullptr;
  int64_t rows{}, cols{};
  float* values{};
  size_t value_count{};
  cublasHandle_t handle{};
  std::vector<Slot> slots;
  ~Impl() {
    for (auto& slot : slots) {
      cudaFree(slot.device_matrix_a_pointers);
      cudaFree(slot.device_matrix_b_pointers);
      cudaFree(slot.device_output_pointers);
      cudaFree(slot.device_input_offsets);
      cudaFree(slot.device_output_offsets);
    }
    if (handle)
      cublasDestroy(handle);
    cudaFree(values);
  }
};

GroupedGemmPlan::GroupedGemmPlan(const HostMatrix& a, int rhs, bool cache_pointers)
    : impl_(new Impl) {
  impl_->cache_pointers = cache_pointers;
  a.validate();
  if (a.scalar_rows() > INT32_MAX || a.scalar_cols() > INT32_MAX)
    throw std::invalid_argument("grouped GEMM leading dimensions must fit INT32_MAX");
  if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
    throw std::invalid_argument("invalid rhs width");
  impl_->rhs = rhs;
  impl_->rows = a.scalar_rows();
  impl_->cols = a.scalar_cols();
  impl_->value_count = a.values.size();
  cuda_check(cudaMalloc(&impl_->values, a.values.size() * sizeof(float)));
  cuda_check(cudaMemcpy(impl_->values, a.values.data(), a.values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  blas_check(cublasCreate(&impl_->handle));
  int max_degree = 0;
  for (int i = 0; i < a.block_rows; i++) {
    max_degree = std::max(max_degree, a.row_ptr[i + 1] - a.row_ptr[i]);
    impl_->has_empty_rows |= a.row_ptr[i + 1] == a.row_ptr[i];
  }
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
    const size_t offset_bytes = slot_data.input_offsets.size() * sizeof(int64_t);
    cuda_check(cudaMalloc(&slot_data.device_input_offsets, offset_bytes));
    cuda_check(cudaMalloc(&slot_data.device_output_offsets, offset_bytes));
    cuda_check(cudaMemcpy(slot_data.device_input_offsets, slot_data.input_offsets.data(), offset_bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(slot_data.device_output_offsets, slot_data.output_offsets.data(), offset_bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(slot_data.device_matrix_a_pointers, slot_data.matrix_a_pointers.data(),
                          pointer_bytes, cudaMemcpyHostToDevice));
  }
}

GroupedGemmPlan::~GroupedGemmPlan() = default;
int GroupedGemmPlan::launch_count() const { return int(impl_->slots.size()); }
size_t GroupedGemmPlan::storage_bytes() const {
  size_t bytes = impl_->value_count * sizeof(float);
  for (const auto& slot : impl_->slots)
    bytes += slot.matrix_a_pointers.size() * (3 * sizeof(float*) + 2 * sizeof(int64_t));
  return bytes;
}
void GroupedGemmPlan::execute(const float* B, float* C, cudaStream_t stream) {
  blas_check(cublasSetStream(impl_->handle, stream));
  if (!impl_->cache_pointers || impl_->has_empty_rows) {
    cuda_check(cudaMemsetAsync(C, 0, impl_->rows * impl_->rhs * sizeof(float), stream));
  }
  const bool refresh =
      !impl_->cache_pointers || B != impl_->cached_input || C != impl_->cached_output;
  for (auto& slot : impl_->slots) {
    // A pointers are persistent. B and C pointers depend on this call's base
    // addresses.
    if (refresh) {
      const size_t count = slot.matrix_a_pointers.size();
      refresh_grouped_pointers<<<unsigned((count + 255) / 256), 256, 0, stream>>>(
          B, C, slot.device_input_offsets, slot.device_output_offsets,
          slot.device_matrix_b_pointers, slot.device_output_pointers, count);
      cuda_check(cudaGetLastError());
    }
    blas_check(cublasSgemmGroupedBatched(
        impl_->handle, slot.matrix_a_operations.data(), slot.matrix_b_operations.data(),
        slot.row_counts.data(), slot.column_counts.data(), slot.inner_dimensions.data(),
        slot.alphas.data(), slot.device_matrix_a_pointers, slot.matrix_a_strides.data(),
        slot.device_matrix_b_pointers, slot.matrix_b_strides.data(), slot.betas.data(),
        slot.device_output_pointers, slot.output_strides.data(), int(slot.group_sizes.size()),
        slot.group_sizes.data()));
  }
  impl_->cached_input = B;
  impl_->cached_output = C;
}

void slot_split_baseline(const HostMatrix& a, const float* B, float* C, int rhs,
                         cudaStream_t stream) {
  GroupedGemmPlan p(a, rhs);
  p.execute(B, C, stream);
  cuda_check(cudaStreamSynchronize(stream));
}
} // namespace vbsr
