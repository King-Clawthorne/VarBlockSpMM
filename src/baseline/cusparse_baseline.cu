#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "varblockspmm/vbsr.hpp"

namespace vbsr {
namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check_cusparse(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE failure: " + std::to_string(int(status)));
  }
}

bool is_supported_rhs_width(int rhs_width) {
  return rhs_width == 8 || rhs_width == 16 || rhs_width == 32 || rhs_width == 64;
}

struct ScalarCsrData {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<float> values;
};

ScalarCsrData expand_to_scalar_csr(const HostMatrix& matrix) {
  if (matrix.values.size() > size_t(INT32_MAX) || matrix.scalar_cols() > INT32_MAX) {
    throw std::invalid_argument("scalar CSR requires at most INT32_MAX entries and columns");
  }
  ScalarCsrData csr;
  csr.row_offsets.resize(matrix.scalar_rows() + 1);

  int32_t nonzero_count = 0;
  for (int block_row = 0; block_row < matrix.block_rows; ++block_row) {
    const int row_height = matrix.row_size[block_row];

    for (int local_row = 0; local_row < row_height; ++local_row) {
      const int64_t scalar_row = matrix.row_scalar_off[block_row] + local_row;
      csr.row_offsets[scalar_row] = nonzero_count;

      for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1];
           ++block_index) {
        const int block_column = matrix.block_col[block_index];
        const int column_width = matrix.col_size[block_column];

        for (int local_column = 0; local_column < column_width; ++local_column) {
          csr.column_indices.push_back(int32_t(matrix.col_scalar_off[block_column] + local_column));
          csr.values.push_back(
              matrix.values[matrix.value_off[block_index] + local_row + local_column * row_height]);
          ++nonzero_count;
        }
      }
    }
  }

  csr.row_offsets.back() = nonzero_count;
  return csr;
}

cusparseSpMMAlg_t select_csr_algorithm(int algorithm) {
  switch (algorithm) {
  case 0:
    return CUSPARSE_SPMM_ALG_DEFAULT;
  case 1:
    return CUSPARSE_SPMM_CSR_ALG1;
  case 2:
    return CUSPARSE_SPMM_CSR_ALG2;
  case 3:
    return CUSPARSE_SPMM_CSR_ALG3;
  default:
    throw std::invalid_argument("invalid CSR algorithm");
  }
}

ScalarCsrData pack_fixed_bsr(const HostMatrix& matrix, int size) {
  ScalarCsrData csr;
  if (size != 8 && size != 32) {
    throw std::invalid_argument("BSR block size must be 8 or 32");
  }
  for (int extent : matrix.row_size)
    if (extent % size) throw std::invalid_argument("row partition cannot be subdivided into BSR");
  for (int extent : matrix.col_size)
    if (extent % size) throw std::invalid_argument("column partition cannot be subdivided into BSR");
  if (matrix.values.size() / (size * size) > size_t(INT32_MAX) ||
      matrix.scalar_cols() / size > INT32_MAX) {
    throw std::invalid_argument("BSR index range exceeds INT32_MAX");
  }
  csr.row_offsets.push_back(0);
  csr.values.reserve(matrix.values.size());
  for (int row = 0; row < matrix.block_rows; ++row) {
    const int height = matrix.row_size[row];
    for (int r = 0; r < height; r += size) {
      for (int block = matrix.row_ptr[row]; block < matrix.row_ptr[row + 1]; ++block) {
        const int column = matrix.block_col[block];
        for (int c = 0; c < matrix.col_size[column]; c += size) {
          csr.column_indices.push_back(int32_t((matrix.col_scalar_off[column] + c) / size));
          for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
              csr.values.push_back(matrix.values[matrix.value_off[block] + r + i + (c + j) * height]);
        }
      }
      csr.row_offsets.push_back(int32_t(csr.column_indices.size()));
    }
  }
  return csr;
}

template <typename T> T* copy_to_device(const std::vector<T>& source) {
  T* destination = nullptr;
  check_cuda(cudaMalloc(&destination, source.size() * sizeof(T)));
  try {
    check_cuda(
        cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
  } catch (...) {
    cudaFree(destination);
    throw;
  }
  return destination;
}

} // namespace

struct ScalarCsrPlan::Impl {
  cusparseSpMMAlg_t algorithm = CUSPARSE_SPMM_ALG_DEFAULT;
  bool preprocess = false;
  int rhs_width{};
  int64_t row_count{};
  int64_t column_count{};
  int64_t nonzero_count{};

  int32_t* row_offsets{};
  int32_t* column_indices{};
  float* values{};
  void* workspace{};
  size_t workspace_size{};
  size_t format_bytes{};

  cusparseHandle_t handle{};
  cusparseSpMatDescr_t sparse_matrix{};
  cusparseDnMatDescr_t input_matrix{};
  cusparseDnMatDescr_t output_matrix{};

  ~Impl() {
    cudaFree(workspace);
    if (input_matrix) {
      cusparseDestroyDnMat(input_matrix);
    }
    if (output_matrix) {
      cusparseDestroyDnMat(output_matrix);
    }
    if (sparse_matrix) {
      cusparseDestroySpMat(sparse_matrix);
    }
    if (handle) {
      cusparseDestroy(handle);
    }
    cudaFree(row_offsets);
    cudaFree(column_indices);
    cudaFree(values);
  }

  void create_dense_descriptors(const float* input, float* output) {
    check_cusparse(cusparseCreateDnMat(&input_matrix, column_count, rhs_width, column_count,
                                       const_cast<float*>(input), CUDA_R_32F, CUSPARSE_ORDER_COL));
    check_cusparse(cusparseCreateDnMat(&output_matrix, row_count, rhs_width, row_count, output,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));
  }

  void allocate_workspace() {
    constexpr float one = 1.0f;
    constexpr float zero = 0.0f;

    check_cusparse(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        sparse_matrix, input_matrix, &zero, output_matrix, CUDA_R_32F, algorithm, &workspace_size));

    if (workspace_size > 0) {
      check_cuda(cudaMalloc(&workspace, workspace_size));
    }
  }

  void update_dense_pointers(const float* input, float* output) {
    check_cusparse(cusparseDnMatSetValues(input_matrix, const_cast<float*>(input)));
    check_cusparse(cusparseDnMatSetValues(output_matrix, output));
  }
};

ScalarCsrPlan::ScalarCsrPlan(const HostMatrix& matrix, int rhs_width, int algorithm,
                             bool preprocess, bool fixed_bsr, int bsr_block_size)
    : impl_(new Impl) {
  matrix.validate();
  if (!is_supported_rhs_width(rhs_width)) {
    throw std::invalid_argument("invalid rhs width");
  }

  impl_->algorithm = select_csr_algorithm(algorithm);
  impl_->preprocess = preprocess;
  ScalarCsrData csr;
  if (fixed_bsr) {
    csr = pack_fixed_bsr(matrix, bsr_block_size);
    impl_->algorithm = CUSPARSE_SPMM_BSR_ALG1;
    impl_->preprocess = false;
  } else {
    csr = expand_to_scalar_csr(matrix);
  }
  impl_->rhs_width = rhs_width;
  impl_->row_count = matrix.scalar_rows();
  impl_->column_count = matrix.scalar_cols();
  impl_->nonzero_count = int64_t(csr.values.size());
  impl_->row_offsets = copy_to_device(csr.row_offsets);
  impl_->column_indices = copy_to_device(csr.column_indices);
  impl_->values = copy_to_device(csr.values);
  impl_->format_bytes = csr.values.size() * sizeof(float) +
                        (csr.row_offsets.size() + csr.column_indices.size()) * sizeof(int32_t);

  check_cusparse(cusparseCreate(&impl_->handle));
  if (fixed_bsr) {
    check_cusparse(cusparseCreateBsr(
        &impl_->sparse_matrix, matrix.scalar_rows() / bsr_block_size,
        matrix.scalar_cols() / bsr_block_size, csr.column_indices.size(), bsr_block_size,
        bsr_block_size, impl_->row_offsets, impl_->column_indices, impl_->values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    return;
  }
  check_cusparse(cusparseCreateCsr(&impl_->sparse_matrix, impl_->row_count, impl_->column_count,
                                   impl_->nonzero_count, impl_->row_offsets, impl_->column_indices,
                                   impl_->values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
}

ScalarCsrPlan::~ScalarCsrPlan() = default;

size_t ScalarCsrPlan::workspace_bytes() const { return impl_->workspace_size; }
size_t ScalarCsrPlan::storage_bytes() const { return impl_->format_bytes + impl_->workspace_size; }

void ScalarCsrPlan::execute(const float* input, float* output, cudaStream_t stream) {
  check_cusparse(cusparseSetStream(impl_->handle, stream));

  if (!impl_->input_matrix) {
    impl_->create_dense_descriptors(input, output);
    impl_->allocate_workspace();
    if (impl_->preprocess) {
      constexpr float one = 1.0f, zero = 0.0f;
      check_cusparse(cusparseSpMM_preprocess(
          impl_->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
          impl_->sparse_matrix, impl_->input_matrix, &zero, impl_->output_matrix, CUDA_R_32F,
          impl_->algorithm, impl_->workspace));
    }
  } else {
    impl_->update_dense_pointers(input, output);
  }

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  check_cusparse(cusparseSpMM(impl_->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &one, impl_->sparse_matrix,
                              impl_->input_matrix, &zero, impl_->output_matrix, CUDA_R_32F,
                              impl_->algorithm, impl_->workspace));
}

void cusparse_scalar_baseline(const HostMatrix& matrix, const float* input, float* output,
                              int rhs_width, cudaStream_t stream) {
  ScalarCsrPlan plan(matrix, rhs_width);
  plan.execute(input, output, stream);
  check_cuda(cudaStreamSynchronize(stream));
}

} // namespace vbsr
