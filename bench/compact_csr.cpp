#include "compact_csr.hpp"
#include "support.hpp"
#include <cusparse.h>

namespace vbsr::bench {
namespace {
void sparse_check(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("compact CSR failure " + std::to_string(int(status)));
    }
}
cusparseSpMMAlg_t native_algorithm(CsrAlgorithm algorithm) {
    switch (algorithm) {
    case CsrAlgorithm::One:
        return CUSPARSE_SPMM_CSR_ALG1;
    case CsrAlgorithm::Two:
        return CUSPARSE_SPMM_CSR_ALG2;
    case CsrAlgorithm::Three:
        return CUSPARSE_SPMM_CSR_ALG3;
    }
    throw std::invalid_argument("unsupported CSR algorithm");
}
}
struct CompactCsrPlan::Impl {
    DeviceBuffer<int32_t> row_offsets;
    DeviceBuffer<int32_t> column_indices;
    DeviceBuffer<float> sparse_values;
    void* workspace{};
    size_t workspace_size{};
    cusparseHandle_t handle{};
    cusparseSpMatDescr_t sparse_matrix{};
    cusparseDnMatDescr_t input_matrix{}, output_matrix{};
    cusparseSpMMAlg_t algorithm{};

    explicit Impl(const CompactCsrData& data)
        : row_offsets(data.row_offsets.size()), column_indices(data.column_indices.size()),
          sparse_values(data.values.size()) {}
    ~Impl() {
        cudaFree(workspace);
        if (input_matrix)
            cusparseDestroyDnMat(input_matrix);
        if (output_matrix)
            cusparseDestroyDnMat(output_matrix);
        if (sparse_matrix)
            cusparseDestroySpMat(sparse_matrix);
        if (handle)
            cusparseDestroy(handle);
    }
    void initialize(const CompactCsrData& data, int64_t row_count, int64_t column_count,
                    int rhs_width, float* input, float* output, CsrAlgorithm selected_algorithm) {
        algorithm = native_algorithm(selected_algorithm);
        row_offsets.upload(data.row_offsets);
        column_indices.upload(data.column_indices);
        sparse_values.upload(data.values);
        sparse_check(cusparseCreate(&handle));
        sparse_check(cusparseCreateCsr(&sparse_matrix, row_count, column_count, data.values.size(),
                                       row_offsets.data(), column_indices.data(),
                                       sparse_values.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        sparse_check(cusparseCreateDnMat(&input_matrix, column_count, rhs_width, column_count,
                                         input, CUDA_R_32F, CUSPARSE_ORDER_COL));
        sparse_check(cusparseCreateDnMat(&output_matrix, row_count, rhs_width, row_count, output,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL));
        float one = 1, zero = 0;
        size_t bytes;
        sparse_check(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            sparse_matrix, input_matrix, &zero, output_matrix, CUDA_R_32F, algorithm, &bytes));
        if (bytes)
            check_cuda(cudaMalloc(&workspace, bytes));
        workspace_size = bytes;
        if (selected_algorithm != CsrAlgorithm::Two)
            sparse_check(cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                 sparse_matrix, input_matrix, &zero, output_matrix,
                                                 CUDA_R_32F, algorithm, workspace));
    }
    void execute() {
        float one = 1, zero = 0;
        sparse_check(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            sparse_matrix, input_matrix, &zero, output_matrix, CUDA_R_32F, algorithm, workspace));
    }
};
CompactCsrPlan::CompactCsrPlan(const CompactCsrData& data, int64_t rows, int64_t columns,
                               int rhs_width, float* input, float* output, CsrAlgorithm algorithm) {
    data.validate(rows, columns);
    validate_panel_width(rhs_width);
    impl_ = std::make_unique<Impl>(data);
    impl_->initialize(data, rows, columns, rhs_width, input, output, algorithm);
}
CompactCsrPlan::~CompactCsrPlan() = default;
void CompactCsrPlan::execute() { impl_->execute(); }
size_t CompactCsrPlan::storage_bytes() const {
    return impl_->workspace_size + impl_->row_offsets.size() * sizeof(int32_t) +
           impl_->column_indices.size() * sizeof(int32_t) +
           impl_->sparse_values.size() * sizeof(float);
}
}
