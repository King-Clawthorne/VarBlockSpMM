#include "dense.hpp"
#include "support.hpp"
#include <cublas_v2.h>

namespace vbsr::bench {
namespace {
void check_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("dense cuBLAS failure: " + std::to_string(int(status)));
}
}
struct DensePlan::Impl {
  cublasHandle_t handle{};
  DeviceBuffer<float> matrix;
  int rows, columns, rhs;
  const float* input;
  float* output;
  Impl(size_t count, int r, int c, int n, const float* b, float* out)
      : matrix(count), rows(r), columns(c), rhs(n), input(b), output(out) {}
  ~Impl() { if (handle) cublasDestroy(handle); }
};
DensePlan::DensePlan(const CompactCsrData& csr, int64_t rows, int64_t columns, int rhs,
                     const float* input, float* output) {
  csr.validate(rows, columns);
  validate_panel_width(rhs);
  if (rows > INT32_MAX || columns > INT32_MAX ||
      uint64_t(rows) > std::numeric_limits<size_t>::max() / sizeof(float) / uint64_t(columns))
    throw std::invalid_argument("dense control dimensions exceed supported range");
  std::vector<float> dense(size_t(rows) * size_t(columns), 0.0f);
  for (int64_t row = 0; row < rows; ++row)
    for (int index = csr.row_offsets[row]; index < csr.row_offsets[row + 1]; ++index)
      dense[row + int64_t(csr.column_indices[index]) * rows] = csr.values[index];
  impl_ = std::make_unique<Impl>(dense.size(), int(rows), int(columns), rhs, input, output);
  impl_->matrix.upload(dense);
  check_blas(cublasCreate(&impl_->handle));
  // SGEMM uses the same default cuBLAS math contract as the grouped control.
}
DensePlan::~DensePlan() = default;
void DensePlan::execute() {
  constexpr float one = 1, zero = 0;
  check_blas(cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         impl_->rows, impl_->rhs, impl_->columns, &one,
                         impl_->matrix.data(), impl_->rows, impl_->input, impl_->columns,
                         &zero, impl_->output, impl_->rows));
}
size_t DensePlan::storage_bytes() const { return impl_->matrix.size() * sizeof(float); }
}
