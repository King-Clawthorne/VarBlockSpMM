#include "application_data.hpp"
#include "support.hpp"

#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

namespace vbsr::bench {

enum class CsrAlgorithm { One = 1, Two = 2, Three = 3 };

class CompactCsrPlan {
public:
  CompactCsrPlan(const CompactCsrData& data, int64_t rows, int64_t columns, int rhs_width,
                 float* input, float* output, CsrAlgorithm algorithm);
  ~CompactCsrPlan();
  CompactCsrPlan(const CompactCsrPlan&) = delete;
  CompactCsrPlan& operator=(const CompactCsrPlan&) = delete;
  void execute();
  size_t storage_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

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
} // namespace
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
                                   row_offsets.data(), column_indices.data(), sparse_values.data(),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
    sparse_check(cusparseCreateDnMat(&input_matrix, column_count, rhs_width, column_count, input,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL));
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
      sparse_check(cusparseSpMM_preprocess(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
          sparse_matrix, input_matrix, &zero, output_matrix, CUDA_R_32F, algorithm, workspace));
  }
  void execute() {
    float one = 1, zero = 0;
    sparse_check(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, input_matrix,
                              &zero, output_matrix, CUDA_R_32F, algorithm, workspace));
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
         impl_->column_indices.size() * sizeof(int32_t) + impl_->sparse_values.size() * sizeof(float);
}

// Persistent dense SGEMM control for application inputs small enough to expand.
class DensePlan {
public:
  DensePlan(const CompactCsrData&, int64_t rows, int64_t columns, int rhs,
            const float* input, float* output);
  ~DensePlan();
  DensePlan(const DensePlan&) = delete;
  DensePlan& operator=(const DensePlan&) = delete;
  void execute();
  size_t storage_bytes() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

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

} // namespace vbsr::bench

namespace {
namespace bench = vbsr::bench;

struct Arguments {
  std::filesystem::path input_path;
  int rhs_width;
  unsigned order_seed;
  int repetitions;
};

Arguments parse_arguments(int count, char** values) {
  if (count != 5) {
    throw std::invalid_argument("usage: application input.bin rhs order_seed reps");
  }
  Arguments args{values[1], bench::parse_integer<int>(values[2], "RHS width"),
                 bench::parse_integer<unsigned>(values[3], "order seed"),
                 bench::parse_integer<int>(values[4], "repetitions")};
  bench::validate_panel_width(args.rhs_width);
  bench::validate_repetitions(args.repetitions);
  return args;
}

enum class Method { Direct, CsrOne, CsrTwo, CsrThree, GroupedCached, BsrEight, Dense };
std::string_view method_name(Method method) {
  switch (method) {
  case Method::Direct:
    return "direct";
  case Method::CsrOne:
    return "compact_alg1_pre";
  case Method::CsrTwo:
    return "compact_alg2";
  case Method::CsrThree:
    return "compact_alg3_pre";
  case Method::GroupedCached:
    return "grouped_cached";
  case Method::BsrEight:
    return "bsr8";
  case Method::Dense:
    return "dense";
  }
  throw std::invalid_argument("unknown application method");
}

class ApplicationAudit {
public:
  explicit ApplicationAudit(const Arguments& args)
      : args_(args), data_(bench::load_application(args.input_path)),
        host_input_(
            bench::make_input(bench::panel_elements(data_.packed.scalar_cols(), args.rhs_width))),
        input_(host_input_.size()),
        output_(bench::panel_elements(data_.packed.scalar_rows(), args.rhs_width)),
        reference_(data_.compact.reference(host_input_, data_.packed.scalar_rows(),
                                           data_.packed.scalar_cols(), args.rhs_width)) {
    input_.upload(host_input_);
  }

  void run() {
    bench::print_environment();
    std::vector<Method> methods = {Method::Direct, Method::CsrOne, Method::CsrTwo, Method::CsrThree,
                                   Method::GroupedCached, Method::BsrEight, Method::Dense};
    std::mt19937 random_engine(args_.order_seed);
    std::shuffle(methods.begin(), methods.end(), random_engine);
    bench::print_timing_header();
    for (size_t position = 0; position < methods.size(); ++position)
      run_method(methods[position], int(position));
  }

private:
  Arguments args_;
  bench::ApplicationData data_;
  std::vector<float> host_input_;
  bench::DeviceBuffer<float> input_, output_;
  std::vector<float> reference_;

  void measure(Method method, int position, const std::function<void(int)>& operation) {
    bench::check_cuda(cudaMemset(output_.data(), 0xff, output_.size() * sizeof(float)));
    bench::measure(
        method_name(method), operation, [&] { bench::verify_output(reference_, output_.data()); },
        args_.repetitions, position);
  }

  void setup_record(Method method, std::chrono::steady_clock::time_point start, size_t bytes) {
    bench::check_cuda(cudaDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cerr << "setup," << method_name(method) << ',' << ms << ',' << bytes << '\n';
  }

  void measure_sparse(Method method, int position, bench::CsrAlgorithm algorithm) {
    auto start = std::chrono::steady_clock::now();
    bench::CompactCsrPlan plan(data_.compact, data_.packed.scalar_rows(),
                               data_.packed.scalar_cols(), args_.rhs_width, input_.data(),
                               output_.data(), algorithm);
    plan.execute();
    setup_record(method, start, plan.storage_bytes());
    measure(method, position, [&](int) { plan.execute(); });
  }

  void run_method(Method method, int position) {
    switch (method) {
    case Method::Direct: {
      auto start = std::chrono::steady_clock::now();
      vbsr::Matrix device(data_.packed);
      vbsr::Plan plan(device, {args_.rhs_width, vbsr::Kernel::RowOwned});
      plan.execute(input_.data(), output_.data());
      setup_record(method, start, device.storage_bytes());
      measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
      break;
    }
    case Method::GroupedCached: {
      auto start = std::chrono::steady_clock::now();
      vbsr::GroupedGemmPlan plan(data_.packed, args_.rhs_width, true);
      plan.execute(input_.data(), output_.data());
      setup_record(method, start, plan.storage_bytes());
      measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
      break;
    }
    case Method::BsrEight: {
      auto start = std::chrono::steady_clock::now();
      vbsr::ScalarCsrPlan plan(data_.packed, args_.rhs_width, 0, false, true, 8);
      plan.execute(input_.data(), output_.data());
      setup_record(method, start, plan.storage_bytes());
      measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
      break;
    }
    case Method::Dense: {
      auto start = std::chrono::steady_clock::now();
      bench::DensePlan plan(data_.compact, data_.packed.scalar_rows(), data_.packed.scalar_cols(),
                             args_.rhs_width, input_.data(), output_.data());
      plan.execute();
      setup_record(method, start, plan.storage_bytes());
      measure(method, position, [&](int) { plan.execute(); });
      break;
    }
    case Method::CsrOne:
      measure_sparse(method, position, bench::CsrAlgorithm::One);
      break;
    case Method::CsrTwo:
      measure_sparse(method, position, bench::CsrAlgorithm::Two);
      break;
    case Method::CsrThree:
      measure_sparse(method, position, bench::CsrAlgorithm::Three);
      break;
    }
  }
};
} // namespace

int main(int argc, char** argv) {
  try {
    ApplicationAudit audit(parse_arguments(argc, argv));
    audit.run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
