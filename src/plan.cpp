#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "vbsr.hpp"
namespace vbsr {
namespace {

void validate_options(PlanOptions options) {
  if (options.rhs_width != 8 && options.rhs_width != 16 && options.rhs_width != 32 &&
      options.rhs_width != 64) {
    throw std::invalid_argument("rhs_width must be one of 8, 16, 32, or 64");
  }
  if (options.kernel != Kernel::Auto && options.kernel != Kernel::RowOwned) {
    throw std::invalid_argument("unsupported kernel selection");
  }
}

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class T> std::vector<T> read_metadata(const T* source, size_t count) {
  if (count && !source)
    throw std::invalid_argument("null device metadata");
  std::vector<T> result(count);
  if (count)
    check(cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
}
Plan::Plan(DeviceMatrix matrix, PlanOptions options, cudaStream_t stream)
    : matrix_(matrix), options_(options) {
  validate_options(options);
  if (matrix.block_rows <= 0 || matrix.block_cols <= 0 || matrix.nnzb < 0)
    throw std::invalid_argument("invalid device matrix dimensions");
  check(cudaStreamSynchronize(stream));
  HostMatrix host;
  host.block_rows = matrix.block_rows;
  host.block_cols = matrix.block_cols;
  host.row_ptr = read_metadata(matrix.row_ptr, size_t(matrix.block_rows) + 1);
  host.block_col = read_metadata(matrix.block_col, matrix.nnzb);
  host.row_size = read_metadata(matrix.row_size, matrix.block_rows);
  host.col_size = read_metadata(matrix.col_size, matrix.block_cols);
  host.row_scalar_off = read_metadata(matrix.row_scalar_off, size_t(matrix.block_rows) + 1);
  host.col_scalar_off = read_metadata(matrix.col_scalar_off, size_t(matrix.block_cols) + 1);
  host.value_off = read_metadata(matrix.value_off, size_t(matrix.nnzb) + 1);
  if (host.value_off.back() < 0 || (host.value_off.back() && !matrix.values))
    throw std::invalid_argument("invalid device value payload");
  host.validate_structure(size_t(host.value_off.back()));
  if (host.scalar_rows() != matrix.scalar_rows || host.scalar_cols() != matrix.scalar_cols)
    throw std::invalid_argument("device scalar dimensions disagree with metadata");
  std::vector<int32_t> order(matrix.block_rows);
  std::iota(order.begin(), order.end(), 0);
  auto split = std::stable_partition(order.begin(), order.end(),
                                     [&](int r) { return host.row_size[r] <= 16; });
  small_row_count_ = int(split - order.begin());
  large_row_count_ = matrix.block_rows - small_row_count_;
  int32_t* allocation = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&allocation), order.size() * sizeof(int32_t)));
  owned_row_order_ = std::shared_ptr<int32_t>(allocation, [](int32_t* p) { cudaFree(p); });
  check(
      cudaMemcpy(allocation, order.data(), order.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

  check(cudaStreamSynchronize(nullptr));
  row_shape_order_ = allocation;
}
Plan::Plan(const Matrix& matrix, PlanOptions options)
    : matrix_(matrix.device_view()), row_shape_order_(matrix.row_shape_order_),
      small_row_count_(matrix.small_row_count_), large_row_count_(matrix.large_row_count_),
      options_(options) {
  validate_options(options);
}
void Plan::execute(const float* input, float* output, cudaStream_t stream) const {
  launch_row_owned(matrix_, row_shape_order_, small_row_count_, large_row_count_, input, output,
                   options_.rhs_width, stream);
}
int Plan::launch_count() const {
  if (options_.rhs_width != 32) {
    return 1;
  }
  if (small_row_count_ != 0 && large_row_count_ != 0 &&
      matrix_.nnzb < int64_t(8) * matrix_.block_rows) {
    return 1;
  }
  return int(small_row_count_ != 0) + int(large_row_count_ != 0);
}
}
