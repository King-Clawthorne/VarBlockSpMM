#include <stdexcept>

#include "varblockspmm/vbsr.hpp"
namespace vbsr {
Plan::Plan(const Matrix& matrix, PlanOptions options)
    : a_(matrix.device_view()), row_shape_order_(matrix.row_shape_order_),
      small_row_count_(matrix.small_row_count_), large_row_count_(matrix.large_row_count_),
      options_(options) {
  if (options.rhs_width != 8 && options.rhs_width != 16 && options.rhs_width != 32 &&
      options.rhs_width != 64) {
    throw std::invalid_argument("rhs_width must be 8,16,32,64");
  }
  if (options.kernel == Kernel::SplitRow) {
    throw std::invalid_argument(
        "split-row was not justified by the 1.0 regime map; use RowOwned or "
        "GroupedGemmPlan");
  }
}
void Plan::execute(const float* input, float* output, cudaStream_t stream) const {
  launch_row_owned(a_, row_shape_order_, small_row_count_, large_row_count_, input, output,
                   options_.rhs_width, stream);
}
int Plan::launch_count() const {
  if (options_.rhs_width != 32) {
    return 1;
  }
  if (small_row_count_ != 0 && large_row_count_ != 0 && a_.nnzb < 8 * a_.block_rows) {
    return 1;
  }
  return int(small_row_count_ != 0) + int(large_row_count_ != 0);
}
} // namespace vbsr
