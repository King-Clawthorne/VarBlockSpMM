#include "dg_fused.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vbsr::bench {
namespace {

constexpr int classes = 4;  // The order schedule repeats with period four.

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
}

// Everything an element needs, derived from its index. No per-element metadata.
struct FusedView {
  int elements;
  int64_t scalar_rows;
  int64_t cycle;              // Scalar rows spanned by one period of four.
  const float* operators;
  int size[classes];          // Row height of each order class.
  int upstream_size[classes]; // Column width of the upwind block.
  int64_t prefix[classes];    // Scalar offset of the class within a period.
  int64_t self_offset[classes];
  int64_t upstream_offset[classes];
  // Zero when one copy of each class operator serves every element. Otherwise
  // the floats one period of four elements occupies in the duplicated buffer.
  int64_t period_stride;
};

__device__ inline int64_t element_offset(const FusedView& view, int element) {
  return int64_t(element / classes) * view.cycle + view.prefix[element % classes];
}

__device__ inline int64_t operator_base(const FusedView& view, int element) {
  return int64_t(element / classes) * view.period_stride;
}

// Mirrors row_owned_single_buffered for the narrow widths and
// row_owned_single_buffered_direct for RHS 64. The thread mapping, staging,
// accumulation and single write are the release kernels'. The operand source
// is the only change: `operators` holds one copy per order class.
template <int RHS, int VectorWidth, int Threads>
__global__ __launch_bounds__(Threads) void dg_fused_kernel(FusedView view,
                                                           const float* __restrict__ input,
                                                           float* __restrict__ output) {
  constexpr int max_block_width = 64;
  constexpr int shared_stride = max_block_width + 1;
  __shared__ float shared_input[RHS * shared_stride];

  const int element = blockIdx.x;
  const int order = element % classes;
  const int row_height = view.size[order];
  const int upstream = (element + view.elements - 1) % view.elements;

  const int rhs_groups = RHS / VectorWidth;
  const int work_index = threadIdx.x;
  const bool active = work_index < row_height * rhs_groups;
  const int local_row = work_index % row_height;
  const int first_rhs_column = (work_index / row_height) * VectorWidth;
  float accumulators[VectorWidth] = {};

  // Contribution 0 is the element's own block, contribution 1 its upwind
  // neighbor. Both accumulate into the same registers, so the output is
  // written once by its single owner.
#pragma unroll 1
  for (int contribution = 0; contribution < 2; ++contribution) {
    const int column_width = contribution == 0 ? row_height : view.upstream_size[order];
    const int64_t column_offset =
        contribution == 0 ? element_offset(view, element) : element_offset(view, upstream);
    const float* block_values =
        view.operators + operator_base(view, element) +
        (contribution == 0 ? view.self_offset[order] : view.upstream_offset[order]) + local_row;

    const int local_column = threadIdx.x % max_block_width;
    for (int rhs_column = threadIdx.x / max_block_width; rhs_column < RHS;
         rhs_column += Threads / max_block_width) {
      if (local_column < column_width) {
        shared_input[rhs_column * shared_stride + local_column] =
            input[column_offset + local_column + int64_t(rhs_column) * view.scalar_rows];
      }
    }
    __syncthreads();

    if (active) {
#pragma unroll 4
      for (int column = 0; column < column_width; ++column) {
        const float matrix_value = block_values[column * row_height];
#pragma unroll
        for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
          accumulators[vector_index] =
              fmaf(matrix_value,
                   shared_input[(first_rhs_column + vector_index) * shared_stride + column],
                   accumulators[vector_index]);
        }
      }
    }
    if (contribution == 0)
      __syncthreads();
  }

  if (active) {
    const int64_t row_offset = element_offset(view, element);
#pragma unroll
    for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
      output[row_offset + local_row + int64_t(first_rhs_column + vector_index) * view.scalar_rows] =
          accumulators[vector_index];
    }
  }
}

} // namespace

struct DgFusedPlan::Impl {
  int rhs{};
  FusedView view{};
  float* operators{};
  size_t operator_bytes{};
  ~Impl() { cudaFree(operators); }
};

DgFusedPlan::DgFusedPlan(const HostMatrix& a, int rhs, bool share_operators)
    : impl_(new Impl) {
  a.validate();
  if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
    throw std::invalid_argument("invalid rhs width");
  const int elements = a.block_rows;
  if (elements < classes || elements % classes || a.block_cols != elements)
    throw std::invalid_argument("fused comparator needs a periodic element cycle");
  impl_->rhs = rhs;

  FusedView view{};
  view.elements = elements;
  view.scalar_rows = a.scalar_rows();
  view.cycle = a.row_scalar_off[classes];
  for (int order = 0; order < classes; ++order) {
    view.size[order] = a.row_size[order];
    view.prefix[order] = a.row_scalar_off[order];
    view.upstream_size[order] = a.col_size[(order + elements - 1) % elements];
  }

  // The assumed structure must hold for every element, not just the first
  // period. Verify offsets, degrees, columns and shapes before trusting it.
  for (int element = 0; element < elements; ++element) {
    const int order = element % classes;
    const int64_t expected = int64_t(element / classes) * view.cycle + view.prefix[order];
    if (a.row_size[element] != view.size[order] || a.col_size[element] != view.size[order] ||
        a.row_scalar_off[element] != expected || a.col_scalar_off[element] != expected)
      throw std::invalid_argument("fused comparator: element sizes or offsets are not periodic");
    if (a.row_ptr[element + 1] - a.row_ptr[element] != 2)
      throw std::invalid_argument("fused comparator: every element needs two blocks");
    const int upstream = (element + elements - 1) % elements;
    int seen_self = 0, seen_upstream = 0;
    for (int p = a.row_ptr[element]; p < a.row_ptr[element + 1]; ++p) {
      seen_self += a.block_col[p] == element;
      seen_upstream += a.block_col[p] == upstream;
    }
    if (seen_self != 1 || seen_upstream != 1)
      throw std::invalid_argument("fused comparator: blocks are not self and upwind neighbor");
  }

  // Collect one copy of each class operator, then check every element's packed
  // payload against it. A mismatch means the operator does not actually repeat.
  std::vector<float> operators;
  auto block_of = [&](int element, int column) {
    for (int p = a.row_ptr[element]; p < a.row_ptr[element + 1]; ++p)
      if (a.block_col[p] == column)
        return p;
    throw std::invalid_argument("fused comparator: missing expected block");
  };
  for (int order = 0; order < classes; ++order) {
    const int upstream = (order + elements - 1) % elements;
    for (int contribution = 0; contribution < 2; ++contribution) {
      const int column = contribution == 0 ? order : upstream;
      const int p = block_of(order, column);
      const size_t count = size_t(a.row_size[order]) * a.col_size[column];
      const int64_t offset = int64_t(operators.size());
      if (contribution == 0)
        view.self_offset[order] = offset;
      else
        view.upstream_offset[order] = offset;
      const float* payload = a.values.data() + a.value_off[p];
      operators.insert(operators.end(), payload, payload + count);
    }
  }
  for (int element = 0; element < elements; ++element) {
    const int order = element % classes;
    const int upstream = (element + elements - 1) % elements;
    for (int contribution = 0; contribution < 2; ++contribution) {
      const int column = contribution == 0 ? element : upstream;
      const int p = block_of(element, column);
      const size_t count = size_t(a.row_size[element]) * a.col_size[column];
      const int64_t offset =
          contribution == 0 ? view.self_offset[order] : view.upstream_offset[order];
      if (std::memcmp(a.values.data() + a.value_off[p], operators.data() + offset,
                      count * sizeof(float)) != 0)
        throw std::invalid_argument("fused comparator: block payloads do not repeat by order");
    }
  }

  // The matched control repeats the verified period across every element, so
  // each one reads a private copy of bytes that are identical in value. The
  // kernel, thread mapping, addressing arithmetic and contribution order are
  // unchanged, which leaves operand sharing as the only difference.
  const int64_t period = int64_t(operators.size());
  view.period_stride = share_operators ? 0 : period;
  std::vector<float> payload;
  if (share_operators) {
    payload = std::move(operators);
  } else {
    payload.reserve(size_t(period) * (elements / classes));
    for (int start = 0; start < elements; start += classes)
      payload.insert(payload.end(), operators.begin(), operators.end());
  }

  impl_->operator_bytes = payload.size() * sizeof(float);
  cuda_check(cudaMalloc(&impl_->operators, impl_->operator_bytes));
  cuda_check(cudaMemcpy(impl_->operators, payload.data(), impl_->operator_bytes,
                        cudaMemcpyHostToDevice));
  // Pageable host uploads may return before the device transfer completes.
  // Construction must make the data ready for execution on any stream.
  cuda_check(cudaStreamSynchronize(nullptr));
  view.operators = impl_->operators;
  impl_->view = view;
}

DgFusedPlan::~DgFusedPlan() = default;
int DgFusedPlan::distinct_operators() const { return 2 * classes; }
int DgFusedPlan::launch_count() const { return 1; }
size_t DgFusedPlan::operator_bytes() const { return impl_->operator_bytes; }
size_t DgFusedPlan::storage_bytes() const { return impl_->operator_bytes; }

void DgFusedPlan::execute(const float* input, float* output, cudaStream_t stream) {
  const FusedView& view = impl_->view;
  // The same width policy the release dispatcher uses for these panels.
  switch (impl_->rhs) {
  case 8:
    dg_fused_kernel<8, 8, 64><<<view.elements, 64, 0, stream>>>(view, input, output);
    break;
  case 16:
    dg_fused_kernel<16, 16, 64><<<view.elements, 64, 0, stream>>>(view, input, output);
    break;
  case 32:
    dg_fused_kernel<32, 8, 256><<<view.elements, 256, 0, stream>>>(view, input, output);
    break;
  default:
    dg_fused_kernel<64, 16, 256><<<view.elements, 256, 0, stream>>>(view, input, output);
    break;
  }
  cuda_check(cudaGetLastError());
}

} // namespace vbsr::bench
