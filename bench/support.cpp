#include "support.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace vbsr::bench {
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
void validate_panel_width(int rhs_width) {
  switch (rhs_width) {
  case 8:
  case 16:
  case 32:
  case 64:
    return;
  default:
    throw std::invalid_argument("RHS width must be 8, 16, 32, or 64");
  }
}
void validate_repetitions(int repetitions) {
  if (repetitions < 2)
    throw std::invalid_argument("at least two repetitions required");
}
bool parse_locality(std::string_view locality) {
  if (locality == "local")
    return true;
  if (locality == "random")
    return false;
  throw std::invalid_argument("locality must be local or random");
}
Distribution parse_distribution(std::string_view name) {
  if (name == "uniform")
    return Distribution::Uniform;
  if (name == "low")
    return Distribution::LowVariance;
  if (name == "high")
    return Distribution::HighVariance;
  if (name == "bimodal")
    return Distribution::Bimodal;
  throw std::invalid_argument("invalid block-size distribution");
}
size_t panel_elements(int64_t scalar_rows, int rhs_width) {
  validate_panel_width(rhs_width);
  if (scalar_rows <= 0 ||
      uint64_t(scalar_rows) > std::numeric_limits<size_t>::max() / size_t(rhs_width)) {
    throw std::invalid_argument("invalid dense panel size");
  }
  return size_t(scalar_rows) * size_t(rhs_width);
}
std::vector<float> make_input(size_t count) {
  std::vector<float> input(count);
  std::mt19937 random_engine(input_seed);
  std::uniform_real_distribution<float> value(-1.f, 1.f);
  for (float& element : input)
    element = value(random_engine);
  return input;
}
void print_environment(const HostMatrix* matrix) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0));
  int driver, runtime;
  check_cuda(cudaDriverGetVersion(&driver));
  check_cuda(cudaRuntimeGetVersion(&runtime));
  std::cerr << "gpu=" << properties.name << ",cc=" << properties.major << '.' << properties.minor
            << ",driver_api=" << driver << ",runtime=" << runtime;
  if (matrix) {
    std::cerr << ",scalar_rows=" << matrix->scalar_rows()
              << ",scalar_cols=" << matrix->scalar_cols() << ",blocks=" << matrix->block_col.size()
              << ",values=" << matrix->values.size();
  }
  std::cerr << '\n';
}
void print_timing_header() {
  std::cout << std::setprecision(9) << "method,position,iteration,gpu_ms,host_ms\n";
}
void verify_output(const std::vector<float>& reference, const float* output) {
  std::vector<float> actual(reference.size());
  check_cuda(
      cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double squared_error = 0, reference_norm = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    double error = double(actual[index]) - reference[index];
    if (!std::isfinite(actual[index]) || std::abs(error) > absolute_tolerance) {
      throw std::runtime_error("output absolute error exceeds tolerance at index " +
                               std::to_string(index));
    }
    squared_error += error * error;
    reference_norm += double(reference[index]) * reference[index];
  }
  if (std::sqrt(squared_error / (reference_norm + 1e-30)) > relative_tolerance) {
    throw std::runtime_error("output relative error exceeds tolerance");
  }
}

void verify_probes(const vbsr::HostMatrix& matrix, const std::vector<float>& input,
                   const float* output, int rhs_width) {
  std::mt19937 random_engine(711);
  for (int trial = 0; trial < 64; ++trial) {
    int row = trial < 4 ? trial % matrix.block_rows : int(random_engine() % matrix.block_rows);
    int local_row = int(random_engine() % matrix.row_size[row]);
    int panel_column = int(random_engine() % rhs_width);
    double expected = 0;
    for (int block_index = matrix.row_ptr[row]; block_index < matrix.row_ptr[row + 1];
         ++block_index) {
      int block_column = matrix.block_col[block_index];
      for (int local_column = 0; local_column < matrix.col_size[block_column]; ++local_column) {
        expected += double(matrix.values[matrix.value_off[block_index] + local_row +
                                         local_column * matrix.row_size[row]]) *
                    input[matrix.col_scalar_off[block_column] + local_column +
                          panel_column * matrix.scalar_cols()];
      }
    }
    float actual;
    check_cuda(cudaMemcpy(&actual,
                          output + matrix.row_scalar_off[row] + local_row +
                              panel_column * matrix.scalar_rows(),
                          sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute_tolerance) {
      throw std::runtime_error("audit CPU probe mismatch");
    }
  }
}

void measure(std::string_view name, const std::function<void(int)>& operation,
             const std::function<void()>& validate, int reps, int position) {
  for (int i = 0; i < warmup_count; ++i)
    operation(i);
  check_cuda(cudaDeviceSynchronize());
  validate();
  Event begin, end;
  for (int i = 0; i < reps; ++i) {
    auto start = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(begin.get()));
    operation(i);
    check_cuda(cudaEventRecord(end.get()));
    check_cuda(cudaEventSynchronize(end.get()));
    auto finish = std::chrono::steady_clock::now();
    float gpu;
    check_cuda(cudaEventElapsedTime(&gpu, begin.get(), end.get()));
    double host = std::chrono::duration<double, std::milli>(finish - start).count();
    if (!(gpu > 0) || !std::isfinite(gpu))
      throw std::runtime_error("invalid timing");
    std::cout << name << ',' << position << ',' << i << ',' << gpu << ',' << host << '\n';
  }
}
} // namespace vbsr::bench
