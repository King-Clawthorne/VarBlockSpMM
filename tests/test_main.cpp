#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "varblockspmm/vbsr.hpp"

namespace {

void compare_device_result(const std::vector<float>& reference, const float* device_result,
                           const char* implementation_name) {
  std::vector<float> result(reference.size());
  cudaMemcpy(result.data(), device_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

  double maximum_error = 0.0;
  double squared_error = 0.0;
  double reference_norm = 0.0;

  for (size_t index = 0; index < result.size(); ++index) {
    if (!std::isfinite(result[index])) {
      throw std::runtime_error(std::string(implementation_name) + " produced NaN/Inf");
    }

    const double error = result[index] - reference[index];
    maximum_error = std::max(maximum_error, std::abs(error));
    squared_error += error * error;
    reference_norm += double(reference[index]) * reference[index];
  }

  const double relative_error = std::sqrt(squared_error / (reference_norm + 1e-30));
  if (maximum_error > 5e-4 || relative_error > 5e-5) {
    throw std::runtime_error(std::string(implementation_name) +
                             " mismatch: max=" + std::to_string(maximum_error) +
                             ", rel=" + std::to_string(relative_error));
  }
}

std::vector<float> make_random_input(int64_t element_count, uint64_t seed) {
  std::vector<float> input(element_count);
  std::mt19937 random_engine{unsigned(seed)};
  std::uniform_real_distribution<float> random_value(-1.0f, 1.0f);

  for (float& value : input) {
    value = random_value(random_engine);
  }
  return input;
}

void execute_and_compare(const std::function<void()>& execute, cudaStream_t stream,
                         const std::vector<float>& reference, float* device_output,
                         const char* implementation_name) {
  execute();
  cudaStreamSynchronize(stream);
  compare_device_result(reference, device_output, implementation_name);
}

void run_case(vbsr::Distribution distribution, int degree, int rhs_width, bool local_columns,
              uint64_t seed) {
  vbsr::GeneratorOptions options;
  options.block_rows = 9;
  options.block_cols = 17;
  options.degree = degree;
  options.distribution = distribution;
  options.local_columns = local_columns;
  options.seed = seed;

  const vbsr::HostMatrix host_matrix = vbsr::generate(options);
  const std::vector<float> input = make_random_input(host_matrix.scalar_cols() * rhs_width, seed);
  const std::vector<float> reference = vbsr::cpu_reference(host_matrix, input, rhs_width);

  vbsr::Matrix device_matrix(host_matrix);
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaMalloc(&device_input, input.size() * sizeof(float));
  cudaMalloc(&device_output, reference.size() * sizeof(float));
  cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  vbsr::Plan direct_plan(device_matrix, {rhs_width, vbsr::Kernel::RowOwned});
  vbsr::ScalarCsrPlan scalar_csr_plan(host_matrix, rhs_width);
  vbsr::GroupedGemmPlan grouped_gemm_plan(host_matrix, rhs_width);

  execute_and_compare([&] { direct_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "row-owned ILP");
  execute_and_compare(
      [&] {
        vbsr::launch_row_owned_scalar(device_matrix.device_view(), device_input, device_output,
                                      rhs_width, stream);
      },
      stream, reference, device_output, "row-owned scalar");
  execute_and_compare([&] { scalar_csr_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "cuSPARSE");
  execute_and_compare([&] { grouped_gemm_plan.execute(device_input, device_output, stream); },
                      stream, reference, device_output, "grouped cuBLAS");

  cudaStreamDestroy(stream);
  cudaFree(device_input);
  cudaFree(device_output);
}

void expect_invalid_argument(const std::function<void()>& operation, const char* failure_message) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(failure_message);
}

void run_negative_tests() {
  vbsr::GeneratorOptions options;
  options.block_rows = 4;
  options.block_cols = 4;
  options.degree = 2;

  const vbsr::HostMatrix valid_matrix = vbsr::generate(options);
  vbsr::HostMatrix malformed_matrix = valid_matrix;
  malformed_matrix.row_ptr.back()++;

  expect_invalid_argument([&] { malformed_matrix.validate(); }, "malformed row_ptr accepted");

  vbsr::Matrix device_matrix(valid_matrix);
  expect_invalid_argument(
      [&] {
        vbsr::Plan plan(device_matrix, {7, vbsr::Kernel::RowOwned});
      },
      "invalid RHS accepted");
  expect_invalid_argument(
      [&] {
        vbsr::Plan plan(device_matrix, {8, vbsr::Kernel::SplitRow});
      },
      "unjustified split-row accepted");
}

void run_parameter_matrix() {
  constexpr vbsr::Distribution distributions[] = {
      vbsr::Distribution::Uniform, vbsr::Distribution::LowVariance,
      vbsr::Distribution::HighVariance, vbsr::Distribution::Bimodal};
  constexpr int degrees[] = {1, 4, 8, 16};
  constexpr int rhs_widths[] = {8, 16, 32, 64};

  uint64_t seed = 11;
  for (vbsr::Distribution distribution : distributions) {
    for (int degree : degrees) {
      for (int rhs_width : rhs_widths) {
        run_case(distribution, degree, rhs_width, (seed & 1) != 0, seed);
        ++seed;
      }
    }
  }
}

} // namespace

int main() {
  try {
    run_negative_tests();
    run_parameter_matrix();
    std::cout << "PASS: 64 parameter cases x hybrid/scalar/cuSPARSE/grouped "
                 "paths, non-default streams, validation failures, NaN/Inf "
                 "checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
