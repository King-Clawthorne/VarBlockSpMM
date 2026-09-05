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
              uint64_t seed, bool empty_first_row = false) {
  vbsr::GeneratorOptions options;
  options.block_rows = 9;
  options.block_cols = 17;
  options.degree = degree;
  options.distribution = distribution;
  options.local_columns = local_columns;
  options.seed = seed;

  vbsr::HostMatrix host_matrix = vbsr::generate(options);
  if (empty_first_row) {
    int removed_blocks = host_matrix.row_ptr[1];
    int64_t removed_values = host_matrix.value_off[removed_blocks];
    host_matrix.block_col.erase(host_matrix.block_col.begin(),
                                host_matrix.block_col.begin() + removed_blocks);
    host_matrix.values.erase(host_matrix.values.begin(),
                             host_matrix.values.begin() + removed_values);
    host_matrix.value_off.erase(host_matrix.value_off.begin(),
                                host_matrix.value_off.begin() + removed_blocks);
    for (auto& offset : host_matrix.value_off)
      offset -= removed_values;
    for (size_t i = 1; i < host_matrix.row_ptr.size(); ++i)
      host_matrix.row_ptr[i] -= removed_blocks;
    host_matrix.validate();
  }
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

  vbsr::GroupedGemmPlan cached_plan(host_matrix, rhs_width, true);
  execute_and_compare([&] { cached_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "cached grouped cuBLAS");
  execute_and_compare([&] { cached_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "cached grouped reuse");
  float* alternate_input = nullptr;
  cudaMalloc(&alternate_input, input.size() * sizeof(float));
  auto alternate_host = input;
  for (float& x : alternate_host)
    x *= 0.5f;
  cudaMemcpy(alternate_input, alternate_host.data(), alternate_host.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  const auto alternate_reference = vbsr::cpu_reference(host_matrix, alternate_host, rhs_width);
  execute_and_compare([&] { cached_plan.execute(alternate_input, device_output, stream); }, stream,
                      alternate_reference, device_output, "cached grouped changed address");
  execute_and_compare([&] { cached_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "cached grouped restored address");
  cudaFree(alternate_input);
  for (int algorithm = 1; algorithm <= 3; ++algorithm) {
    vbsr::ScalarCsrPlan explicit_plan(host_matrix, rhs_width, algorithm, algorithm != 2);
    execute_and_compare([&] { explicit_plan.execute(device_input, device_output, stream); }, stream,
                        reference, device_output, "explicit CSR algorithm");
  }
  if (distribution == vbsr::Distribution::Uniform) {
    vbsr::ScalarCsrPlan bsr_plan(host_matrix, rhs_width, 0, false, true);
    execute_and_compare([&] { bsr_plan.execute(device_input, device_output, stream); }, stream,
                        reference, device_output, "fixed BSR");
  }

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

  malformed_matrix = valid_matrix;
  malformed_matrix.row_size[0] = 7;
  expect_invalid_argument([&] { malformed_matrix.validate(); }, "unsupported block size accepted");

  for (int shift : {-1, 1}) {
    malformed_matrix = valid_matrix;
    for (auto& offset : malformed_matrix.row_scalar_off)
      offset += shift;
    expect_invalid_argument([&] { malformed_matrix.validate(); },
                            "nonzero row offset origin accepted");
    malformed_matrix = valid_matrix;
    for (auto& offset : malformed_matrix.col_scalar_off)
      offset += shift;
    expect_invalid_argument([&] { malformed_matrix.validate(); },
                            "nonzero column offset origin accepted");
  }

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

void run_reference_cancellation_test() {
  vbsr::HostMatrix matrix;
  matrix.block_rows = 2;
  matrix.block_cols = 3;
  matrix.row_ptr = {0, 3, 3};
  matrix.block_col = {0, 1, 2};
  matrix.row_size = {8, 8};
  matrix.col_size = {8, 8, 8};
  matrix.row_scalar_off = {0, 8, 16};
  matrix.col_scalar_off = {0, 8, 16, 24};
  matrix.value_off = {0, 64, 128, 192};
  matrix.values.resize(192, 0.0f);
  matrix.values[0] = 100000000.0f;
  matrix.values[64] = 1.0f;
  matrix.values[128] = -100000000.0f;
  matrix.validate();

  for (int rhs : {8, 16, 32, 64}) {
    std::vector<float> input(matrix.scalar_cols() * rhs);
    for (int column = 0; column < rhs; ++column) {
      for (int row = 0; row < matrix.scalar_cols(); ++row)
        input[row + column * matrix.scalar_cols()] = float(column + 1);
    }
    const auto result = vbsr::cpu_reference(matrix, input, rhs);
    for (int column = 0; column < rhs; ++column) {
      for (int row = 0; row < matrix.scalar_rows(); ++row) {
        const float expected = row == 0 ? float(column + 1) : 0.0f;
        if (result[row + column * matrix.scalar_rows()] != expected)
          throw std::runtime_error("CPU reference lost cross-block precision or output layout");
      }
    }
  }
}

void run_rhs8_shape_test() {
  vbsr::HostMatrix matrix;
  matrix.block_rows = 512;
  matrix.block_cols = 8;
  matrix.row_ptr = {0};
  matrix.row_scalar_off = matrix.col_scalar_off = {0};
  matrix.value_off = {0};
  for (int size = 8; size <= 64; size += 8) {
    matrix.col_size.push_back(size);
    matrix.col_scalar_off.push_back(matrix.col_scalar_off.back() + size);
  }
  for (int row = 0; row < matrix.block_rows; ++row) {
    const int size = (row % 8 + 1) * 8;
    matrix.row_size.push_back(size);
    matrix.row_scalar_off.push_back(matrix.row_scalar_off.back() + size);
    // Alternate eight populated shapes with eight empty shapes.
    for (int column = 0; column < (row % 16 < 8 ? 8 : 0); ++column) {
      matrix.block_col.push_back(column);
      matrix.value_off.push_back(matrix.value_off.back() +
                                 matrix.row_size[row] * matrix.col_size[column]);
    }
    matrix.row_ptr.push_back(int32_t(matrix.block_col.size()));
  }
  matrix.values = make_random_input(matrix.value_off.back(), 731);
  matrix.validate();
  const auto input = make_random_input(matrix.scalar_cols() * 8, 732);
  const auto reference = vbsr::cpu_reference(matrix, input, 8);
  vbsr::Matrix device(matrix);
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaStream_t stream = nullptr;
  auto check = [](cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  };
  try {
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(&device_input, input.size() * sizeof(float)));
    check(cudaMalloc(&device_output, reference.size() * sizeof(float)));
    check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0xff, reference.size() * sizeof(float)));
    vbsr::Plan plan(device, {8, vbsr::Kernel::RowOwned});
    plan.execute(device_input, device_output, stream);
    check(cudaStreamSynchronize(stream));
    compare_device_result(reference, device_output, "RHS-8 all 64 block shapes");
  } catch (...) {
    cudaStreamDestroy(stream);
    cudaFree(device_input);
    cudaFree(device_output);
    throw;
  }
  cudaStreamDestroy(stream);
  cudaFree(device_input);
  cudaFree(device_output);
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
        run_case(distribution, degree, rhs_width, true, seed);
        run_case(distribution, degree, rhs_width, false, seed);
        ++seed;
      }
    }
  }
  for (auto distribution : distributions) {
    for (int rhs : rhs_widths) {
      run_case(distribution, 16, rhs, false, seed++, true);
    }
  }
}

} // namespace

int main() {
  try {
    run_negative_tests();
    run_reference_cancellation_test();
    run_rhs8_shape_test();
    run_parameter_matrix();
    std::cout << "PASS: 128 parameter cases and 16 empty-row cases, explicit CSR algorithms, "
                 "uniform BSR, cached and changing grouped pointers, non-default streams\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
