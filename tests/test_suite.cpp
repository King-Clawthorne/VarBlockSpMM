#include <cuda_runtime.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include "varblockspmm/vbsr.hpp"
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "../bench/dg_fused.cuh"
#include "../bench/dg_specialized.hpp"
#include <algorithm>
#ifdef VBSR_ENABLE_MAGMA_TEST
#include "magma.hpp"
#include <magma_v2.h>
#endif

namespace test_case_0 {



namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void compare_device_result(const std::vector<float>& reference, const float* device_result,
                           const char* implementation_name) {
  std::vector<float> result(reference.size());
  check_cuda(cudaMemcpy(result.data(), device_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost));

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
  check_cuda(cudaMemsetAsync(device_output, 0xff, reference.size() * sizeof(float), stream));
  execute();
  check_cuda(cudaStreamSynchronize(stream));
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
  check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)));
  check_cuda(cudaMalloc(&device_output, reference.size() * sizeof(float)));
  check_cuda(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream));

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
  check_cuda(cudaMalloc(&alternate_input, input.size() * sizeof(float)));
  auto alternate_host = input;
  for (float& x : alternate_host)
    x *= 0.5f;
  check_cuda(cudaMemcpy(alternate_input, alternate_host.data(), alternate_host.size() * sizeof(float),
             cudaMemcpyHostToDevice));
  const auto alternate_reference = vbsr::cpu_reference(host_matrix, alternate_host, rhs_width);
  execute_and_compare([&] { cached_plan.execute(alternate_input, device_output, stream); }, stream,
                      alternate_reference, device_output, "cached grouped changed address");
  execute_and_compare([&] { cached_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "cached grouped restored address");
  float* alternate_output = nullptr;
  check_cuda(cudaMalloc(&alternate_output, reference.size() * sizeof(float)));
  check_cuda(cudaMemsetAsync(device_output, 0xff, reference.size() * sizeof(float), stream));
  check_cuda(cudaMemsetAsync(alternate_output, 0xff, reference.size() * sizeof(float), stream));
  // Queue refreshes without host synchronization. Every pointer update must be
  // ordered after previous GEMMs and before the GEMMs consuming the new bases.
  cached_plan.execute(device_input, device_output, stream);
  cached_plan.execute(alternate_input, alternate_output, stream);
  cached_plan.execute(device_input, device_output, stream);
  check_cuda(cudaStreamSynchronize(stream));
  compare_device_result(reference, device_output, "queued grouped original output");
  compare_device_result(alternate_reference, alternate_output, "queued grouped alternate output");
  check_cuda(cudaFree(alternate_output));
  check_cuda(cudaFree(alternate_input));
  vbsr::ScalarCsrPlan bsr8_plan(host_matrix, rhs_width, 0, false, true, 8);
  execute_and_compare([&] { bsr8_plan.execute(device_input, device_output, stream); }, stream,
                      reference, device_output, "subdivided BSR8");
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

  check_cuda(cudaStreamDestroy(stream));
  check_cuda(cudaFree(device_input));
  check_cuda(cudaFree(device_output));
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
  malformed_matrix = valid_matrix;
  malformed_matrix.block_col[1] = malformed_matrix.block_col[0];
  expect_invalid_argument([&] { malformed_matrix.validate(); }, "duplicate block column accepted");
  malformed_matrix = valid_matrix;
  malformed_matrix.value_off[1] = INT64_MIN;
  expect_invalid_argument([&] { malformed_matrix.validate(); }, "overflowing value offset accepted");
  malformed_matrix = valid_matrix;
  malformed_matrix.row_scalar_off[1] = INT64_MIN;
  expect_invalid_argument([&] { malformed_matrix.validate(); }, "overflowing scalar offset accepted");

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

void run_shape_test(int rhs, bool all_empty = false) {
  vbsr::HostMatrix matrix;
  matrix.block_rows = 512;
  matrix.block_cols = 16;
  matrix.row_ptr = {0};
  matrix.row_scalar_off = matrix.col_scalar_off = {0};
  matrix.value_off = {0};
  for (int column = 0; column < 16; ++column) {
    const int size = (column % 8 + 1) * 8;
    matrix.col_size.push_back(size);
    matrix.col_scalar_off.push_back(matrix.col_scalar_off.back() + size);
  }
  for (int row = 0; row < matrix.block_rows; ++row) {
    const int size = (row % 8 + 1) * 8;
    matrix.row_size.push_back(size);
    matrix.row_scalar_off.push_back(matrix.row_scalar_off.back() + size);
    // Alternate eight populated shapes with eight empty shapes.
    for (int column = 0; column < (!all_empty && row % 16 < 8 ? 16 : 0); ++column) {
      matrix.block_col.push_back(column);
      matrix.value_off.push_back(matrix.value_off.back() +
                                 matrix.row_size[row] * matrix.col_size[column]);
    }
    matrix.row_ptr.push_back(int32_t(matrix.block_col.size()));
  }
  matrix.values = make_random_input(matrix.value_off.back(), 731);
  matrix.validate();
  const auto input = make_random_input(matrix.scalar_cols() * rhs, 732);
  const auto reference = vbsr::cpu_reference(matrix, input, rhs);
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
    // A no-op must fail even when a previous method left the right result.
    check(cudaMemcpy(device_output, reference.data(), reference.size() * sizeof(float), cudaMemcpyHostToDevice));
    bool rejected = false;
    try {
      execute_and_compare([] {}, stream, reference, device_output, "no-op regression");
    } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("test harness accepted missing writes");
    vbsr::Plan plan(device, {rhs, vbsr::Kernel::RowOwned});
    execute_and_compare([&] { plan.execute(device_input, device_output, stream); }, stream,
                        reference, device_output, "all shapes direct");
    vbsr::GroupedGemmPlan grouped(matrix, rhs, true);
    execute_and_compare([&] { grouped.execute(device_input, device_output, stream); }, stream,
                        reference, device_output, "all shapes grouped");
    vbsr::ScalarCsrPlan bsr(matrix, rhs, 0, false, true, 8);
    execute_and_compare([&] { bsr.execute(device_input, device_output, stream); }, stream,
                        reference, device_output, "all shapes BSR8");
    for (int algorithm : {0, 1, 2, 3}) {
      vbsr::ScalarCsrPlan csr(matrix, rhs, algorithm, algorithm == 1 || algorithm == 3);
      execute_and_compare([&] { csr.execute(device_input, device_output, stream); }, stream,
                          reference, device_output, "all shapes scalar CSR");
    }
  } catch (...) {
    cudaStreamDestroy(stream);
    cudaFree(device_input);
    cudaFree(device_output);
    throw;
  }
  check_cuda(cudaStreamDestroy(stream));
  check_cuda(cudaFree(device_input));
  check_cuda(cudaFree(device_output));
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

void run_device_api_test(int rhs) {
  // Degree eight forces RHS-32's classification-dependent dispatch.
  auto host = vbsr::generate({8, 8, 8, rhs, vbsr::Distribution::Bimodal, false, 17});
  auto input = make_random_input(host.scalar_cols() * rhs, 13);
  auto reference = vbsr::cpu_reference(host, input, rhs);
  if (std::none_of(host.row_size.begin(), host.row_size.end(), [](int n) { return n <= 16; }) ||
      std::none_of(host.row_size.begin(), host.row_size.end(), [](int n) { return n > 16; }))
    throw std::runtime_error("constructor regression requires both row classes");
  cudaStream_t stream{};
  float *b = nullptr, *c = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check_cuda(cudaMalloc(&b, input.size() * sizeof(float)));
  check_cuda(cudaMalloc(&c, host.scalar_rows() * rhs * sizeof(float)));
  try {
    check_cuda(cudaMemcpyAsync(b, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemsetAsync(c, 0xff, reference.size() * sizeof(float), stream));
    check_cuda(cudaStreamSynchronize(stream));
    // Panels are ready before construction. No allocation, copy, or host
    // synchronization may mask the upload-to-execution boundary below.
    vbsr::Matrix owner(host);
    vbsr::Plan owned(owner, {rhs});
    owned.execute(b, c, stream);
    check_cuda(cudaStreamSynchronize(stream));
    compare_device_result(reference, c, "immediate owning constructor execution");
    check_cuda(cudaMemsetAsync(c, 0xff, reference.size() * sizeof(float), stream));
    check_cuda(cudaStreamSynchronize(stream));
    vbsr::Plan borrowed(owner.device_view(), {rhs}, stream);
    borrowed.execute(b, c, stream);
    check_cuda(cudaStreamSynchronize(stream));
    compare_device_result(reference, c, "immediate borrowed constructor execution");
    auto immediate_baseline = [&](auto make_plan) {
      check_cuda(cudaMemsetAsync(c, 0xff, reference.size() * sizeof(float), stream));
      check_cuda(cudaStreamSynchronize(stream));
      auto plan = make_plan();
      plan.execute(b, c, stream);
      check_cuda(cudaStreamSynchronize(stream));
      compare_device_result(reference, c, "immediate baseline constructor execution");
    };
    for (bool cached : {false, true})
      immediate_baseline([&] { return vbsr::GroupedGemmPlan(host, rhs, cached); });
    for (int algorithm : {1, 2, 3})
      immediate_baseline([&] { return vbsr::ScalarCsrPlan(host, rhs, algorithm, algorithm != 2); });
    immediate_baseline([&] { return vbsr::ScalarCsrPlan(host, rhs, 0, false, true, 8); });
    vbsr::Matrix replacement(host);
    // A queued update must be visible to both existing plans, without reconstruction.
    check_cuda(cudaMemsetAsync(const_cast<float*>(replacement.device_view().values), 0,
                              host.values.size() * sizeof(float), stream));
    owner.update_values(replacement.device_view().values, host.values.size(), stream);
    owned.execute(b, c, stream);
    check_cuda(cudaStreamSynchronize(stream));
    std::fill(reference.begin(), reference.end(), 0.0f);
    compare_device_result(reference, c, "owned value update");
    execute_and_compare([&] { borrowed.execute(b, c, stream); }, stream, reference, c, "borrowed value update");
    bool rejected = false;
    try { owner.update_values(nullptr, host.values.size(), stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("null update accepted");
    rejected = false;
    try { owner.update_values(replacement.device_view().values, host.values.size() - 1, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("wrong update count accepted");
    auto bad = owner.device_view();
    bad.scalar_rows++;
    rejected = false;
    try { vbsr::Plan invalid(bad, {rhs}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("inconsistent device metadata accepted");
  } catch (...) {
    cudaFree(b); cudaFree(c); cudaStreamDestroy(stream); throw;
  }
  check_cuda(cudaFree(b)); check_cuda(cudaFree(c)); check_cuda(cudaStreamDestroy(stream));
}

} // namespace

int run() {
  try {
    run_negative_tests();
    run_reference_cancellation_test();
    for (int rhs : {8, 16, 32, 64}) {
      run_device_api_test(rhs);
      run_shape_test(rhs);
      run_shape_test(rhs, true);
    }
    run_parameter_matrix();
    std::cout << "PASS: 128 parameter cases and 16 empty-row cases, explicit CSR algorithms, "
                 "uniform BSR, cached and changing grouped pointers, non-default streams\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
} // namespace test_case_0

namespace test_case_1 {
// Numerical range contract. The kernels are compiled with --use_fast_math,
// which sets -ftz=true. Flushing applies to every intermediate operation, not
// only to the caller's operands and the final result, so these checks pin down
// intermediate underflow, intermediate overflow, and cancellation as well as
// the single-product boundary. Normal operands alone do not buy agreement with
// a double-accumulated reference: an accumulation of normal operands whose
// exact sum is normal can still return zero when its intermediates are
// subnormal, and that case is checked here rather than promised away.



namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
}

float bits_to_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// One 8 by 8 block, so a single scalar product decides each output entry.
vbsr::HostMatrix single_block(float value) {
  vbsr::HostMatrix matrix;
  matrix.block_rows = 1;
  matrix.block_cols = 1;
  matrix.row_ptr = {0, 1};
  matrix.block_col = {0};
  matrix.row_size = {8};
  matrix.col_size = {8};
  matrix.row_scalar_off = {0, 8};
  matrix.col_scalar_off = {0, 8};
  matrix.value_off = {0, 64};
  matrix.values.assign(64, 0.0f);
  matrix.values[0] = value;  // Column-major, so this is entry (0, 0).
  matrix.validate();
  return matrix;
}

// Returns C[0,0] for the given single block and a panel whose first row is
// `input` and whose remaining rows are zero.
float product(const vbsr::HostMatrix& matrix, float input, int rhs) {
  std::vector<float> panel(size_t(matrix.scalar_cols()) * rhs, 0.0f);
  panel[0] = input;
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t output_count = size_t(matrix.scalar_rows()) * rhs;
  check(cudaMalloc(&device_input, panel.size() * sizeof(float)));
  check(cudaMalloc(&device_output, output_count * sizeof(float)));
  float result = std::numeric_limits<float>::quiet_NaN();
  try {
    check(cudaMemcpy(device_input, panel.data(), panel.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0xff, output_count * sizeof(float)));
    vbsr::Matrix device_matrix(matrix);
    vbsr::Plan plan(device_matrix.device_view(), {rhs});
    plan.execute(device_input, device_output);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_input);
    cudaFree(device_output);
    throw;
  }
  cudaFree(device_input);
  cudaFree(device_output);
  return result;
}

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void run_normal_range_cases() {
  // Exponents spanning most of the normal FP32 range, in both directions.
  const float magnitudes[] = {1e-30f, 1e-12f, 1e-3f, 1.0f, 7.5f, 1e3f, 1e12f, 1e30f};
  for (int rhs : {8, 16, 32, 64}) {
    for (float value : magnitudes) {
      for (float input : {1.0f, 0.5f, -3.25f}) {
        const float actual = product(single_block(value), input, rhs);
        const float expected = float(double(value) * double(input));
        if (std::abs(expected) < std::numeric_limits<float>::min())
          continue;  // Covered by the subnormal cases below.
        const double error = std::abs(double(actual) - double(expected)) / std::abs(expected);
        require(error <= 1e-6, "normal-range product diverged from the double reference at " +
                                   std::to_string(value));
      }
    }
  }
}

void run_subnormal_cases() {
  const float smallest_normal = std::numeric_limits<float>::min();
  const float largest_subnormal = bits_to_float(0x007FFFFFu);
  const float smallest_subnormal = bits_to_float(0x00000001u);
  require(largest_subnormal < smallest_normal && smallest_subnormal > 0.0f,
          "test constants are not the intended subnormal values");

  for (int rhs : {8, 64}) {
    // A subnormal stored value flushes to zero, so the product is exactly zero.
    for (float value : {largest_subnormal, smallest_subnormal}) {
      const float actual = product(single_block(value), 1.0f, rhs);
      require(actual == 0.0f, "subnormal matrix value did not flush to zero");
    }
    // A subnormal panel entry flushes the same way.
    const float from_panel = product(single_block(1.0f), smallest_subnormal, rhs);
    require(from_panel == 0.0f, "subnormal panel value did not flush to zero");

    // Normal operands whose product underflows into the subnormal range also
    // produce zero rather than a gradual-underflow result.
    const float underflow = product(single_block(1e-30f), 1e-15f, rhs);
    require(underflow == 0.0f, "underflowing product did not flush to zero");

    // The smallest normal result is retained when both operands are normal.
    const float retained = product(single_block(smallest_normal), 1.0f, rhs);
    require(retained == smallest_normal, "smallest normal product was not retained");

    // Operand flushing is not confined to subnormal results. A subnormal
    // operand is flushed before the multiply, so a product whose exact value
    // is normal still comes back as zero. The largest subnormal scaled by
    // 2^126 has an exact value just below one.
    const float scaled = product(single_block(largest_subnormal), std::ldexp(1.0f, 126), rhs);
    const double exact = double(largest_subnormal) * std::ldexp(1.0, 126);
    require(exact >= double(smallest_normal), "the scaling test no longer targets a normal result");
    require(scaled == 0.0f, "a flushed subnormal operand unexpectedly produced a nonzero result");
  }
}

// One block row holding `blocks` blocks of 8 by 8, each in its own block
// column, so a row of the output accumulates one contribution per block.
vbsr::HostMatrix accumulating_row(int blocks) {
  vbsr::HostMatrix matrix;
  matrix.block_rows = 1;
  matrix.block_cols = blocks;
  matrix.row_ptr = {0, blocks};
  matrix.row_size = {8};
  matrix.row_scalar_off = {0, 8};
  matrix.col_scalar_off.push_back(0);
  matrix.value_off.push_back(0);
  for (int block = 0; block < blocks; ++block) {
    matrix.block_col.push_back(block);
    matrix.col_size.push_back(8);
    matrix.col_scalar_off.push_back(8 * (block + 1));
    matrix.value_off.push_back(64 * (block + 1));
  }
  matrix.values.assign(size_t(64) * blocks, 0.0f);
  return matrix;
}

// Sets entry (0,0) of block `index` and the matching panel entry, then returns
// C[0,0]. `values[i] * inputs[i]` is contribution i.
float accumulate(const std::vector<float>& values, const std::vector<float>& inputs, int rhs) {
  const int blocks = int(values.size());
  auto matrix = accumulating_row(blocks);
  for (int block = 0; block < blocks; ++block)
    matrix.values[size_t(64) * block] = values[block];
  matrix.validate();
  std::vector<float> panel(size_t(matrix.scalar_cols()) * rhs, 0.0f);
  for (int block = 0; block < blocks; ++block)
    panel[size_t(8) * block] = inputs[block];

  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t output_count = size_t(matrix.scalar_rows()) * rhs;
  check(cudaMalloc(&device_input, panel.size() * sizeof(float)));
  check(cudaMalloc(&device_output, output_count * sizeof(float)));
  float result = std::numeric_limits<float>::quiet_NaN();
  try {
    check(cudaMemcpy(device_input, panel.data(), panel.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0xff, output_count * sizeof(float)));
    vbsr::Matrix device_matrix(matrix);
    vbsr::Plan plan(device_matrix.device_view(), {rhs});
    plan.execute(device_input, device_output);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_input);
    cudaFree(device_output);
    throw;
  }
  cudaFree(device_input);
  cudaFree(device_output);
  return result;
}

void run_intermediate_underflow_cases() {
  const float smallest_normal = std::numeric_limits<float>::min();
  for (int rhs : {8, 16, 32, 64}) {
    // Every operand is normal and the exact sum is exactly the smallest
    // normal, but each intermediate product is subnormal and flushes, so the
    // accumulation returns zero. Normal operands therefore do not imply
    // agreement with a double-accumulated reference.
    const float halves = accumulate({smallest_normal, smallest_normal}, {0.5f, 0.5f}, rhs);
    require(halves == 0.0f, "intermediate underflow no longer flushes as documented");

    // The same shape with a normal intermediate is retained exactly.
    const float retained = accumulate({smallest_normal, smallest_normal}, {1.0f, 1.0f}, rhs);
    require(retained == 2.0f * smallest_normal, "normal intermediates were not accumulated");

    // An FMA's internal product is not separately rounded or flushed: it is
    // added to the accumulator before a single rounding. Here the first
    // contribution leaves FLT_MIN in the accumulator and the second has a
    // subnormal product, but the instruction's result is normal and survives.
    // So flushing is a property of each instruction's inputs and result, not
    // of the product term inside it.
    const float fused = accumulate({smallest_normal, smallest_normal}, {1.0f, 0.5f}, rhs);
    require(fused == 1.5f * smallest_normal,
            "a subnormal FMA product was flushed independently of the instruction result");
  }
}

void run_intermediate_overflow_cases() {
  const float huge = std::numeric_limits<float>::max();
  for (int rhs : {8, 64}) {
    // The exact sum is finite and representable, but the running sum passes
    // through an intermediate above FLT_MAX, which becomes an infinity.
    const float cancelling = accumulate({huge, huge, huge}, {1.0f, 1.0f, -1.0f}, rhs);
    require(std::isinf(cancelling) || cancelling == huge,
            "intermediate overflow produced neither an infinity nor the exact sum");
    // Documented behavior: the order is unspecified, so callers cannot rely on
    // a cancelling term rescuing an overflow.
  }
}

void run_cancellation_cases() {
  for (int rhs : {8, 64}) {
    // Plain FP32 accumulation with no compensation. The small term is lost
    // against the large one, so the exact sum of 1.0 does not survive.
    const float cancelled = accumulate({1e8f, 1.0f, -1e8f}, {1.0f, 1.0f, 1.0f}, rhs);
    require(cancelled == 0.0f || cancelled == 1.0f,
            "cancellation result was neither the FP32 nor the exact value");

    // Many normal terms of equal magnitude stay within FP32 rounding of the
    // double-accumulated reference.
    std::vector<float> values(16, 0.1f), inputs(16, 1.0f);
    const float summed = accumulate(values, inputs, rhs);
    double exact = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
      exact += double(values[i]) * double(inputs[i]);
    require(std::abs(double(summed) - exact) / exact <= 1e-6,
            "normal-range accumulation diverged from the double reference");
  }
}

} // namespace

int run() {
  try {
    run_normal_range_cases();
    run_subnormal_cases();
    run_intermediate_underflow_cases();
    run_intermediate_overflow_cases();
    run_cancellation_cases();
    std::cout << "PASS: single products, intermediate underflow and overflow, and "
                 "cancellation all behave as the FP32 accumulation contract states\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
} // namespace test_case_1

namespace test_case_2 {

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  float* p{};
  explicit Buffer(size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(float)));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};

// Four repeats of four element orders, with dense self/upstream operators.
// This small algebraic fixture needs no downloaded or generated input files.
vbsr::HostMatrix fixture(bool nonperiodic = false, bool wrong_neighbor = false,
                         bool missing_block = false) {
  vbsr::HostMatrix a;
  a.block_rows = a.block_cols = 16;
  const int sizes[] = {8, 16, 32, 64};
  for (int r = 0; r < 16; ++r)
    a.row_size.push_back(nonperiodic && r == 4 ? 24 : sizes[r % 4]);
  a.col_size = a.row_size;
  a.row_scalar_off = a.col_scalar_off = {0};
  for (int size : a.row_size) {
    a.row_scalar_off.push_back(a.row_scalar_off.back() + size);
    a.col_scalar_off.push_back(a.col_scalar_off.back() + size);
  }
  a.row_ptr = {0};
  a.value_off = {0};
  for (int r = 0; r < 16; ++r) {
    std::vector<int> columns{r, (r + 15) % 16};
    if (wrong_neighbor && r == 0)
      columns[1] = 11; // Same shape, wrong adjacency.
    if (missing_block && r == 0)
      columns.resize(1);
    std::sort(columns.begin(), columns.end());
    for (int c : columns) {
      a.block_col.push_back(c);
      for (int j = 0; j < a.col_size[c]; ++j)
        for (int i = 0; i < a.row_size[r]; ++i)
          a.values.push_back((c == r && i == j ? 0.6f : 0.f) +
                             (0.02f + 0.003f * float((i + 3 * j + r % 4) % 7)) / a.col_size[c]);
      a.value_off.push_back(int64_t(a.values.size()));
    }
    a.row_ptr.push_back(int(a.block_col.size()));
  }
  a.validate();
  return a;
}

template <class F> void reject(F make) {
  try {
    make();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("unsupported comparator structure was accepted");
}

void run(int rhs) {
  const auto a = fixture();
  const size_t count = size_t(a.scalar_rows()) * rhs;
  Buffer x(count), y(count);
  Stream stream;
  std::vector<float> initial(count);
  for (size_t i = 0; i < count; ++i)
    initial[i] = float(int(i % 101) - 50) / 51.f;
  // Odd and even dependent traces check both final output addresses.
  std::vector<std::vector<float>> expected{initial};
  for (int step = 0; step < 5; ++step)
    expected.push_back(vbsr::cpu_reference(a, expected.back(), rhs));
  if (expected.back() == initial)
    throw std::runtime_error("identity fixture");
  auto trace = [&](const char* name, auto execute) {
    for (int steps : {4, 5}) {
      check(cudaMemcpyAsync(x.p, initial.data(), count * sizeof(float), cudaMemcpyHostToDevice,
                            stream.value));
      for (int step = 0; step < steps; ++step) {
        float* input = step % 2 ? y.p : x.p;
        float* output = step % 2 ? x.p : y.p;
        check(cudaMemsetAsync(output, 0xff, count * sizeof(float), stream.value));
        execute(step, input, output);
        // No host synchronization between dependent products.
      }
      check(cudaStreamSynchronize(stream.value));
      std::vector<float> actual(count);
      check(cudaMemcpy(actual.data(), steps % 2 ? y.p : x.p, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
      double error2 = 0, norm2 = 0;
      for (size_t i = 0; i < count; ++i) {
        const double error = double(actual[i]) - expected[steps][i];
        if (!std::isfinite(actual[i]) || std::abs(error) > 5e-4)
          throw std::runtime_error(std::string(name) + " RHS " + std::to_string(rhs) + " steps " +
                                   std::to_string(steps) + " index " + std::to_string(i) +
                                   " actual " + std::to_string(actual[i]) + " expected " +
                                   std::to_string(expected[steps][i]));
        error2 += error * error;
        norm2 += double(expected[steps][i]) * expected[steps][i];
      }
      if (std::sqrt(error2 / (norm2 + 1e-30)) > 5e-5)
        throw std::runtime_error("specialized trace relative error");
    }
  };
  vbsr::bench::DgSpecializedPlan forward(a, rhs, x.p, y.p), backward(a, rhs, y.p, x.p);
  trace("batched", [&](int step, const float*, float*) {
    (step % 2 ? backward : forward).execute(stream.value);
  });
  for (bool shared : {true, false}) {
    vbsr::bench::DgFusedPlan plan(a, rhs, shared);
    trace(shared ? "fused shared" : "fused copies", [&](int, const float* input, float* output) {
      plan.execute(input, output, stream.value);
    });
    for (const auto& invalid : {fixture(true), fixture(false, true), fixture(false, false, true)})
      reject([&] { vbsr::bench::DgFusedPlan bad(invalid, rhs, shared); });
    auto changed = a;
    changed.values[size_t(changed.value_off[size_t(changed.row_ptr[4])])] += 0.01f;
    reject([&] { vbsr::bench::DgFusedPlan bad(changed, rhs, shared); });
  }
  auto unique = a;
  for (size_t block = 0; block < unique.block_col.size(); ++block)
    unique.values[size_t(unique.value_off[block])] = 2.f + float(block);
  unique.validate();
  reject([&] { vbsr::bench::DgSpecializedPlan bad(unique, rhs, x.p, y.p); });
}
} // namespace

int run() {
  try {
    for (int rhs : {8, 16, 32, 64})
      run(rhs);
    std::cout << "PASS: specialized batched and both fused storage modes, queued alternating "
                 "buffers, all widths, poisoned outputs, structural rejection\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
} // namespace test_case_2

#ifdef VBSR_ENABLE_MAGMA_TEST
namespace test_case_3 {
int run() {
  namespace b=vbsr::bench;
  try {
    if(magma_init()!=MAGMA_SUCCESS) throw std::runtime_error("MAGMA init failed");
    struct Stream {
      cudaStream_t value{};
      Stream() { b::check_cuda(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
      ~Stream() { cudaStreamDestroy(value); }
    } stream;
    for(int rhs:{8,16,32,64}) for(bool empty:{false,true}) {
      vbsr::HostMatrix h;
      h.block_rows=h.block_cols=8; h.row_ptr={0};h.value_off={0};
      h.row_scalar_off=h.col_scalar_off={0};
      for(int r=0;r<8;++r) {
        h.row_size.push_back((r+1)*8);h.col_size.push_back((r+1)*8);
        h.row_scalar_off.push_back(h.row_scalar_off.back()+(r+1)*8);
        h.col_scalar_off.push_back(h.col_scalar_off.back()+(r+1)*8);
      }
      for(int r=0;r<8;++r) {
        for(int c=0;c<8 && !(empty && r==0);++c) {
          h.block_col.push_back(c);h.value_off.push_back(h.value_off.back()+h.row_size[r]*h.col_size[c]);
        }
        h.row_ptr.push_back(int(h.block_col.size()));
      }
      h.values=b::make_input(size_t(h.value_off.back()));h.validate();
      vbsr::Matrix matrix(h);
      auto input=b::make_input(b::panel_elements(h.scalar_cols(),rhs));
      auto reference=vbsr::cpu_reference(h,input,rhs);
      b::DeviceBuffer<float> x(input.size()),y(reference.size());x.upload(input);
      b::check_cuda(cudaStreamSynchronize(nullptr));
      for(bool reduce:{false,true}) {
        b::check_cuda(cudaMemsetAsync(y.data(),0xff,y.size()*sizeof(float),stream.value));
        b::check_cuda(cudaStreamSynchronize(stream.value));
        b::MagmaPlan plan(h,matrix.device_view(),x.data(),y.data(),rhs,reduce,stream.value);
        plan.execute();b::check_cuda(cudaStreamSynchronize(stream.value));b::verify_output(reference,y.data());
      }
    }
    magma_finalize();std::cout<<"PASS: both MAGMA compositions, all 64 shapes, four RHS widths, empty rows\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
} // namespace test_case_3
#endif // VBSR_ENABLE_MAGMA_TEST

int main() {
  for (auto test : {test_case_0::run, test_case_1::run, test_case_2::run}) {
    const int result = test();
    if (result != 0) return result;
  }
#ifdef VBSR_ENABLE_MAGMA_TEST
  return test_case_3::run();
#else
  return 0;
#endif
}
