#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "varblockspmm/vbsr.hpp"

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

int main() {
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
