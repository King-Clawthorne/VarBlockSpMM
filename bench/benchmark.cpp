#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "varblockspmm/vbsr.hpp"

namespace {

struct Arguments {
  int rows = 1024;
  int degree = 8;
  int rhs_width = 32;
  int repetitions = 50;
  int warmup_repetitions = 10;
  uint64_t seed = 1;
  bool local_columns = true;
  vbsr::Distribution distribution = vbsr::Distribution::HighVariance;
};

struct TimingResult {
  double gpu_median;
  double gpu_p95;
  double host_median;
  double host_p95;
};

enum class Option {
  Rows,
  Degree,
  RhsWidth,
  Repetitions,
  WarmupRepetitions,
  Seed,
  Locality,
  Distribution,
};

Option parse_option(std::string_view value) {
  static constexpr std::pair<std::string_view, Option> options[] = {
      {"--rows", Option::Rows},
      {"--degree", Option::Degree},
      {"--rhs", Option::RhsWidth},
      {"--reps", Option::Repetitions},
      {"--warmup", Option::WarmupRepetitions},
      {"--seed", Option::Seed},
      {"--locality", Option::Locality},
      {"--distribution", Option::Distribution},
  };

  for (const auto& [name, option] : options) {
    if (value == name) {
      return option;
    }
  }
  throw std::runtime_error("unknown argument: " + std::string(value));
}

std::string next_value(int& argument_index, int argument_count, char** argument_values,
                       const std::string& option) {
  ++argument_index;
  if (argument_index >= argument_count) {
    throw std::runtime_error("missing value for " + option);
  }
  return argument_values[argument_index];
}

vbsr::Distribution parse_distribution(std::string_view value) {
  static constexpr std::pair<std::string_view, vbsr::Distribution> distributions[] = {
      {"uniform", vbsr::Distribution::Uniform},
      {"low", vbsr::Distribution::LowVariance},
      {"high", vbsr::Distribution::HighVariance},
      {"bimodal", vbsr::Distribution::Bimodal},
  };

  for (const auto& [name, distribution] : distributions) {
    if (value == name) {
      return distribution;
    }
  }
  throw std::runtime_error("unknown distribution: " + std::string(value));
}

const char* distribution_name(vbsr::Distribution distribution) {
  switch (distribution) {
  case vbsr::Distribution::Uniform:
    return "uniform";
  case vbsr::Distribution::LowVariance:
    return "low";
  case vbsr::Distribution::HighVariance:
    return "high";
  case vbsr::Distribution::Bimodal:
    return "bimodal";
  }
  throw std::runtime_error("invalid distribution enum");
}

Arguments parse_arguments(int argument_count, char** argument_values) {
  Arguments arguments;

  for (int index = 1; index < argument_count; ++index) {
    const std::string option = argument_values[index];
    const auto value = [&] { return next_value(index, argument_count, argument_values, option); };

    switch (parse_option(option)) {
    case Option::Rows:
      arguments.rows = std::stoi(value());
      break;
    case Option::Degree:
      arguments.degree = std::stoi(value());
      break;
    case Option::RhsWidth:
      arguments.rhs_width = std::stoi(value());
      break;
    case Option::Repetitions:
      arguments.repetitions = std::stoi(value());
      break;
    case Option::WarmupRepetitions:
      arguments.warmup_repetitions = std::stoi(value());
      break;
    case Option::Seed:
      arguments.seed = std::stoull(value());
      break;
    case Option::Locality:
      arguments.local_columns = value() == "local";
      break;
    case Option::Distribution:
      arguments.distribution = parse_distribution(value());
      break;
    }
  }

  return arguments;
}

double percentile_95(const std::vector<double>& sorted_values) {
  const size_t index = std::min(sorted_values.size() - 1, size_t(sorted_values.size() * 0.95));
  return sorted_values[index];
}

TimingResult time_operation(const std::function<void()>& operation, int warmup_repetitions,
                            int repetitions) {
  for (int index = 0; index < warmup_repetitions; ++index) {
    operation();
  }
  cudaDeviceSynchronize();

  std::vector<double> gpu_times;
  std::vector<double> host_times;
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  for (int index = 0; index < repetitions; ++index) {
    const auto host_begin = std::chrono::steady_clock::now();
    cudaEventRecord(begin);
    operation();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    const auto host_end = std::chrono::steady_clock::now();

    float gpu_milliseconds;
    cudaEventElapsedTime(&gpu_milliseconds, begin, end);
    gpu_times.push_back(gpu_milliseconds);
    host_times.push_back(std::chrono::duration<double, std::milli>(host_end - host_begin).count());
  }

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  std::sort(gpu_times.begin(), gpu_times.end());
  std::sort(host_times.begin(), host_times.end());

  return {gpu_times[gpu_times.size() / 2], percentile_95(gpu_times),
          host_times[host_times.size() / 2], percentile_95(host_times)};
}

vbsr::GeneratorOptions make_generator_options(const Arguments& arguments) {
  vbsr::GeneratorOptions options;
  options.block_rows = arguments.rows;
  options.block_cols = arguments.rows;
  options.degree = arguments.degree;
  options.rhs_width = arguments.rhs_width;
  options.distribution = arguments.distribution;
  options.local_columns = arguments.local_columns;
  options.seed = arguments.seed;
  return options;
}

double count_useful_flops(const vbsr::HostMatrix& matrix, int rhs_width) {
  double flops = 0.0;
  for (int block_row = 0; block_row < matrix.block_rows; ++block_row) {
    for (int block_index = matrix.row_ptr[block_row]; block_index < matrix.row_ptr[block_row + 1];
         ++block_index) {
      const int block_column = matrix.block_col[block_index];
      flops += 2.0 * matrix.row_size[block_row] * matrix.col_size[block_column] * rhs_width;
    }
  }
  return flops;
}

void print_csv_header() {
  std::cout << "method,block_rows,degree,distribution,locality,rhs,seed,"
               "gpu_median_ms,gpu_p95_ms,hot_median_ms,hot_p95_ms,useful_gflops,"
               "launches,workspace_bytes\n";
}

void print_result(const char* method, const Arguments& arguments, const TimingResult& timing,
                  double useful_flops, int launch_count, size_t workspace_bytes) {
  const double useful_gflops = useful_flops / (timing.gpu_median * 1e6);

  std::cout << method << ',' << arguments.rows << ',' << arguments.degree << ','
            << distribution_name(arguments.distribution) << ','
            << (arguments.local_columns ? "local" : "random") << ',' << arguments.rhs_width << ','
            << arguments.seed << ',' << std::fixed << std::setprecision(5) << timing.gpu_median
            << ',' << timing.gpu_p95 << ',' << timing.host_median << ',' << timing.host_p95 << ','
            << useful_gflops << ',' << launch_count << ',' << workspace_bytes << '\n';
}

void benchmark_method(const char* method, const Arguments& arguments, double useful_flops,
                      const std::function<void()>& operation, int launch_count,
                      size_t workspace_bytes) {
  const TimingResult timing =
      time_operation(operation, arguments.warmup_repetitions, arguments.repetitions);
  print_result(method, arguments, timing, useful_flops, launch_count, workspace_bytes);
}

int run_benchmark(const Arguments& arguments) {
  const vbsr::HostMatrix host_matrix = vbsr::generate(make_generator_options(arguments));
  vbsr::Matrix device_matrix(host_matrix);

  std::vector<float> input(host_matrix.scalar_cols() * arguments.rhs_width, 0.01f);
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaMalloc(&device_input, input.size() * sizeof(float));
  cudaMalloc(&device_output, host_matrix.scalar_rows() * arguments.rhs_width * sizeof(float));
  cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

  vbsr::Plan direct_plan(device_matrix, {arguments.rhs_width, vbsr::Kernel::RowOwned});
  vbsr::ScalarCsrPlan scalar_csr_plan(host_matrix, arguments.rhs_width);
  vbsr::GroupedGemmPlan grouped_gemm_plan(host_matrix, arguments.rhs_width);
  const double useful_flops = count_useful_flops(host_matrix, arguments.rhs_width);

  print_csv_header();
  benchmark_method(
      "row_owned_hybrid", arguments, useful_flops,
      [&] { direct_plan.execute(device_input, device_output); }, direct_plan.launch_count(), 0);
  benchmark_method(
      "row_owned_scalar", arguments, useful_flops,
      [&] {
        vbsr::launch_row_owned_scalar(device_matrix.device_view(), device_input, device_output,
                                      arguments.rhs_width, 0);
      },
      1, 0);
  benchmark_method(
      "scalar_csr_cusparse", arguments, useful_flops,
      [&] { scalar_csr_plan.execute(device_input, device_output); }, 1,
      scalar_csr_plan.workspace_bytes());
  benchmark_method(
      "slot_grouped_cublas", arguments, useful_flops,
      [&] { grouped_gemm_plan.execute(device_input, device_output); },
      grouped_gemm_plan.launch_count(), 0);

  cudaFree(device_input);
  cudaFree(device_output);
  return 0;
}

} // namespace

int main(int argument_count, char** argument_values) {
  try {
    return run_benchmark(parse_arguments(argument_count, argument_values));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
