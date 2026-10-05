#include "vbsr.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

int positive_integer(std::string_view text) {
  int value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 1)
    throw std::invalid_argument("expected a positive integer: " + std::string(text));
  return value;
}

template <class T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) { check(cudaMalloc(&data_, count * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }

private:
  T* data_{};
};

class Event {
public:
  Event() { check(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_{};
};

struct Profile {
  std::string_view name;
  std::vector<int32_t> sizes;
  // Empty means columns use the same size family as rows.
  std::vector<int32_t> col_sizes{};
};

const std::array<Profile, 11> profiles{{
    {"small8", {8}},
    {"small16", {16}},
    {"mixed-small", {8, 16}},
    {"medium32", {32}},
    {"large64", {64}},
    {"mixed-all", {8, 16, 24, 32, 40, 48, 56, 64}},
    {"mixed-large", {24, 32, 40, 48, 56, 64}},
    {"small8-wide", {8}, {64}},
    {"small16-wide", {16}, {64}},
    {"tall-narrow8", {64}, {8}},
    {"tall-narrow16", {64}, {16}},
}};

const Profile& find_profile(std::string_view name) {
  for (const auto& profile : profiles)
    if (profile.name == name)
      return profile;
  throw std::invalid_argument("unknown profile: " + std::string(name));
}

// An explicit integer generator makes the inputs reproducible across platforms.
class Random {
public:
  explicit Random(uint32_t seed) : state_(seed) {}
  float signed_value() {
    state_ = state_ * 1664525u + 1013904223u;
    return (float(state_ >> 8) / 8388608.0f) - 1.0f;
  }

private:
  uint32_t state_;
};

vbsr::HostMatrix make_matrix(const Profile& profile, int rows, int degree, bool ones) {
  vbsr::HostMatrix host{.block_rows = rows, .block_cols = rows};
  host.row_ptr.reserve(size_t(rows) + 1);
  host.block_col.reserve(size_t(rows) * degree);
  host.row_size.reserve(rows);
  host.col_size.reserve(rows);
  host.row_scalar_off.push_back(0);
  host.col_scalar_off.push_back(0);
  host.value_off.push_back(0);
  const auto& column_sizes = profile.col_sizes.empty() ? profile.sizes : profile.col_sizes;
  for (int row = 0; row < rows; ++row) {
    const int height = profile.sizes[size_t(row) % profile.sizes.size()];
    const int width = column_sizes[(size_t(row) * 5 + 1) % column_sizes.size()];
    host.row_size.push_back(height);
    host.col_size.push_back(width);
    host.row_scalar_off.push_back(host.row_scalar_off.back() + height);
    host.col_scalar_off.push_back(host.col_scalar_off.back() + width);
  }

  std::vector<int32_t> columns(degree);
  for (int row = 0; row < rows; ++row) {
    host.row_ptr.push_back(int32_t(host.block_col.size()));
    for (int edge = 0; edge < degree; ++edge)
      columns[edge] = (row + edge * 211) % rows;
    std::ranges::sort(columns);
    for (int column : columns) {
      host.block_col.push_back(column);
      host.value_off.push_back(host.value_off.back() +
                               int64_t(host.row_size[row]) * host.col_size[column]);
    }
  }
  host.row_ptr.push_back(int32_t(host.block_col.size()));
  host.values.resize(size_t(host.value_off.back()));
  Random random(0x13579bdfu);
  for (float& value : host.values)
    value = ones ? 1.0f : random.signed_value();
  return host;
}

// Check every output for finiteness, then compare 64 scalar rows, spread over
// block rows and local positions, against double-precision CPU accumulation.
// This keeps the sweep practical even for the largest dense block payloads.
size_t verify(const vbsr::HostMatrix& host, std::span<const float> input,
              std::span<const float> output, int rhs_width, bool ones) {
  for (float value : output)
    if (!std::isfinite(value))
      throw std::runtime_error("nonfinite output value");
  if (ones) {
    for (float value : output)
      if (value != 8.0f)
        throw std::runtime_error("unexpected output value");
    return output.size();
  }

  constexpr int samples = 64;
  for (int sample = 0; sample < samples; ++sample) {
    const int row = (sample * 73) % host.block_rows;
    const int local_row = (sample * 13 + 3) % host.row_size[row];
    for (int rhs = 0; rhs < rhs_width; ++rhs) {
      double expected = 0.0;
      double magnitude = 0.0;
      for (int block = host.row_ptr[row]; block < host.row_ptr[row + 1]; ++block) {
        const int column = host.block_col[block];
        for (int local_column = 0; local_column < host.col_size[column]; ++local_column) {
          const auto value_index =
              host.value_off[block] + local_row + int64_t(local_column) * host.row_size[row];
          const auto input_index =
              host.col_scalar_off[column] + local_column + int64_t(rhs) * host.scalar_cols();
          const double product = double(host.values[value_index]) * input[input_index];
          expected += product;
          magnitude += std::abs(product);
        }
      }
      const auto index = host.row_scalar_off[row] + local_row + int64_t(rhs) * host.scalar_rows();
      const double tolerance = 2.0e-5 * std::max(1.0, magnitude);
      if (std::abs(double(output[index]) - expected) > tolerance)
        throw std::runtime_error("CPU comparison failed at scalar row " +
                                 std::to_string(host.row_scalar_off[row] + local_row) +
                                 ", RHS column " + std::to_string(rhs));
    }
  }
  return size_t(samples) * rhs_width;
}

struct Result {
  float average_ms;
  double gflops;
  size_t checked_values;
  size_t matrix_bytes;
  int launches;
};

Result run_case(const Profile& profile, int rows, int degree, int rhs_width, int iterations,
                bool ones = false) {
  const auto host = make_matrix(profile, rows, degree, ones);
  vbsr::Matrix matrix(host);
  vbsr::Plan plan(matrix, {.rhs_width = rhs_width});
  std::vector<float> input(size_t(matrix.scalar_cols()) * rhs_width);
  Random random(0x2468ace0u);
  for (float& value : input)
    value = ones ? 1.0f : random.signed_value();
  DeviceBuffer<float> device_input(input.size());
  const size_t output_size = size_t(matrix.scalar_rows()) * rhs_width;
  DeviceBuffer<float> device_output(output_size);
  check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  // Poison output so omitted writes are caught by the finiteness check.
  check(cudaMemset(device_output.get(), 0xff, output_size * sizeof(float)));
  const int warmup = ones ? 100 : 20;
  for (int iteration = 0; iteration < warmup; ++iteration)
    plan.execute(device_input.get(), device_output.get());
  check(cudaDeviceSynchronize());

  Event start, stop;
  std::array<float, 3> times{};
  for (float& time : times) {
    check(cudaEventRecord(start.get()));
    for (int iteration = 0; iteration < iterations; ++iteration)
      plan.execute(device_input.get(), device_output.get());
    check(cudaEventRecord(stop.get()));
    check(cudaEventSynchronize(stop.get()));
    check(cudaEventElapsedTime(&time, start.get(), stop.get()));
    time /= iterations;
  }
  std::ranges::sort(times);

  std::vector<float> output(output_size);
  check(cudaMemcpy(output.data(), device_output.get(), output_size * sizeof(float),
                   cudaMemcpyDeviceToHost));
  const size_t checked_values = verify(host, input, output, rhs_width, ones);
  const double operations = 2.0 * double(host.values.size()) * rhs_width;
  return {times[1], operations / (double(times[1]) * 1.0e6), checked_values, matrix.storage_bytes(),
          plan.launch_count()};
}

void print_csv_metadata() {
  int device{}, runtime{};
  cudaDeviceProp properties{};
  check(cudaGetDevice(&device));
  check(cudaGetDeviceProperties(&properties, device));
  check(cudaRuntimeGetVersion(&runtime));
  std::cout << "# device=" << properties.name << " cuda_runtime=" << runtime << '\n'
            << "# timing=median_of_3_cuda_event_batches warmup=20 block_rows=1024\n"
            << "# matrix_seed=0x13579bdf input_seed=0x2468ace0 values=signed_uniform\n"
            << "# correctness=all_outputs_finite_and_64_sampled_scalar_rows_all_rhs\n"
            << "profile,block_rows,degree,rhs,iterations,average_ms,gflops,matrix_bytes,"
               "launches,checked_values\n";
}

void print_csv_case(const Profile& profile, int degree, int rhs, int iterations) {
  constexpr int rows = 1024;
  const Result result = run_case(profile, rows, degree, rhs, iterations);
  std::cout << profile.name << ',' << rows << ',' << degree << ',' << rhs << ',' << iterations
            << ',' << result.average_ms << ',' << result.gflops << ',' << result.matrix_bytes << ','
            << result.launches << ',' << result.checked_values << std::endl;
}
}

int main(int argc, char** argv) try {
  std::cout << std::setprecision(8);
  const std::string_view mode = argc > 1 ? argv[1] : "";
  if (mode == "--help") {
    std::cout << "Usage: vbsr_benchmark [iterations]\n"
                 "       vbsr_benchmark --sweep [iterations]\n"
                 "       vbsr_benchmark --case profile degree rhs [iterations]\n"
                 "Profiles: small8 small16 mixed-small medium32 large64 mixed-all mixed-large\n"
                 "          small8-wide small16-wide tall-narrow8 tall-narrow16\n";
    return 0;
  }
  if (mode == "--sweep") {
    if (argc > 3)
      throw std::invalid_argument("usage: --sweep [iterations]");
    const int iterations = argc == 3 ? positive_integer(argv[2]) : 200;
    print_csv_metadata();
    for (const auto& profile : profiles)
      for (int degree : {1, 4, 8, 16})
        for (int rhs : {8, 16, 32, 64})
          print_csv_case(profile, degree, rhs, iterations);
  } else if (mode == "--case") {
    if (argc != 5 && argc != 6)
      throw std::invalid_argument("usage: --case profile degree rhs [iterations]");
    const auto& profile = find_profile(argv[2]);
    const int degree = positive_integer(argv[3]);
    const int rhs = positive_integer(argv[4]);
    if (degree > 1024)
      throw std::invalid_argument("degree must not exceed 1024");
    if (rhs != 8 && rhs != 16 && rhs != 32 && rhs != 64)
      throw std::invalid_argument("rhs must be 8, 16, 32, or 64");
    const int iterations = argc == 6 ? positive_integer(argv[5]) : 200;
    print_csv_metadata();
    print_csv_case(profile, degree, rhs, iterations);
  } else {
    if (argc > 2)
      throw std::invalid_argument("usage: vbsr_benchmark [iterations]");
    const int iterations = argc == 2 ? positive_integer(argv[1]) : 10000;
    const Result result = run_case(profiles.front(), 4096, 1, 32, iterations, true);
    std::cout << "rows=4096 rhs=32 iterations=" << iterations << " average_ms=" << result.average_ms
              << '\n';
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "benchmark failed: " << error.what() << '\n';
  return 1;
}
