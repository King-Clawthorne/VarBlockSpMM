#include "vbsr.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> std::vector<T> sequence(int count, int stride = 1) {
  return std::views::iota(0, count) | std::views::transform([=](int i) { return T(i) * stride; }) |
         std::ranges::to<std::vector>();
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
}

int main(int argc, char** argv) try {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 10000;
  if (iterations < 1)
    throw std::invalid_argument("iteration count must be positive");

  constexpr int block_count = 4096;
  constexpr int rhs_width = 32;
  vbsr::HostMatrix host{
      .block_rows = block_count,
      .block_cols = block_count,
      .row_ptr = sequence<int32_t>(block_count + 1),
      .block_col = sequence<int32_t>(block_count),
      .row_size = std::vector<int32_t>(block_count, 8),
      .col_size = std::vector<int32_t>(block_count, 8),
      .row_scalar_off = sequence<int64_t>(block_count + 1, 8),
      .col_scalar_off = sequence<int64_t>(block_count + 1, 8),
      .value_off = sequence<int64_t>(block_count + 1, 64),
      .values = std::vector<float>(size_t(block_count) * 64, 1.0f),
  };

  vbsr::Matrix matrix(host);
  vbsr::Plan plan(matrix, {.rhs_width = rhs_width});
  std::vector<float> input(size_t(matrix.scalar_cols()) * rhs_width, 1.0f);
  DeviceBuffer<float> device_input(input.size());
  const size_t output_size = size_t(matrix.scalar_rows()) * rhs_width;
  DeviceBuffer<float> device_output(output_size);
  check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  for (int i = 0; i < 100; ++i)
    plan.execute(device_input.get(), device_output.get());
  check(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  check(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    plan.execute(device_input.get(), device_output.get());
  check(cudaEventRecord(stop));
  check(cudaEventSynchronize(stop));
  float elapsed_ms{};
  check(cudaEventElapsedTime(&elapsed_ms, start, stop));
  check(cudaEventDestroy(stop));
  check(cudaEventDestroy(start));

  std::vector<float> output(output_size);
  check(cudaMemcpy(output.data(), device_output.get(), output_size * sizeof(float),
                   cudaMemcpyDeviceToHost));
  for (float value : output)
    if (value != 8.0f)
      throw std::runtime_error("unexpected output value");

  std::cout << "rows=" << block_count << " rhs=" << rhs_width << " iterations=" << iterations
            << " average_ms=" << elapsed_ms / iterations << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "benchmark failed: " << error.what() << '\n';
  return 1;
}
