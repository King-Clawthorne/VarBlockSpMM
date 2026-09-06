#include <varblockspmm/vbsr.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Buffer {
  float* data = nullptr;
  explicit Buffer(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
int main() {
  try {
    vbsr::GeneratorOptions options;
    options.block_rows = options.block_cols = 8;
    options.degree = 2;
    auto host = vbsr::generate(options);
    constexpr int rhs = 8;
    std::vector<float> input(host.scalar_cols() * rhs, 1.0f);
    auto expected = vbsr::cpu_reference(host, input, rhs);
    Buffer b(input.size()), c(expected.size());
    check(cudaMemcpy(b.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(c.data, 0xff, expected.size() * sizeof(float)));
    vbsr::Matrix matrix(host);
    vbsr::Plan plan(matrix, {rhs});
    plan.execute(b.data, c.data);
    check(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    check(cudaMemcpy(actual.data(), c.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 5e-4f)
        throw std::runtime_error("Consumer output differs from the CPU reference");
    }
    std::cout << "Installed package: full output matches CPU reference\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
