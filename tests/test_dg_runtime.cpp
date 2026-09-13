#include "../bench/dg_fused.cuh"
#include "../bench/dg_specialized.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

int main() {
  try {
    for (int rhs : {8, 16, 32, 64})
      run(rhs);
    std::cout << "PASS: specialized batched and both fused storage modes, queued alternating "
                 "buffers, all widths, poisoned outputs, structural rejection\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
