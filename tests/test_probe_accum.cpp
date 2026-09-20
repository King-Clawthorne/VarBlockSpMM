// Two contributions of FLT_MIN * 0.5: every nonzero operand is normal and the
// exact sum is FLT_MIN, but each intermediate product is subnormal.
#include "varblockspmm/vbsr.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}

int main() {
  const float tiny = std::numeric_limits<float>::min();
  // One block row with two 8x8 blocks in different block columns.
  vbsr::HostMatrix m;
  m.block_rows = 1; m.block_cols = 2;
  m.row_ptr = {0, 2}; m.block_col = {0, 1};
  m.row_size = {8}; m.col_size = {8, 8};
  m.row_scalar_off = {0, 8}; m.col_scalar_off = {0, 8, 16};
  m.value_off = {0, 64, 128};
  m.values.assign(128, 0.0f);
  m.values[0] = tiny;    // block 0, entry (0,0)
  m.values[64] = tiny;   // block 1, entry (0,0)
  m.validate();
  for (int rhs : {8, 16, 32, 64}) {
    std::vector<float> panel(size_t(16) * rhs, 0.0f);
    panel[0] = 0.5f;   // row 0  -> contributes tiny*0.5
    panel[8] = 0.5f;   // row 8  -> contributes tiny*0.5
    float *in = nullptr, *out = nullptr;
    check(cudaMalloc(&in, panel.size() * 4));
    check(cudaMalloc(&out, size_t(8) * rhs * 4));
    check(cudaMemcpy(in, panel.data(), panel.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemset(out, 0xff, size_t(8) * rhs * 4));
    vbsr::Matrix dm(m);
    vbsr::Plan plan(dm.device_view(), {rhs});
    plan.execute(in, out);
    check(cudaDeviceSynchronize());
    float got = 0;
    check(cudaMemcpy(&got, out, 4, cudaMemcpyDeviceToHost));
    const double exact = double(tiny) * 0.5 + double(tiny) * 0.5;
    std::cout << "rhs=" << rhs << " kernel=" << got << " exact=" << exact
              << " exact_is_normal=" << (exact >= double(tiny)) << "\n";
    cudaFree(in); cudaFree(out);
  }
  return 0;
}
