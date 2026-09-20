

#include "varblockspmm/vbsr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    matrix.values[0] = value;
    matrix.validate();
    return matrix;
}

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

    const float magnitudes[] = {1e-30f, 1e-12f, 1e-3f, 1.0f, 7.5f, 1e3f, 1e12f, 1e30f};
    for (int rhs : {8, 16, 32, 64}) {
        for (float value : magnitudes) {
            for (float input : {1.0f, 0.5f, -3.25f}) {
                const float actual = product(single_block(value), input, rhs);
                const float expected = float(double(value) * double(input));
                if (std::abs(expected) < std::numeric_limits<float>::min())
                    continue;
                const double error =
                    std::abs(double(actual) - double(expected)) / std::abs(expected);
                require(error <= 1e-6,
                        "normal-range product diverged from the double reference at " +
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

        for (float value : {largest_subnormal, smallest_subnormal}) {
            const float actual = product(single_block(value), 1.0f, rhs);
            require(actual == 0.0f, "subnormal matrix value did not flush to zero");
        }

        const float from_panel = product(single_block(1.0f), smallest_subnormal, rhs);
        require(from_panel == 0.0f, "subnormal panel value did not flush to zero");

        const float underflow = product(single_block(1e-30f), 1e-15f, rhs);
        require(underflow == 0.0f, "underflowing product did not flush to zero");

        const float retained = product(single_block(smallest_normal), 1.0f, rhs);
        require(retained == smallest_normal, "smallest normal product was not retained");

        const float scaled = product(single_block(largest_subnormal), std::ldexp(1.0f, 126), rhs);
        const double exact = double(largest_subnormal) * std::ldexp(1.0, 126);
        require(exact >= double(smallest_normal),
                "the scaling test no longer targets a normal result");
        require(scaled == 0.0f,
                "a flushed subnormal operand unexpectedly produced a nonzero result");
    }
}

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

        const float halves = accumulate({smallest_normal, smallest_normal}, {0.5f, 0.5f}, rhs);
        require(halves == 0.0f, "intermediate underflow no longer flushes as documented");

        const float retained = accumulate({smallest_normal, smallest_normal}, {1.0f, 1.0f}, rhs);
        require(retained == 2.0f * smallest_normal, "normal intermediates were not accumulated");

        const float fused = accumulate({smallest_normal, smallest_normal}, {1.0f, 0.5f}, rhs);
        require(fused == 1.5f * smallest_normal,
                "a subnormal FMA product was flushed independently of the instruction result");
    }
}

void run_intermediate_overflow_cases() {
    const float huge = std::numeric_limits<float>::max();
    for (int rhs : {8, 64}) {

        const float cancelling = accumulate({huge, huge, huge}, {1.0f, 1.0f, -1.0f}, rhs);
        require(std::isinf(cancelling) || cancelling == huge,
                "intermediate overflow produced neither an infinity nor the exact sum");
    }
}

void run_cancellation_cases() {
    for (int rhs : {8, 64}) {

        const float cancelled = accumulate({1e8f, 1.0f, -1e8f}, {1.0f, 1.0f, 1.0f}, rhs);
        require(cancelled == 0.0f || cancelled == 1.0f,
                "cancellation result was neither the FP32 nor the exact value");

        std::vector<float> values(16, 0.1f), inputs(16, 1.0f);
        const float summed = accumulate(values, inputs, rhs);
        double exact = 0.0;
        for (size_t i = 0; i < values.size(); ++i)
            exact += double(values[i]) * double(inputs[i]);
        require(std::abs(double(summed) - exact) / exact <= 1e-6,
                "normal-range accumulation diverged from the double reference");
    }
}

}

int main() {
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
