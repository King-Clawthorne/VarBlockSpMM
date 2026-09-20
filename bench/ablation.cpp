#include "support.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

namespace vbsr {
void launch_ablation(DeviceMatrix, const int32_t*, const float*, float*, int, int, cudaStream_t);
}

int main(int argc, char** argv) {
    try {
        namespace bench = vbsr::bench;
        if (argc != 8 && argc != 9)
            throw std::invalid_argument(
                "usage: ablation rows degree rhs distribution seed order_seed reps [tuning]");
        const bool tuning =
            argc == 9 && (std::string(argv[8]) == "tuning" || std::string(argv[8]) == "tuning2");
        if (argc == 9 && !tuning)
            throw std::invalid_argument("unknown experiment mode");
        vbsr::GeneratorOptions options;
        options.block_rows = options.block_cols = bench::parse_integer<int>(argv[1], "rows");
        options.degree = bench::parse_integer<int>(argv[2], "degree");
        const int rhs = bench::parse_integer<int>(argv[3], "rhs");
        bench::validate_panel_width(rhs);
        options.distribution = bench::parse_distribution(argv[4]);
        options.local_columns = false;
        options.seed = bench::parse_integer<uint64_t>(argv[5], "seed");
        const auto order_seed = bench::parse_integer<unsigned>(argv[6], "order seed");
        const int reps = bench::parse_integer<int>(argv[7], "reps");
        bench::validate_repetitions(reps);
        const auto host = vbsr::generate(options);
        vbsr::Matrix matrix(host);
        vbsr::Plan dispatch(matrix, {rhs, vbsr::Kernel::RowOwned});
        const auto input = bench::make_input(bench::panel_elements(host.scalar_cols(), rhs));
        const auto reference = vbsr::cpu_reference(host, input, rhs);
        bench::DeviceBuffer<float> device_input(input.size()), output(reference.size());
        device_input.upload(input);
        std::vector<int32_t> rows(host.block_rows);
        std::iota(rows.begin(), rows.end(), 0);
        std::stable_partition(rows.begin(), rows.end(),
                              [&](int r) { return host.row_size[r] <= 16; });
        bench::DeviceBuffer<int32_t> row_order(rows.size());
        row_order.upload(rows);
        std::vector<int> methods(rhs == 8 ? 11 : (rhs == 16 ? 8 : 6));
        std::iota(methods.begin(), methods.end(), -1);
        if (rhs == 8 || rhs == 16)
            methods.push_back(200);
        if (tuning) {
            if (rhs != 8 && rhs != 16)
                throw std::invalid_argument("tuning requires RHS 8 or 16");
            methods = {-1, 100, 101, 102, 103, 104};
            if (std::string(argv[8]) == "tuning2")
                methods = {-1, 102, 103, 105, 106, 107};
        }
        std::mt19937 rng(order_seed);
        std::shuffle(methods.begin(), methods.end(), rng);
        bench::print_environment(&host);
        bench::print_timing_header();
        int position = 0;
        for (int method : methods) {
            bench::check_cuda(cudaMemset(output.data(), 0xff, output.size() * sizeof(float)));
            bench::measure(
                method == -1 ? "dispatch" : "v" + std::to_string(method),
                [&](int) {
                    if (method == -1)
                        dispatch.execute(device_input.data(), output.data());
                    else
                        vbsr::launch_ablation(matrix.device_view(), row_order.data(),
                                              device_input.data(), output.data(), rhs, method, 0);
                },
                [&] { bench::verify_output(reference, output.data()); }, reps, position++);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
