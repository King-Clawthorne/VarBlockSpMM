#include "support.hpp"
#include <algorithm>
#include <iostream>
#include <random>

// Focused RHS-8 comparison against the scalar implementation used by the
// September 5 release. Both paths receive identical data and execution timing.
int main(int argc, char** argv) {
  try {
    namespace bench = vbsr::bench;
    if (argc != 9)
      throw std::invalid_argument(
          "usage: kernel_audit rows degree distribution locality seed order_seed irregular reps");
    vbsr::GeneratorOptions options;
    options.block_rows = options.block_cols = bench::parse_integer<int>(argv[1], "rows");
    options.degree = bench::parse_integer<int>(argv[2], "degree");
    options.distribution = bench::parse_distribution(argv[3]);
    options.local_columns = bench::parse_locality(argv[4]);
    options.seed = bench::parse_integer<uint64_t>(argv[5], "seed");
    const auto order_seed = bench::parse_integer<unsigned>(argv[6], "order seed");
    const int irregular = bench::parse_integer<int>(argv[7], "irregular");
    const int repetitions = bench::parse_integer<int>(argv[8], "reps");
    bench::validate_repetitions(repetitions);
    if (irregular != 0 && irregular != 1)
      throw std::invalid_argument("irregular must be zero or one");
    auto host = vbsr::generate(options);
    if (irregular) {
      const auto source = host;
      host.row_ptr = {0};
      host.block_col.clear();
      host.value_off = {0};
      host.values.clear();
      for (int row = 0; row < host.block_rows; ++row) {
        const int count = row % 4 == 0 ? 0 : row % 4 == 1 ? 1 : options.degree;
        for (int slot = 0; slot < count; ++slot) {
          const int block = source.row_ptr[row] + slot;
          host.block_col.push_back(source.block_col[block]);
          host.values.insert(host.values.end(), source.values.begin() + source.value_off[block],
                             source.values.begin() + source.value_off[block + 1]);
          host.value_off.push_back(int64_t(host.values.size()));
        }
        host.row_ptr.push_back(int32_t(host.block_col.size()));
      }
      host.validate();
    }
    vbsr::Matrix matrix(host);
    vbsr::Plan plan(matrix, {8, vbsr::Kernel::RowOwned});
    auto input = bench::make_input(bench::panel_elements(host.scalar_cols(), 8));
    bench::DeviceBuffer<float> device_input(input.size());
    bench::DeviceBuffer<float> output(bench::panel_elements(host.scalar_rows(), 8));
    device_input.upload(input);
    bench::print_environment(&host);
    bench::print_timing_header();
    std::vector<int> methods = {0, 1};
    std::mt19937 random_engine(order_seed);
    std::shuffle(methods.begin(), methods.end(), random_engine);
    int position = 0;
    for (int method : methods) {
      bench::measure(method ? "tiled" : "legacy",
                     [&](int) {
                       if (method)
                         plan.execute(device_input.data(), output.data());
                       else
                         vbsr::launch_row_owned_scalar(matrix.device_view(), device_input.data(),
                                                       output.data(), 8, 0);
                     },
                     [&] { bench::verify_probes(host, input, output.data(), 8); }, repetitions,
                     position++);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
