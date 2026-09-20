#include <algorithm>
#include <random>
#include <stdexcept>

#include "varblockspmm/vbsr.hpp"
namespace vbsr {
HostMatrix generate(const GeneratorOptions& options) {
  if (options.degree < 1 || options.degree > 16 || options.block_rows < 1 ||
      options.block_cols < 1 || options.degree > options.block_cols) {
    throw std::invalid_argument("invalid generator dimensions");
  }

  std::mt19937_64 random_engine(options.seed);
  auto random_block_size = [&]() {
    static constexpr int all_sizes[] = {8, 16, 24, 32, 40, 48, 56, 64};
    static constexpr int low_variance_sizes[] = {24, 32, 40};
    static constexpr int bimodal_sizes[] = {8, 16, 48, 64};

    switch (options.distribution) {
    case Distribution::Uniform:
      return 32;
    case Distribution::LowVariance:
      return low_variance_sizes[random_engine() % 3];
    case Distribution::HighVariance:
      return all_sizes[random_engine() % 8];
    case Distribution::Bimodal:
      return bimodal_sizes[random_engine() % 4];
    }
    throw std::invalid_argument("unknown block-size distribution");
  };

  HostMatrix matrix;
  matrix.block_rows = options.block_rows;
  matrix.block_cols = options.block_cols;
  matrix.row_size.resize(options.block_rows);
  matrix.col_size.resize(options.block_cols);
  for (int& row_size : matrix.row_size) {
    row_size = random_block_size();
  }
  for (int& column_size : matrix.col_size) {
    column_size = random_block_size();
  }

  // Prefix sums map variable-sized block coordinates to scalar coordinates.
  matrix.row_scalar_off = {0};
  for (int row_size : matrix.row_size) {
    matrix.row_scalar_off.push_back(matrix.row_scalar_off.back() + row_size);
  }
  matrix.col_scalar_off = {0};
  for (int column_size : matrix.col_size) {
    matrix.col_scalar_off.push_back(matrix.col_scalar_off.back() + column_size);
  }

  matrix.row_ptr = {0};
  matrix.value_off = {0};
  std::uniform_real_distribution<float> random_value(-1.0f, 1.0f);
  for (int block_row = 0; block_row < options.block_rows; ++block_row) {
    // Sample without replacement so each block row contains exactly `degree`
    // blocks.
    std::vector<int> selected_columns;
    while (int(selected_columns.size()) < options.degree) {
      const int block_column = options.local_columns
                                   ? (block_row + int(random_engine() % (2 * options.degree + 1)) -
                                      options.degree + options.block_cols) %
                                         options.block_cols
                                   : int(random_engine() % options.block_cols);
      if (std::find(selected_columns.begin(), selected_columns.end(), block_column) ==
          selected_columns.end()) {
        selected_columns.push_back(block_column);
      }
    }
    std::sort(selected_columns.begin(), selected_columns.end());

    for (int block_column : selected_columns) {
      matrix.block_col.push_back(block_column);
      const int64_t value_count =
          int64_t(matrix.row_size[block_row]) * matrix.col_size[block_column];
      for (int64_t value_index = 0; value_index < value_count; ++value_index) {
        matrix.values.push_back(random_value(random_engine));
      }
      matrix.value_off.push_back(matrix.value_off.back() + value_count);
    }
    matrix.row_ptr.push_back(int(matrix.block_col.size()));
  }

  matrix.validate();
  return matrix;
}
} // namespace vbsr
