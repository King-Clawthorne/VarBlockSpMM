#include "vbsr.hpp"
#include <algorithm>
#include <array>
#include <random>
#include <span>
#include <stdexcept>

namespace vbsr {
// Build a reproducible variable-block matrix with sorted unique block columns
// per row. Each dense block payload is emitted in column-major order.
HostMatrix generate(const GeneratorOptions& options) {
  constexpr int max_degree = 16;
  if (options.degree < 1 || options.degree > max_degree || options.block_rows < 1 ||
      options.block_cols < 1 || options.degree > options.block_cols) {
    throw std::invalid_argument("invalid generator dimensions");
  }

  std::mt19937_64 random_engine(options.seed);
  // Draw one legal block dimension from the selected shape distribution.
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

  // Sample either globally or from a wrapped neighborhood of the block row.
  auto random_block_column = [&](int block_row) {
    if (!options.local_columns)
      return int(random_engine() % options.block_cols);
    const int window_size = 2 * options.degree + 1;
    const int offset = int(random_engine() % window_size) - options.degree;
    return (block_row + offset + options.block_cols) % options.block_cols;
  };

  HostMatrix matrix;
  matrix.block_rows = options.block_rows;
  matrix.block_cols = options.block_cols;
  matrix.row_size.resize(options.block_rows);
  matrix.col_size.resize(options.block_cols);
  std::generate(matrix.row_size.begin(), matrix.row_size.end(), random_block_size);
  std::generate(matrix.col_size.begin(), matrix.col_size.end(), random_block_size);

  // Prefix sums map block coordinates to scalar row and column coordinates.
  matrix.row_scalar_off = {0};
  for (int row_size : matrix.row_size) {
    matrix.row_scalar_off.push_back(matrix.row_scalar_off.back() + row_size);
  }

  matrix.col_scalar_off = {0};
  for (int column_size : matrix.col_size) {
    matrix.col_scalar_off.push_back(matrix.col_scalar_off.back() + column_size);
  }

  // Append each block row in sorted order and maintain a prefix sum over the
  // column-major dense block payloads.
  matrix.row_ptr = {0};
  matrix.value_off = {0};
  std::uniform_real_distribution<float> random_value(-1.0f, 1.0f);
  for (int block_row = 0; block_row < options.block_rows; ++block_row) {
    // The validated degree cap bounds this scratch space; avoid one heap
    // allocation for every generated block row.
    std::array<int, max_degree> selected_columns;
    int selected_count = 0;
    while (selected_count < options.degree) {
      const int block_column = random_block_column(block_row);
      const auto selected = std::span{selected_columns}.first(selected_count);
      if (!std::ranges::contains(selected, block_column)) {
        selected_columns[selected_count++] = block_column;
      }
    }

    const auto selected = std::span{selected_columns}.first(selected_count);
    std::ranges::sort(selected);

    for (int block_column : selected) {
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
}
