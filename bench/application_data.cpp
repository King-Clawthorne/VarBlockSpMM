#include "application_data.hpp"
#include "support.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>

namespace vbsr::bench {
namespace {

template <class T> std::vector<T> read_vector(std::ifstream& file, const char* name) {
  static_assert(std::endian::native == std::endian::little);
  uint64_t count = 0;
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  constexpr uint64_t maximum_elements = 100000000;
  if (!file || count > maximum_elements) {
    throw std::invalid_argument(std::string("invalid vector length: ") + name);
  }
  // Check remaining bytes before allocating from a potentially damaged header.
  const auto start = file.tellg();
  file.seekg(0, std::ios::end);
  const auto remaining = file.tellg() - start;
  file.seekg(start);
  if (!file || remaining < 0 || count * sizeof(T) > uint64_t(remaining)) {
    throw std::invalid_argument(std::string("truncated vector: ") + name);
  }
  std::vector<T> values(static_cast<size_t>(count));
  file.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(T)));
  if (!file) {
    throw std::invalid_argument(std::string("could not read vector: ") + name);
  }
  return values;
}

} // namespace

void CompactCsrData::validate(int64_t rows, int64_t columns) const {
  if (rows <= 0 || columns <= 0 || row_offsets.size() != uint64_t(rows) + 1 ||
      row_offsets.front() != 0 || row_offsets.back() < 0 ||
      size_t(row_offsets.back()) != values.size() || column_indices.size() != values.size()) {
    throw std::invalid_argument("inconsistent compact CSR dimensions");
  }
  for (size_t row = 1; row < row_offsets.size(); ++row) {
    if (row_offsets[row] < row_offsets[row - 1]) {
      throw std::invalid_argument("compact CSR row offsets must be monotonic");
    }
  }
  for (size_t row = 0; row + 1 < row_offsets.size(); ++row) {
    for (int64_t index = int64_t(row_offsets[row]) + 1; index < row_offsets[row + 1]; ++index) {
      if (column_indices[index - 1] >= column_indices[index])
        throw std::invalid_argument("compact CSR columns must be sorted and unique within each row");
    }
  }
  for (size_t index = 0; index < values.size(); ++index) {
    if (column_indices[index] < 0 || column_indices[index] >= columns ||
        !std::isfinite(values[index])) {
      throw std::invalid_argument("invalid compact CSR column or value");
    }
  }
}

std::vector<float> CompactCsrData::reference(const std::vector<float>& input, int64_t rows,
                                             int64_t columns, int rhs_width) const {
  validate(rows, columns);
  if (input.size() != panel_elements(columns, rhs_width)) {
    throw std::invalid_argument("CPU reference input size mismatch");
  }
  std::vector<float> output(panel_elements(rows, rhs_width));
  for (int panel_column = 0; panel_column < rhs_width; ++panel_column) {
    for (int64_t row = 0; row < rows; ++row) {
      double sum = 0;
      for (int index = row_offsets[row]; index < row_offsets[row + 1]; ++index) {
        sum += double(values[index]) * input[column_indices[index] + panel_column * columns];
      }
      output[row + panel_column * rows] = float(sum);
    }
  }
  return output;
}

ApplicationData load_application(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::invalid_argument("cannot open application input: " + path.string());
  }
  ApplicationData data;
  auto& matrix = data.packed;
  matrix.row_size = read_vector<int32_t>(file, "block row sizes");
  matrix.col_size = read_vector<int32_t>(file, "block column sizes");
  matrix.block_rows = int(matrix.row_size.size());
  matrix.block_cols = int(matrix.col_size.size());
  matrix.row_scalar_off = read_vector<int64_t>(file, "scalar row offsets");
  matrix.col_scalar_off = read_vector<int64_t>(file, "scalar column offsets");
  matrix.row_ptr = read_vector<int32_t>(file, "block row pointers");
  matrix.block_col = read_vector<int32_t>(file, "block column indices");
  matrix.value_off = read_vector<int64_t>(file, "packed value offsets");
  matrix.values = read_vector<float>(file, "packed values");
  data.compact.row_offsets = read_vector<int32_t>(file, "CSR row offsets");
  data.compact.column_indices = read_vector<int32_t>(file, "CSR column indices");
  data.compact.values = read_vector<float>(file, "CSR values");
  if (file.peek() != std::char_traits<char>::eof()) {
    throw std::invalid_argument("unexpected trailing application data");
  }
  matrix.validate();
  data.compact.validate(matrix.scalar_rows(), matrix.scalar_cols());
  return data;
}

} // namespace vbsr::bench
