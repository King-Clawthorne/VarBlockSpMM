#pragma once

#include "varblockspmm/vbsr.hpp"
#include <filesystem>
#include <vector>

namespace vbsr::bench {

struct CompactCsrData {
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<float> values;

    void validate(int64_t rows, int64_t columns) const;
    std::vector<float> reference(const std::vector<float>& input, int64_t rows, int64_t columns,
                                 int rhs_width) const;
};

struct ApplicationData {
    HostMatrix packed;
    CompactCsrData compact;
};

ApplicationData load_application(const std::filesystem::path& path);

}
