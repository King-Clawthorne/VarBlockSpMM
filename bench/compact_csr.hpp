#pragma once

#include "application_data.hpp"
#include <memory>

namespace vbsr::bench {

enum class CsrAlgorithm { One = 1, Two = 2, Three = 3 };

class CompactCsrPlan {
public:
  CompactCsrPlan(const CompactCsrData& data, int64_t rows, int64_t columns, int rhs_width,
                 float* input, float* output, CsrAlgorithm algorithm);
  ~CompactCsrPlan();
  CompactCsrPlan(const CompactCsrPlan&) = delete;
  CompactCsrPlan& operator=(const CompactCsrPlan&) = delete;
  void execute();
  size_t storage_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vbsr::bench
