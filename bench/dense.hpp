#pragma once
#include "application_data.hpp"
#include <memory>

namespace vbsr::bench {

class DensePlan {
  public:
    DensePlan(const CompactCsrData&, int64_t rows, int64_t columns, int rhs, const float* input,
              float* output);
    ~DensePlan();
    DensePlan(const DensePlan&) = delete;
    DensePlan& operator=(const DensePlan&) = delete;
    void execute();
    size_t storage_bytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
