#pragma once

#include "vbsr.hpp"
#include <cstddef>
#include <memory>

namespace vbsr::bench {

class DgFusedPlan {
  public:
    DgFusedPlan(const HostMatrix&, int rhs, bool share_operators = true);
    ~DgFusedPlan();
    DgFusedPlan(const DgFusedPlan&) = delete;
    DgFusedPlan& operator=(const DgFusedPlan&) = delete;

    void execute(const float* input, float* output, cudaStream_t stream = 0);

    int distinct_operators() const;

    int launch_count() const;

    size_t storage_bytes() const;

    size_t operator_bytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
