#pragma once

#include "vbsr.hpp"
#include <cstddef>
#include <memory>

namespace vbsr::bench {

class DgSpecializedPlan {
  public:
    DgSpecializedPlan(const HostMatrix&, int rhs, const float* input, float* output);
    ~DgSpecializedPlan();
    DgSpecializedPlan(const DgSpecializedPlan&) = delete;
    DgSpecializedPlan& operator=(const DgSpecializedPlan&) = delete;

    void execute(cudaStream_t stream = 0);

    int distinct_operators() const;

    int launch_count() const;

    size_t storage_bytes() const;

    size_t operator_bytes() const;

    size_t assembled_bytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
