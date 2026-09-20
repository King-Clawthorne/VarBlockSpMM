#pragma once
#include "support.hpp"
#include <memory>
namespace vbsr::bench {

class MagmaPlan {
  public:
    MagmaPlan(const HostMatrix&, DeviceMatrix, const float*, float*, int rhs, bool reduce,
              cudaStream_t stream = 0);
    ~MagmaPlan();
    void execute();
    size_t workspace_bytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
