#pragma once

#include "varblockspmm/vbsr.hpp"
#include <cstddef>
#include <memory>

namespace vbsr::bench {

/** Application-specific comparator for repeated-operator block structures.
 *
 * A discontinuous Galerkin transport operator with repeating element orders and
 * a constant translation contains only a few distinct dense blocks. This plan
 * discovers those distinct operators from the packed matrix, keeps one copy of
 * each, and executes every element that shares an operator as a single strided
 * batched SGEMM with a zero stride over the operator. The assembled matrix is
 * never stored on the device, so this is the specialized implementation a
 * transport code would write instead of calling a general sparse product.
 *
 * Construction throws when the input does not have this structure, so a
 * misapplied comparator fails loudly instead of reporting a misleading time.
 */
class DgSpecializedPlan {
public:
  DgSpecializedPlan(const HostMatrix&, int rhs);
  ~DgSpecializedPlan();
  DgSpecializedPlan(const DgSpecializedPlan&) = delete;
  DgSpecializedPlan& operator=(const DgSpecializedPlan&) = delete;

  /** Computes C = A B for borrowed column-major device panels. */
  void execute(const float* input, float* output, cudaStream_t stream = 0);

  /** Distinct dense operators retained after deduplication. */
  int distinct_operators() const;
  /** Batched SGEMM calls issued by one product. */
  int launch_count() const;
  /** Device bytes held by the plan, including operators and metadata. */
  size_t storage_bytes() const;
  /** Device bytes the same matrix occupies in packed variable-block form. */
  size_t assembled_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vbsr::bench
