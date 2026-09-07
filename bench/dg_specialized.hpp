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
 * each, and executes every element that shares an operator as a single batched
 * SGEMM whose A pointer array repeats that operator. The assembled matrix is
 * never stored on the device, so this is the specialized implementation a
 * transport code would write instead of calling a general sparse product.
 *
 * The panels are bound at construction, so the pointer arrays are uploaded
 * once and never refreshed during execution. Alternating buffers need one
 * plan per direction, as the MAGMA comparator does.
 *
 * Construction throws when the input does not have this structure, so a
 * misapplied comparator fails loudly instead of reporting a misleading time.
 */
class DgSpecializedPlan {
public:
  DgSpecializedPlan(const HostMatrix&, int rhs, const float* input, float* output);
  ~DgSpecializedPlan();
  DgSpecializedPlan(const DgSpecializedPlan&) = delete;
  DgSpecializedPlan& operator=(const DgSpecializedPlan&) = delete;

  /** Computes C = A B into the bound column-major device panels. */
  void execute(cudaStream_t stream = 0);

  /** Distinct dense operators retained after deduplication. */
  int distinct_operators() const;
  /** Batched SGEMM calls issued by one product. */
  int launch_count() const;
  /** Device bytes held by the plan, including operators and pointer arrays. */
  size_t storage_bytes() const;
  /** Device bytes holding the deduplicated operator values alone. */
  size_t operator_bytes() const;
  /** Device bytes the same matrix occupies in packed variable-block form. */
  size_t assembled_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vbsr::bench
