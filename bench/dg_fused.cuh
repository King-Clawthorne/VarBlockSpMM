#pragma once

#include "varblockspmm/vbsr.hpp"
#include <cstddef>
#include <memory>

namespace vbsr::bench {

/** Fused transport comparator: cached operators and one launch per product.
 *
 * The batched comparator keeps one copy of each distinct operator but spends
 * one batched call per shape group. Direct execution makes the opposite trade:
 * it accumulates a block row's contributions inside one thread array, but
 * reads them from the assembled matrix, which stores a separate copy of the
 * same operator for every element. This plan does both.
 *
 * Its kernels mirror the release kernels for the panel width they serve, with
 * the same thread mapping, shared input staging, register accumulators and
 * single output write. Only the operand source differs: an element reads the
 * deduplicated operator for its order class instead of its own packed values,
 * and the block structure is derived from the element index rather than read
 * from metadata, so nothing per element is stored at all.
 *
 * That specialization is only valid for the transport operator this campaign
 * prepares. Construction verifies the assumed structure against the packed
 * matrix, entry by entry, and throws when it does not hold.
 */
class DgFusedPlan {
public:
  /** `share_operators` selects the matched control. When true the plan keeps
   * one copy of each order class's operators. When false it keeps a private
   * copy per element, so the same kernel, thread mapping, addressing
   * arithmetic and contribution order run against duplicated operand bytes.
   * The pair isolates operator sharing from every other difference.
   */
  DgFusedPlan(const HostMatrix&, int rhs, bool share_operators = true);
  ~DgFusedPlan();
  DgFusedPlan(const DgFusedPlan&) = delete;
  DgFusedPlan& operator=(const DgFusedPlan&) = delete;

  /** Computes C = A B for borrowed column-major device panels. */
  void execute(const float* input, float* output, cudaStream_t stream = 0);

  /** Distinct dense operators retained after deduplication. */
  int distinct_operators() const;
  /** Kernel launches issued by one product. */
  int launch_count() const;
  /** Device bytes held by the plan. */
  size_t storage_bytes() const;
  /** Device bytes holding the deduplicated operator values alone. */
  size_t operator_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vbsr::bench
