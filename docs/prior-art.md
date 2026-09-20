# Prior-art scope

The motivating 2025 H² construction paper describes a non-uniform batched BSR product and says it is split into at most `Csp` non-uniform batched GEMM kernels because a GPU implementation for non-uniform blocks was unavailable:

- Boukaram, Liu, Ghysels, and Li, [Adaptive Sketching Based Construction of H² Matrices on GPUs](https://arxiv.org/abs/2506.16759), IPDPSW 2025.

The library interfaces used by this project expose fixed blocks or groups of dense products:

- [cuSPARSE Blocked-ELL](https://docs.nvidia.com/cuda/cusparse/) accepts one user-provided `ellBlockSize`; it is a fixed-size block representation.
- [cuBLAS grouped batched GEMM](https://docs.nvidia.com/cuda/cublas/) accepts different group shapes, but matrices within one grouped call must have non-overlapping outputs. VarBlockSpMM therefore groups one nonzero slot at a time and accumulates slots sequentially.

Searches for GPU VBSR/non-uniform-BSR SpMM were also refreshed. Results included fixed-block/structured block-sparse work and sparse×sparse research, but no directly matching packed variable-row/variable-column FP32 SpMM API or implementation was identified. This is a scoped prior-art search, not a patentability or exhaustive novelty opinion.

The preceding search note is historical (20 August 2026), not evidence of a current exhaustive search. The paper makes no novelty claim for variable-block storage: Vuduc and Moon studied variable blocks in 2005, OSKI documents independent row/column partitions, and GPU SpMM work already uses ILP and row-level scheduling. The present contribution is an implementation and measured execution policy for the specified contract.

Every supported extent is divisible by eight, so fixed BSR8 can represent all these inputs by subdivision without padding. This is now a measured comparator across the variable-block grid. The native covariance experiment uses geometry-derived near-field leaf blocks within a generated model based on the cited admissibility condition. It is not a benchmark of Boukaram's complete implementation.
