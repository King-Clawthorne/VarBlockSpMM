# Prior-art lock — refreshed 20 August 2026

The motivating 2025 H² construction paper describes a non-uniform batched BSR product and says it is split into at most `Csp` non-uniform batched GEMM kernels because a GPU implementation for non-uniform blocks was unavailable:

- Boukaram, Liu, Ghysels, and Li, [Adaptive Sketching Based Construction of H² Matrices on GPUs](https://arxiv.org/abs/2506.16759), IPDPSW 2025.

The current NVIDIA APIs checked for this release do not provide the same packed variable-row/variable-column-block operation:

- [cuSPARSE Blocked-ELL](https://docs.nvidia.com/cuda/cusparse/) accepts one user-provided `ellBlockSize`; it is a fixed-size block representation.
- [cuBLAS grouped batched GEMM](https://docs.nvidia.com/cuda/cublas/) accepts different group shapes, but matrices within one grouped call must have non-overlapping outputs. VarBlockSpMM therefore groups one nonzero slot at a time and accumulates slots sequentially.

Searches for GPU VBSR/non-uniform-BSR SpMM were also refreshed. Results included fixed-block/structured block-sparse work and sparse×sparse research, but no directly matching packed variable-row/variable-column FP32 SpMM API or implementation was identified. This is a scoped prior-art search, not a patentability or exhaustive novelty opinion.
