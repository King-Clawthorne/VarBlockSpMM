# Design

## Storage and lifetime

`HostMatrix` validates packed VBSR metadata. `Matrix` owns its GPU copy; `Plan` stores a non-owning `DeviceMatrix`, so `Matrix` must outlive `Plan`. Structure and values are immutable through the public 1.0 API. Dense blocks, B, and C are column-major.

## Direct kernel

The row-owned family specializes only RHS width `{8,16,32,64}`. Runtime block dimensions remain loop bounds. RHS 8 assigns one output element per thread. RHS 16/32 assign eight RHS columns of one local row to each thread, while RHS 64 assigns sixteen. The wider mapping reuses each A load across more independent accumulators while retaining enough threads per row; still wider candidates lost performance.

RHS 32 and 64 use one CTA per block row. For each nonzero block, 256 threads cooperatively load its `column_width * RHS` input slice into shared memory, synchronize, reuse the tile across every local output row, and synchronize before replacing it; the unnecessary replacement barrier after the final block is omitted. The shared leading dimension is padded from 64 to 65 floats to avoid bank conflicts when a warp spans RHS groups. RHS 16 keeps direct global loads because the synchronization cost outweighs the saved loads at that width. Every CTA owns disjoint output elements, so neither path needs atomics or external workspace.

`Auto` currently resolves to the hybrid row-owned dispatch. `SplitRow` is rejected with a clear exception: the measured release grid did not justify partial-output workspace and reduction. `GroupedGemmPlan` remains a fair library baseline and an available explicit alternative.

## Baselines

`ScalarCsrPlan` expands dense blocks once, uploads persistent CSR arrays, and reuses cuSPARSE descriptors/workspace. `GroupedGemmPlan` copies packed block values once and organizes every block-row slot into groups with identical `(r_i,c_j)`. One `cublasSgemmGroupedBatched` call is issued per slot. Outputs do not overlap within a call; later slots use beta=1 to accumulate into the row output.

Grouped device pointer arrays for B and C are refreshed on the supplied stream because execute-time pointers may change. That marshaling is part of the measured hot path.
