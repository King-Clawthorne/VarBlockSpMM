# Design

## Storage and lifetime

`HostMatrix` validates packed VBSR metadata. `Matrix` owns its GPU copy; `Plan` stores a non-owning `DeviceMatrix`, so `Matrix` must outlive `Plan`. Structure and values are immutable through the public 1.0 API. Dense blocks, B, and C are column-major.

## Direct kernel

The row-owned family specializes only RHS width `{8,16,32,64}`. Runtime block dimensions remain loop bounds. RHS 8 assigns one output element per thread. RHS 16/32 assign eight RHS columns of one local row to each thread, while RHS 64 assigns sixteen. The wider mapping reuses each A load across more independent accumulators while retaining enough threads per row; still wider candidates lost performance. A two-dimensional grid assigns CTAs disjoint output elements, so the path needs neither atomics nor workspace.

`Auto` currently resolves to the hybrid row-owned dispatch. `SplitRow` is rejected with a clear exception: the measured release grid did not justify partial-output workspace and reduction. `GroupedGemmPlan` remains a fair library baseline and an available explicit alternative.

## Baselines

`ScalarCsrPlan` expands dense blocks once, uploads persistent CSR arrays, and reuses cuSPARSE descriptors/workspace. `GroupedGemmPlan` copies packed block values once and organizes every block-row slot into groups with identical `(r_i,c_j)`. One `cublasSgemmGroupedBatched` call is issued per slot. Outputs do not overlap within a call; later slots use beta=1 to accumulate into the row output.

Grouped device pointer arrays for B and C are refreshed on the supplied stream because execute-time pointers may change. That marshaling is part of the measured hot path.
