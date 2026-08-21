# Design

## Storage and lifetime

`HostMatrix` validates packed VBSR metadata. `Matrix` owns its GPU copy; `Plan` stores a non-owning `DeviceMatrix`, so `Matrix` must outlive `Plan`. Structure and values are immutable through the public 1.0 API. Dense blocks, B, and C are column-major.

## Direct kernel

The row-owned family specializes only RHS width `{8,16,32,64}`. Runtime block dimensions remain loop bounds. RHS 8 assigns one output element per thread. RHS 16/32 assign eight RHS columns of one local row to each thread, while RHS 64 assigns sixteen. The wider mapping reuses each A load across more independent accumulators while retaining enough threads per row; still wider candidates lost performance.

RHS 32 classifies block rows once when `Matrix` is constructed. Rows up to 16 scalars high form a light list; the rest form a reuse-heavy list. Light rows use 128-thread single-buffer CTAs. Heavy rows use 256-thread CTAs with two shared tiles: per-thread asynchronous copies preload the next `B` slice while FMAs consume the current slice. A pipeline wait and CTA barrier at each block boundary make the new tile visible and prevent premature buffer reuse. Four-byte asynchronous copies preserve the 65-float shared leading dimension used to avoid bank conflicts.

The two-list dispatch is enabled for mixed-height matrices at mean degree 8 or greater. At lower degree its second launch costs more than shape separation saves, so those matrices use the original single-buffer kernel; homogeneous matrices need only their one fitting launch. RHS 64 also retains the original 256-thread single-buffer kernel because two full RHS-64 tiles increase shared memory from 17,664 to roughly 34 KiB and lose more occupancy than overlap recovers. RHS 16 keeps direct global loads because synchronization costs outweigh saved loads at that width.

The row-order list is immutable metadata owned by `Matrix`; execution still allocates no temporary workspace. Every CTA owns disjoint output elements, so neither path needs atomics.

`Auto` currently resolves to the hybrid row-owned dispatch. `SplitRow` is rejected with a clear exception: the measured release grid did not justify partial-output workspace and reduction. `GroupedGemmPlan` remains a fair library baseline and an available explicit alternative.

## Baselines

`ScalarCsrPlan` expands dense blocks once, uploads persistent CSR arrays, and reuses cuSPARSE descriptors/workspace. `GroupedGemmPlan` copies packed block values once and organizes every block-row slot into groups with identical `(r_i,c_j)`. One `cublasSgemmGroupedBatched` call is issued per slot. Outputs do not overlap within a call; later slots use beta=1 to accumulate into the row output.

Grouped device pointer arrays for B and C are refreshed on the supplied stream because execute-time pointers may change. That marshaling is part of the measured hot path.
