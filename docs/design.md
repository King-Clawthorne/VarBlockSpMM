# Design

## Storage and lifetime

`HostMatrix` validates packed VBSR metadata, including sorted unique block columns and offsets checked without signed-overflow subtraction. `Matrix` owns its GPU copy. `Plan` borrows the matrix allocation, which must outlive the plan and its queued work and must not be replaced through move assignment. Structure is fixed while a plan is in use. `Matrix::update_values` enqueues a device-to-device value replacement without invalidating plans. `Plan(DeviceMatrix, ...)` accepts caller-owned device buffers and validates metadata after waiting for the supplied stream. Device values stay on the GPU and may be updated in place with correct stream ordering. Dense blocks, B, and C are column-major. B and C are non-overlapping buffers on the same CUDA device.

## Direct kernel

The row-owned family specializes RHS width `{8,16,32,64}`, while block dimensions remain runtime loop bounds. Each CTA owns one complete block-row output panel and writes each element once.

RHS 8 and 16 use 64-thread CTAs. Each active thread owns one scalar output row and all RHS columns, reusing every A load across 8 or 16 accumulators. All threads stage the current B tile cooperatively, synchronize before reading it, and synchronize before replacing it with the next block's tile. The shared leading dimension is 65 floats, with 2,080 bytes allocated at RHS 8 and 4,160 at RHS 16. Empty rows write zero. This policy covers every supported row height without a grid-size threshold.

RHS 32 assigns eight output columns per thread, and RHS 64 assigns sixteen. Their larger CTAs distribute panel columns across several threads per scalar row.

RHS 32 classifies block rows once when `Matrix` is constructed. Rows up to 16 scalars high form a light list; the rest form a reuse-heavy list. Light rows use 128-thread single-buffer CTAs. Heavy rows use 256-thread CTAs with two shared tiles: per-thread asynchronous copies preload the next `B` slice while FMAs consume the current slice. A pipeline wait and CTA barrier at each block boundary make the new tile visible and prevent premature buffer reuse. Four-byte asynchronous copies preserve the 65-float shared leading dimension used to avoid bank conflicts.

The two-list dispatch is enabled for mixed-height matrices at mean degree 8 or greater. At lower degree its second launch costs more than shape separation saves, so those matrices use the original single-buffer kernel; homogeneous matrices need only their one fitting launch. RHS 64 also retains the original 256-thread single-buffer kernel because two full RHS-64 tiles increase shared memory from 16,640 to roughly 32.5 KiB and lose more occupancy than overlap recovers. The narrow-panel policy was selected separately on development seed 4. The evaluation retains the previous policy as a paired control and tests shared versus global input loads with matching 64-thread CTAs.

The row-order list is immutable metadata owned by `Matrix`, or by a plan constructed from external device buffers. Execution still allocates no temporary workspace. Every CTA owns disjoint output elements, so neither path needs atomics.

`Auto` currently resolves to the hybrid row-owned dispatch. `SplitRow` is rejected with a clear exception: the measured release grid did not justify partial-output workspace and reduction. `GroupedGemmPlan` is an explicit library alternative with the slot composition described below.

## Baselines

`ScalarCsrPlan` expands dense blocks once, uploads persistent CSR arrays, and reuses cuSPARSE descriptors/workspace. `GroupedGemmPlan` copies packed block values once and organizes every block-row slot into groups with identical `(r_i,c_j)`. One `cublasSgemmGroupedBatched` call is issued per slot. Outputs do not overlap within a call; later slots use beta=1 to accumulate into the row output.

Cached grouped execution refreshes device pointer arrays only when B or C changes. A CUDA kernel builds the pointers from immutable device offset arrays on the execution stream. This supports queued address changes without temporary host-buffer lifetime assumptions. Refresh launches are included in changing-address timing. Cached execution clears output only when empty rows require it. The default uncached mode retains a clear and refresh on each call.

One library plan must execute in stream order, or be externally synchronized across streams and host threads. Its mutable descriptors, cached pointers, and library handle are not independently concurrent objects.

`ScalarCsrPlan` also supports BSR8 subdivision of all allowed blocks, without changing the stored scalar count, and BSR32 when every partition is divisible by 32. Scalar CSR rejects expanded entry counts or columns above INT32_MAX. Grouped GEMM rejects scalar dimensions above INT32_MAX. The direct representation retains its 64-bit scalar/value offsets.

## Experimental controls and evidence

`launch_ablation` is an experimental entry point and is not selected by `Plan`. RHS-8 controls cross mapping, accumulator count, and CTA count in the earlier mapping family. Additional narrow-panel pairs use matching full-panel accumulation and 64-thread CTAs to isolate input staging in the production mapping. Wider controls keep one CTA per block row, and the single/double-buffer comparison uses identical indirect row order. Full-output references validate each measured variant.

Application startup records include construction, transfer, lazy setup, and one completed product from prepared host formats. Device storage accounting includes explicit arrays and cuSPARSE workspace, excluding common dense panels, host metadata, and opaque library allocations. Repeated-product totals are cost models based on synchronized host medians, not measurements of a solver.

The reference campaign uses `scripts/build_verified.py` to bind source hashes to executable hashes after a clean build and CTest. Campaign analyzers verify build receipts, source archives, input identities, specified designs, commands, method coverage, and timing records. Historical archives remain separately identified and are not treated as current measurements.
