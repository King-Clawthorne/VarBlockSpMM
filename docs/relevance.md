# Focused research follow-up

The replacement follow-up repeats the closest-method MAGMA comparison across
seeds, processes, and queue modes, and adds a numerically meaningful transport
trace. The old ablations remain separate. The collector has no default time cap.
An explicitly requested `--max-seconds` cap retains incomplete evidence without
presenting it as a completed campaign.

## MAGMA comparison

`scripts/prepare_magma.py` fetches upstream MAGMA revision
`c3de65f855dd24cdbe44649a146286558f528087` and generates its precision headers.
Configure the optional comparison with
`-DVARBLOCKSPMM_MAGMA_SOURCE=<absolute checkout path>`.

The optional CMake target compiles upstream `sgemm_vbatched_core.cu` and the
upstream CUDA queue/runtime support. It does not build unrelated LAPACK solvers
or change the GEMM kernels. It invokes the same dispatcher as MAGMA's
`magmablas_sgemm_vbatched_max_nocheck`, with dimensions validated during setup.
MAGMA's own dispatch and shared-memory attribute calls remain in timing.

The slot composition follows the method described in
[Boukaram et al., section IV-A](https://arxiv.org/html/2506.16759v1#S4.SS1):
one block per output row per batch, with sequential accumulation between slots.
The second composition batches all block products into separate temporary
outputs and reduces them by block row. The latter is an additional baseline,
not a claim to reproduce their implementation. Both use FP32, column-major
panels, persistent metadata, and device-resident packed values. Both pass full
CPU comparisons over all 64 supported block shapes, every panel width, and
empty rows. The reduction's allocation and extra kernel are reported explicitly.

Their paper reports leaf sizes of 64 to 256. This project supports dimensions
8 through 64 in steps of eight, so only the 64 boundary overlaps those leaf
sizes. The comparison establishes behavior on the supported product, not the
performance of their whole hierarchical algorithm or its larger leaves.

## Standalone transport application

`scripts/prepare_transport.py` implements periodic one-dimensional advection
using semi-Lagrangian DG projection in a Legendre modal basis. Element orders
repeat 7, 15, 31, and 63. Each step translates the field by a quarter cell along
exact characteristics, then projects onto the receiving basis. Exact polynomial
overlap integrals produce dense self and upstream blocks of sizes 8, 16, 32,
and 64. This replaces the archived forward-Euler trace.

All performance cases reach physical time `T = 1/128`, with `dt = h/4`.
Element counts 256, 1024, and 4096 therefore take 8, 32, and 128 products.
The initial waves have frequencies 1 through the RHS width. The two buffers
alternate, so successive products are dependent. Each timing sample includes
the complete trace, the initial-state reset, and output poisoning.

Every final coefficient is checked against repeated FP64 sparse products of
the same assembled FP32 matrix. Actual GPU output also passes the analytic
physical L2 check with relative tolerance `2e-3`. Before timing any method, the
executable must reject returning the unchanged initial state. Preparation
requires a tenfold rejection margin on both coefficient error limits.

A separate FP64 spatial-refinement study holds physical time `T = 1/16`,
frequencies 1 through 8, the order schedule, and quarter-cell displacement fixed.
Refining from 16 to 32 to 64 elements gives physical relative errors
`1.05856e-6`, `6.25343e-9`, and `3.47850e-11`. Preparation rejects failure to
reduce error by at least fourfold per refinement. This isolates discretization
convergence from FP32 arithmetic. The main paper cites the established
[semi-Lagrangian DG formulation](https://doi.org/10.1051/m2an/2016004).

The order schedule is prescribed. This is a standalone transport component,
not an adaptive solver. Startup includes method construction and one completed
trace, excluding common panel allocation, Python assembly, and reference
validation.

## Specialized transport comparator

The repeating orders and the constant translation give the operator only eight
distinct dense blocks, which `prepare_transport.py` caches while assembling.
`bench/dg_specialized.cpp` is an application-specific comparator that exploits
exactly that. It deduplicates the block payloads to recover the distinct
operators, groups the elements sharing an operator, and runs each group as one
`cublasSgemmBatched` call whose A pointer array repeats that cached operator.
Repeated input pointers are ordinary batched usage: only the output matrices
must be distinct, and they are. An earlier revision passed a zero operator
stride to `cublasSgemmStridedBatched`, which left the supported behavior of
that call unresolved, so the pointer-array composition replaced it and every
recorded number comes from the replacement.

Both panels are bound at construction, so the pointer arrays upload once and
are never refreshed mid-trace. The alternating buffers therefore need one plan
per direction, as the MAGMA comparator does. The assembled matrix never
reaches the device: at 4096 elements the plan holds 33.8 KiB of operators, and
225.8 KiB once the batch pointer arrays are counted, against a 33.8 MiB packed
matrix. The operator bytes are constant in the element count and the pointer
arrays are not. Its constructor refuses inputs without repeated operators, so
it cannot be misused as a general product. Precision stays FP32 with
`CUBLAS_DEFAULT_MATH`.

`scripts/run_dg.py` runs the six transport cases with three fresh processes
each into `data/dg/`, under the same horizon, buffers, and numerical checks as
the main campaign. Each record must carry the structure line naming the
discovered operator count, batched call count, and stored bytes.
`scripts/export_dg.py` validates the archive and writes the paper table.
Each process first reduces its six timed traces by median. Ratios and absolute
times then use geometric averaging across the three processes, with equal
configuration weights for the overall ratio. Reported ranges retain the
smallest and largest individual process ratios.
Direct execution is faster than this batched comparator in all six cases, by
1.367x to 5.318x, winning all 18 process comparisons, while the batched plan
uses 1/76.5 of the assembled matrix storage across its two trace plans.

## Fused transport comparator

`bench/dg_fused.cu` combines operator caching with fused accumulation. Its
kernels mirror the release kernels for the width they serve, with the same
thread mapping, shared input staging, register accumulators, and single output
write. One thread array owns an element and accumulates both contributions
before writing, reading the cached operator for its order class rather than its
own packed copy. The order schedule repeats with period four, so it derives the
structure from the element index and stores no per-element metadata: 33.8 KiB
of operators at any element count. Its constructor verifies the assumed
periodicity, degree, block columns, and payload equality against the packed
matrix, and refuses any other input. It is compiled with the library's device
math flags.

It is faster than direct execution in every case, by 1.014x to 2.246x, taking
17 of 18 process comparisons, including 1.359x on the 4,096-element RHS-64
trace and 2.246x at 4,096 elements with RHS 8.

The `dg_fused_copies` method is a matched control for the cause. It is the
same kernel, thread mapping, addressing arithmetic, and contribution order,
with one change: each element reads a private copy of its operators instead of
the shared class copy, so the operand bytes are duplicated the way the
assembled matrix duplicates them. Dividing its time by the fused time isolates
operator sharing. Sharing is worth 1.395x at 4,096 elements and RHS 64 and
2.190x at RHS 8, where the duplicated variant falls back to roughly direct
execution's time. At 256 and 1,024 elements it is worth between 0.976x and
1.077x, showing more modest process-averaged differences than at 4,096 elements. These
observations do not establish equivalence or exclude an operator-sharing effect.

The natural reading is cache residency of the shared operators, but memory
traffic was not measured, so that is an interpretation consistent with the
control rather than a demonstrated mechanism. What the control does establish
is that at 4,096 elements the advantage is attributable to the operand source
rather than to the metadata, addressing, or ordering changes beside it.

## Controlled factors

The crossed design uses 128 or 512 block rows, degrees 4, 16, or 128, fixed
block sizes 8 or 32, and panel widths 8 or 64. Matrix seed 11 has three fresh
process repetitions. At fixed row count, degrees select nested prefixes of
the same per-row column permutation. Shape changes preserve that adjacency.
Increasing row count also changes the working-set size, so that contrast is
not presented as an isolated occupancy measurement. These controls separate
the chosen workload factors without retrospectively assigning a unique cause
to the original covariance result.

## Device inputs and updates

`Plan(DeviceMatrix, PlanOptions, stream)` borrows existing device allocations.
It waits for the supplied stream, copies metadata for validation, and builds
its row classification. It never copies matrix values to the host. The caller
owns the buffers and must keep the structure unchanged for the plan's lifetime.
Device kernels can update values in place before execution on the same stream.

`Matrix::update_values(device_values, count, stream)` enqueues a device-to-device
value copy. Existing plans remain valid. Source and destination must be disjoint
or identical. Callers order updates and products on one stream or through events.
Structural changes require a new validated plan. `vbsr_updates` separately times
value replacement plus execution and borrowed-structure revalidation plus
execution. The latter includes synchronization and row-order allocation.
These measurements do not extend the older prepared-host startup ratios to
GPU-produced or structurally changing matrices.

## Reproduction

Activate the locked Python environment and select the intended CUDA toolkit.
The measured Windows environment uses CUDA 13.4. From the repository root:

```powershell
$env:CUDAToolkit_ROOT = $env:CUDA_PATH_V13_4
$env:CUDA_PATH = $env:CUDAToolkit_ROOT
python scripts/prepare_magma.py
python scripts/prepare_transport.py --all
python scripts/build_relevance.py
python scripts/run_relevance.py --output data/relevance-rerun
python scripts/run_relevance.py --analyze --output data/relevance-rerun
python scripts/export_relevance.py --input data/relevance-rerun
```

The collector runs 2,304 core processes: 128 configurations, seeds 2, 3, and 5,
three processes per seed, and batches of one and eight. Each method receives
five warmups and twenty timing samples. Queued samples divide elapsed time by
eight. The buffers remain fixed in this core comparison. Jobs and methods are
randomized, and GPU processes run serially. It also collects 72 controlled
processes, 18 transport processes, and three API-update processes, for 2,397
processes in total. Those controls and traces retain six samples per method,
and the update experiment retains twenty.

The main headline averages single-product paired log ratios equally over
configuration, seed, and process. Its wins count 128 configurations after
averaging seeds and processes. The repeatability table additionally reports
both modes, seed ranges, process-label ranges, and wins on 96 configuration
and seed combinations per width. Ranges are descriptive, not confidence
intervals. The NVIDIA-only campaigns retain their separate protocols.

`data/relevance-v2/` is the replacement canonical archive. Its exact input ZIP, stored in 64 MiB parts named `inputs.zip.000` onward,
contains every exact matrix, initial state, coefficient reference, analytic
reference, mass-weight vector, metadata file, and the convergence observations.
The analyzer requires complete coverage, checks each payload hash and extent,
and verifies the command/metadata protocol and numerical diagnostic records.
It also validates source snapshots, build receipts, executable hashes, raw
records, method coverage, trial coverage, and the complete crossed design.
The analyzer verifies all parts and reconstructs the original ZIP automatically when the local unsplit file is absent. The upstream license is retained as `MAGMA-COPYRIGHT`.

The earlier `data/relevance/` archive remains historical evidence. Its near-identity
Euler traces and single-seed MAGMA headline do not generate current results.
The analyzer requires explicit `allow_legacy=True` for historical review and
never silently treats its input hashes as verified archived payloads.

For a complete fresh paper rerun, `scripts/run_final_validation.py` now also
prepares MAGMA and transport, builds and tests the comparison, and collects it
under the requested root's `relevance/` directory. The README command sequence
therefore supplies every campaign required by `build_paper.ps1 -ResultsRoot`.

The completed replacement campaign ran 2,397 processes in 11,065.2 seconds.
The fastest-library ratios are 1.139 for individual products and 1.131 for
eight-product batches, with 105 and 101 configuration wins out of 128.
The 4,096-element/RHS-64 transport trace takes 50.824 ms and gives ratios
1.439 over the fastest library and 1.833 over the faster MAGMA composition.
All six cases reject unchanged input, and every measured method passes the
coefficient and analytic checks.

## Routine comparator regression

`vbsr_dg_tests` is part of CTest whenever `BUILD_TESTING` is enabled, including
builds with `VARBLOCKSPMM_BUILD_BENCHMARKS=OFF`. Its in-memory repeated-operator
fixture checks all four RHS widths, both fused storage modes, the batched plan,
odd and even dependent traces, poisoned outputs, and a nonblocking stream. It
rejects violated periodicity, adjacency, degree, repeated payloads, and the
batched plan's operator-reuse requirement. It needs no campaign inputs.

This test exposed incomplete constructor uploads before nonblocking-stream
execution. Both comparator constructors now wait for their setup uploads.
The retained campaign used the default stream, where upload and execution
were already ordered. Kernel arithmetic and timed execution calls are
unchanged. Startup timings have not been remeasured for the added constructor
synchronization.

The same upload-completion guarantee now applies to the public owning matrix,
borrowed direct plan, scalar CSR/BSR plan, and grouped GEMM plan. The public API
regression preallocates and initializes dense panels before construction, then
immediately executes on a nonblocking stream without an intervening allocation
or synchronization. RHS 32 uses both row classes and degree eight to exercise
classification-dependent dispatch. Archived default-stream measurements remain
tied to their original sources. Constructor startup costs were not remeasured
for these added synchronization calls.
