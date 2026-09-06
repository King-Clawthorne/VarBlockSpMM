# Focused research follow-up

This follow-up adds a closest-method MAGMA comparison, an application trace,
controlled workload factors, and a device-input API. The old ablations are not
rerun. The collector stops at its requested wall-clock cap and retains partial
evidence without presenting it as a complete campaign.

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

`scripts/prepare_transport.py` discretizes periodic one-dimensional advection
with upwind discontinuous Galerkin fluxes in a Legendre modal basis. Element
orders repeat 7, 15, 31, and 63, giving genuinely variable dense blocks of 8,
16, 32, and 64 coefficients. Every block row couples its own element and its
upwind neighbor. The dense neighbor block comes from the modal boundary trace.
There is no post hoc grouping of scalar nonzeros.

`vbsr_transport` advances independent sine-wave initial conditions with 32
forward-Euler steps. Its two buffers alternate, so successive products are
dependent. Each timing sample is a whole trace, including the device reset of
the initial state. Methods use the same initial conditions, step count, and
assembled FP32 operator. Every output coefficient is checked against repeated
double-precision sparse products. Preparation separately compares the solution
with the analytically translated sine waves in the physical L2 norm.

The timestep is `0.05 * element_width / 64^2`. This short trace tests a specific
transport component, not time-to-solution for a production PDE solver. The
polynomial-order schedule is prescribed, not an adaptive error estimator.
Startup records include constructing a method from prepared host blocks and
one complete trace. They exclude common input-buffer allocation, Python
assembly, and CPU validation. Steady traces exclude method construction.

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

The retained campaign completed all 221 processes in 299.6 seconds, below its
eight-minute cap. The core ratio including MAGMA is 1.138 with 103/128 direct
wins. The large RHS-64 transport trace has ratios of 1.438 against the fastest
library and 1.823 against the faster MAGMA composition. Five of six transport
configurations favor direct in every process. The smallest RHS-8 case favors
BSR8. Full rows and controlled contrasts are in the paper and supplement.

The accompanying `data/relevance/MAGMA-COPYRIGHT` retains the upstream license
for MAGMA sources in the evidence archive. The project's MIT license applies
to the comparison wrapper and original project code.

Activate the locked Python environment, then run `python scripts/prepare_magma.py`.
Configure and build the optional CMake targets `vbsr_relevance`, `vbsr_transport`,
`vbsr_updates`, and `vbsr_magma_tests`, and run CTest. Generate transport inputs
for element counts 256, 1024, and 4096 and widths 8 and 64 with
`python scripts/prepare_transport.py --elements <count> --rhs <width>`.

On the validated Windows configuration, set `CUDAToolkit_ROOT` to the installed
CUDA toolkit and run `python scripts/build_relevance.py` to build these targets,
run both correctness suites, and write the required build receipt.

`python scripts/run_relevance.py --output data/relevance --max-seconds 480`
collects 128 core processes, 72 controlled processes, 18 transport traces, and
three update processes. Each ordinary process records six timing samples per
method. The update experiment records twenty. This deliberately smaller
protocol remains separate from the earlier twenty-sample campaigns.

The manifest preserves exact commands, executable hashes, inputs, source files,
MAGMA revision, raw timing and diagnostic hashes, and the complete design.
`python scripts/run_relevance.py --analyze` validates the retained evidence.
