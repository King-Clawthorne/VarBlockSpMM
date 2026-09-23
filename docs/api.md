# API contract

Owning matrices and public plan constructors complete their setup uploads before
returning, so an immediate first execution may use a nonblocking stream.
Plans can borrow existing GPU allocations through `Plan(DeviceMatrix, PlanOptions, stream)`.
Construction waits for that stream and validates metadata. Matrix values remain on the GPU.
Keep the supplied structure and allocations live and unchanged while the plan is in use.
GPU kernels may update the values in place on the execution stream. For an owning `Matrix`,
`update_values(device_values, count, stream)` replaces values without invalidating existing
plans. Structural changes require a new plan. See [device inputs and the focused research
follow-up](relevance.md) for stream ordering, MAGMA reproduction, and the transport example.

`HostMatrix` requires positive block dimensions, supported block sizes, sorted unique block columns within each row, and consistent packed payloads. Rows may be empty. Scalar and value offsets are 64-bit, while block indices are 32-bit. Scalar CSR expansion and grouped GEMM explicitly reject sizes that exceed their narrower index or leading-dimension limits.

`Matrix` owns fixed GPU structure with updateable values. `Plan` borrows matrix allocations: they must outlive the plan and all queued work, and must not be replaced by move assignment while referenced. A plan constructed from `DeviceMatrix` owns its row classification but borrows the supplied matrix buffers. Dense input and output buffers must be sufficiently sized, non-overlapping, and on the same CUDA device. Calls enqueue work on the supplied stream.

Library plans own mutable handles and metadata. Order calls to one plan on one stream, or externally synchronize across streams and host threads. Grouped GEMM refreshes pointers on the execution stream and safely supports queued address changes.

See [numerical behavior](numerics.md) for FP32 accumulation, flush-to-zero, and overflow details.
