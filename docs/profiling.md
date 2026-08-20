# Profiling commands

Nsight Systems:

```powershell
nsys profile --trace=cuda,nvtx --force-overwrite=true -o data/nsys_high build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 5 --reps 20
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum data/nsys_high.nsys-rep
```

Nsight Compute:

```powershell
ncu --target-processes all --kernel-name-base demangled --kernel-name "regex:.*row_owned.*" --launch-skip 5 --launch-count 1 --metrics "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread" --csv --log-file data/ncu_high.csv build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 5 --reps 1
ncu --target-processes all --kernel-name-base demangled --kernel-name "regex:.*row_owned.*" --launch-skip 5 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --csv --log-file data/ncu_high_detailed.csv build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 5 --reps 1
ncu --target-processes all --kernel-name-base demangled --kernel-name "regex:.*row_owned.*" --launch-skip 5 --launch-count 1 --section WarpStateStats --csv --log-file data/ncu_high_warp.csv build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 5 --reps 1
ncu --target-processes all --kernel-name-base demangled --kernel-name "regex:.*row_owned_ilp.*" --launch-skip 3 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats --csv --log-file data/ncu_high_ilp4.csv build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 3 --reps 1
ncu --target-processes all --kernel-name-base demangled --kernel-name "regex:.*row_owned_staged.*" --launch-skip 3 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats --csv --log-file data/ncu_high_staged.csv build\Release\vbsr_benchmark.exe --rows 4096 --degree 16 --rhs 64 --distribution bimodal --locality random --warmup 3 --reps 1
```

The `ncu_high_ilp4.csv` command describes the historical four-accumulator revision. The staged command targets the current RHS-64 kernel.

Nsight Compute replays and instruments kernels, so its reported durations are not benchmark timings. Use the unprofiled benchmark CSV for performance comparisons and the profiler exports for counter ratios and stall diagnosis.
