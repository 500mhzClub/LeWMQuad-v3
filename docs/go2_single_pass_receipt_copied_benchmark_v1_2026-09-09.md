# Additional map-index timing benefit on the current controller

Compare ReceiptCopiedMeasuredFloorController with SinglePassReceiptCopiedController
on all514 observations of completed original learned maze2. Both use the same
receipt-copy selector, assigned corrected JEPA model and observed controller
behavior. The second replaces only eight empty-state primary/auxiliary/partition
indices with the existing SinglePassMeasuredSampleBoundsIndex. Its packed-owned
insertion and single-pass query implementation is reused unchanged. Preserve
measured-floor observation, registration, mission, residual owners, all forecasts,
geometry receipts and metadata. No tracking or residual-feasibility intervention.

Require the actual learned cohort SHA
a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720,
completed receipt-copy paired benchmark SHA
60849db8aa7a2b09e5e6de977115cb953f6c1b893ed62b0533ccd19b31f1ff95,
and completed single-pass dual-camera prefix SHA
18b79c77516ef738366298ee3e093952a250cfd083938818004d232e71a7ca96.
Authenticate result/artifact/source/environment identities before and after.
Merge predecessor source bindings without permitting conflicts. Prior results
motivate composition but do not establish its current equivalence or speed.

Fresh independent model/controller instances process each identical public
packet. Alternate which arm runs first every observation. Time only observe();
compare each entire returned decision to the saved original immediately after
that call, with no field normalization. Check shared public arrays for mutation
after each call; fail before following any mismatched decision. Require all514
frames, unchanged weights and absent gradients. Report active-only paired
differences and separate order groups, warmup and terminal counts, medians,
means and100ms deadline misses. Keep negative timing outcomes. This comparison
isolates additional index benefit over receipt-copy on this episode; it does
not time acquisition, decoding, comparison, receipt I/O or physics and cannot
establish whole-loop real-time operation or a native navigation improvement.

One sequential CPU replay with two independent models, one OpenCV/PyTorch/BLAS
thread,16GiB available-RAM admission and64MiB output above40GiB reserve. Refresh
CPU topology/affinity/load, RAM, GPU/VRAM, storage and competing jobs. May overlap
the existing single native scene with measured headroom. No profiler hooks or
global patching. Exclusive output:go2_single_pass_receipt_copied_benchmark_v1_attempt_001.
Preserve any failure and all earlier attempts. No native scene, training, source
export, sealed access, deletion or real-robot movement. A later controller use
requires evidence for its own trajectory and timing scope.
