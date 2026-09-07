# Standing rule: assess hardware before substantial jobs

User instruction, 2026-09-07: assess available hardware before big jobs and plan
for efficient utilization. Apply to subsequent collection, training, evaluation
and large test runs. This does not authorize deleting data or changing a running
experiment's scientific definition.

Before launch:

1. Inspect CPU topology/affinity and actual utilization, available RAM, GPU/VRAM
   and utilization, live competing jobs, and free space on each output volume.
2. Identify independent units suitable for process parallelism. Preserve each
   unit's seeds, sample order, numerical settings and exclusive output ownership.
3. Choose and record concurrency and resource allowances. Benchmark a
   representative workload when throughput or GPU compatibility is unknown;
   do not equate more workers or GPU use with better throughput.
4. Monitor observed throughput, CPU/GPU activity, memory and I/O after launch.
   Distinguish capacity checks from enforced limits and measured workload fit.
5. Preserve running/frozen attempts. Make execution revisions explicit and test
   numerical/scientific equivalence before adopting them. Do not delete old
   data merely because its directory is outside the current repository.

Observed on this machine: Ryzen 9950X3D, 16 physical cores/32 logical CPUs;
91 GiB usable RAM; Radeon AI PRO R9700 with approximately 32 GB VRAM. The original
sequential collector used about one core while the machine was 95–96% CPU idle,
78 GiB RAM was available and GPU utilization was zero. This was a scheduling
limitation, not evidence that the hardware was saturated. Recheck rather than
treating this snapshot as permanent capacity.

Cleanup scope requested: inspect older world-model navigation repositories and
simulation data outside the current V3 iteration; propose exact candidates and
check live-job/source dependencies before asking for deletion approval. Never
open protected benchmark contents or recursively inspect protected directories.
