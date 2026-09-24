# Single-pass measured-bound query benchmark on recorded clouds

Benchmark SinglePassMeasuredSampleBoundsIndex against MeasuredSampleBoundsIndex
on the same128 public clouds used by the completed packed-owned insertion
benchmark. Require identical reconstructed cloud SHA-256 values, all inserted
index state and every complete query receipt. All original bounds, witnesses,
counts, latest frames, closed boundaries, sphere arithmetic and UNKNOWN/no-free-
space/no-uncertainty semantics remain unchanged. Eleven focused tests already
passed. No controller is modified or executed and no simulator/model is loaded.

Each frame inserts its cloud into two independent evolving indices. Use three
fixed-rule centers: admitted position plus(.3,0,-.3), position plus(0,.2,0), and
the middle measured point. At each center query two boxes with half-extents
(.04,.04,.04) and(.25,.12,.08), and a22mm sphere. These are component queries,
not a replay of actual controller query arguments. Check and warm both arms,
then time16 repetitions of all nine complete queries per arm. Retain and
compare every timed return outside timing. Repeat the full128-frame evolving
population from empty indices with implementation order reversed. This gives
36,864 timed queries per implementation across two passes. Hash the complete
reference receipts and require the two passes to agree. Insertion, decoding,
validation and receipt comparison are outside query timing.

Exclusive output:go2_single_pass_maze_queries_v1_attempt_001. Run preflight and
inspect CPU topology/affinity/utilization, GPU/VRAM, RAM, competition and storage.
One CPU process/one numerical thread beside the existing single native maze
scene. Require4GiB available RAM and64MiB output above40GiB reserve; these are
capacity admissions, not OS limits. Bind the prior component result/launch,
same public input files, current source and candidate tests. Revalidate before
and after. Preserve failures; no retry, source mutation or native adoption.

Report measured component effect and limits. Warm repeated query timing on a
shared machine does not establish controller/full-loop speed, real-time
operation, trajectory equivalence, navigation success or hardware validity.
The running settling maze pilot keeps its frozen original query implementation.
