# Packed-owned mapping: complete recorded-controller equivalence

Replay94114 completed with exit0. All1881 complete ninth-maze decisions match
the original, including the original first terminal failure1870 and terminal
drain. The model state and all source/input bindings passed final checks.
No native execution or adoption occurred, and the original failed navigation
outcome remains unchanged.

Output: go2_packed_owned_maze_controller_replay_v1_attempt_001.
Result:4de3195f7976768c7840189e8c5c77227f5842bd090163c5975f2d8d8063b125.
Launch:90dc683b552a22f90cf2c920e2a1e041d2ae9a6ae1ff0a7c438ea08bbfb97b54.
Compressed decisions:ad92a19a38cdb9c4336030fc739b7af0ef61fe444178d0d2e214122e01606c55.
Model state:4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.
1550 frozen source bindings; elapsed1781.3632316070143 seconds. This elapsed
time includes decoding, validation and receipt handling on a shared machine;
it is not a controlled controller speed comparison.

The earlier component benchmark found approximately3x faster insertion with
exact sampled bounds/witnesses/queries. The complete controller replay now
establishes exact decisions throughout one saved trajectory. Neither result
establishes real-time operation, other trajectories, new sensor behavior or
successful navigation. It does not change the separately frozen settling
controller/native experiment.

Next performance measurement is the fixed first256 observations in
scripts/benchmark_go2_packed_owned_controller_pair_v1.py, admitted only by this
completed result. It alternates two fresh controllers per observation and
measures full controller computation plus the production receipt writer,
retaining complete decisions, deadline misses and shared-machine limitations.
Eleven focused admission/order/timing tests passed in2.14s. The full native
sensing/physics loop is outside that benchmark; no automatic adoption follows.
