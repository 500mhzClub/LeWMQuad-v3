# Paired early-prefix controller and receipt timing

First require the completed1881-decision packed-owned equivalence replay,
including the unchanged original terminal failure1870 and final input/source/
model checks. Supply its actual result SHA-256 to
scripts/benchmark_go2_packed_owned_controller_pair_v1.py. Exclusive destination:
go2_packed_owned_controller_pair_v1_attempt_001. Preflight creates no output.

Measure the first256 consecutive public observations of the ninth maze0 run,
fixed before this benchmark. Initialize two independent fresh models and
controllers: LaterFloorResolutionRoundTripController and
PackedOwnedLaterFloorController. Use identical model weights, seeds, public
mission, geometry, budget and numeric settings. Decode each packet once, give
each arm a private deep copy outside timing, and verify copied/public inputs
are unchanged. Alternate which implementation runs first on each observation,
giving128 first positions to each. Both controllers advance every frame in
chronological order. Each complete decision must exactly equal the recorded
original decision. No simulation, new sensing, mission modification or training.

Time complete controller.observe, then the production gzip receipt writer's
JSON serialization, compression and flush. Store only tick and the complete
decision in each benchmark receipt; do not copy historical wall times. Record
wall and process CPU times, each paired frame, medians and100ms deadline misses.
Require identical compressed decision streams and unchanged model states.
Input decoding, private copying, comparison, rendering, physics and actual
sensor acquisition are excluded. The resulting observation-and-receipt timing
is not a native full-loop latency measurement. This early prefix does not
characterize later map growth or return behavior. Preserve all observations,
including warmup, holds and failures; no favorable frame subset is selected.

One CPU process, two independent resident controller/model states, one numeric
thread, no parallel native scenes. Inspect topology/affinity, CPU/GPU/VRAM,
RAM, competing work and both volumes before launch. Require16GiB available RAM
and1GiB output above40GiB reserve. Enforce the receipt allowance after each
paired frame; RAM and storage reserve are admissions, not OS resource limits.
Record shared-machine competition: the independent settling replay may remain
active. Alternating order controls first-position imbalance, not all cache,
frequency, thermal or scheduling effects. No isolated-machine timing claim.

Revalidate source, complete admitted input and output bindings before/after.
Use measured effect and remaining deadline misses to guide the next performance
change. No automatic native adoption, whole-loop speedup, real-time readiness,
navigation success, independent-layout benefit or hardware claim follows.
