# Exact mapping optimization improves measured controller time, misses deadline

Paired benchmark63970 completed with exit0. On the fixed first256 ninth-maze
observations, median controller-plus-receipt wall time fell from716.389810ms
to621.678939ms. Median per-frame paired speed ratio was1.152069. Controller
computation alone fell from699.123943ms to603.449410ms, with paired ratio1.157005.
This is materially smaller than the roughly3x insertion-component ratio.

Both implementations missed the100ms budget on all256 observations, even
excluding sensing, decoding and physics. Median receipt writing was17.299564ms
versus17.278258ms. Maximum combined time was922.268251ms versus842.004290ms.
The result establishes a useful bounded improvement and a remaining deadline
gap, not real-time operation. It does not measure later map growth or a native
full loop. Model, mission, observation, map/contact semantics and decisions
were unchanged.

Each observation was run by two independent fresh controller/model states,
alternating first position, with private copies of the same decoded packet.
Every complete decision matched the recorded original; all copied inputs
remained unchanged. Both production gzip receipt streams were byte-identical.
Final source/input/model checks passed. One CPU process/one numeric thread ran
beside the settling replay; scheduling, thermal and cache effects were not
fully isolated. Median combined process CPU times716.355180ms and621.584081ms
were close to the respective wall times. All measurements, including warmup
and holds, are retained in the result.

Output:go2_packed_owned_controller_pair_v1_attempt_001.
Result:7a91c2a5588002989f607da26db5ad0e6cedd789500a27c6f6f957a2f952a8c5.
Launch:3f83ac410219c5848d567a4931c9b5477ab51a2ca2259270613ce6096c825439.
Each compressed decision stream:
ae76078064eed0096398866bb5c9ee7ffc00cd418e00ee3991dc359ae6d70144.
1553 source bindings; total post-launch time404.3786761770025s.
Admitted full1881-decision equivalence result:
4de3195f7976768c7840189e8c5c77227f5842bd090163c5975f2d8d8063b125.

The packed-owned index is now supported by exact full-trajectory replay and a
bounded paired performance measurement. It remains excluded from the already
defined settling replay/native experiment. A future explicit integration can
use this evidence without claiming it fixes tracking, visibility, verified
arrival/return or the full timing gap. Do not repeat the completed component,
equivalence or paired benchmarks without a changed implementation/question.
