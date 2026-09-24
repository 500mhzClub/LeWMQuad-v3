# Recent qualified reference sources frozen for prospective replay

The recorded height-native anchor loss motivated retaining the immediately
preceding already anchor-qualified visual view alongside the original reference
bank. The candidate tries that view only after original references fail without
conflict. It does not retain bridge/floor/native-derived views, reset history,
relax the bridge budget or change a rigid threshold. See the frozen protocol
docs/go2_recent_qualified_anchor_prefix_v1_2026-09-09.md.

The first raw-image test68760 exposed that the original continuity implementation
calls its parent selector directly, bypassing a subclass override. This was
fixed before freezing: a distinct method copy changes only that dispatch and
keeps the original joint witness wrapper. The syntax-tree test proves the rest
of that method unchanged. Final component/image/witness tests79224 pass16 cases
in3.93s; causal prefix tests pass17 cases in0.16s; replay boundary, mutation and
verification wrapper tests69775 pass9 cases in2.11s. Total42 focused cases.

Read-only19432 authenticated the actual completed height result, launch and raw
audit bindings, then passed the new admission using the original raw-audit
requirements. The full transitive input/source/environment verification still
runs inside the exclusive replay launcher before any output/model execution.

Hardware19432:16 physical/32 logical CPUs, full affinity,4.1% busy;
64,269,475,840 bytes available RAM;705,970,663,424 artifact-volume bytes free,
21,358,182,400 workspace bytes free. GPU card0 idle and card1 at8%; card1 VRAM
1,818,505,216/34,208,743,424 bytes. Supervised parent2534319 and its single native
worker2535576 are the only development consumers. An8GiB CPU replay plus32GiB
native allowance fits, with the40GiB reserve plus2GiB output allowance. The first
supervised case was last observed at complete decision receipt968 and remains
unaudited; no outcome is inferred from its ongoing execution.

Frozen source identities:

| Path | SHA-256 |
| --- | --- |
| lewm/recent_qualified_anchor_development.py | b108f486556003c5bc89b0543903bde4814216f80edbf69cfc616b602056448a |
| lewm/recent_qualified_anchor_controller_development.py | 1d309baa7640cf80d91f78b66a4742bcece293436dc22475765bfd5f52d98446 |
| lewm/recent_qualified_anchor_prefix_development.py | fb55a93a18179b2980d46074fb9f95f3ff563cbe983032812b6403de68206a1d |
| scripts/replay_go2_recent_qualified_anchor_prefix_v1.py | 98cf4ecb409b80ce75856a58c80848d76054e9d36c145a3f17280661a7514965 |
| lewm/tests/test_recent_qualified_anchor_development.py | 7691f6c339d668ffb521ce4b93c4babdec1f5526174f460fa33838cb49df7d13 |
| lewm/tests/test_recent_qualified_anchor_prefix_development.py | 34cad5bad269ada8bb341217455eda99b70b6983814b80ac0f6a5a5dee0ab6c2 |
| lewm/tests/test_recent_qualified_anchor_replay_development.py | d537ad88621f9b6c76341a3df177b5c7f6ee0a14e6cf281364b3275156c610e8 |
| docs/go2_recent_qualified_anchor_prefix_v1_2026-09-09.md | 136e0dc828a2c61f97998332124f86892a050b0f0ad55892512787bfa71cb5d9 |
| docs/go2_partial_floor_height_maze01_anchor_loss_readout_2026-09-09.md | f4649a360ce662cef20e18cf1a505b4010d0ce0bd55d560c9daf67702bd0a2ab |

Next submit one full replay with the fixed native result SHA. Preserve its
exclusive output and every failure; no automatic retry or source change after
launch. This replay stops at the first changed request or either terminal, no
later than original observation645, and cannot establish a physical recovery.
No native retention experiment has yet been prepared or executed. The native
queue remains supervised layouts1,2,3, prepared direct-flow maze3, then the
already preflight-passed anchored-continuation maze2 pilot.

Submitted23895/PID2543628 with the exact recorded native SHA. The process is
live in initial authentication; no replay output or result is assumed yet.
Keep this process and frozen source identities. Supervised19047 remains live,
with first-case decision receipt1272 now complete; its raw audit is pending.

The same replay23895 passed full initial verification and launched:
9c12a4d3683d6bc6f61e837bec7c9afb8441b95a6cf9ffc20086d44bf30c74f7.
Initial scope:5,267,164 digest requests,126,848 unique files,5,140,316 guarded
reuses,26 isolated functions. Initial and final hashing each covered58,830,880,547
bytes; every cached file freshly rehashed, no retained cache or imported-global
mutation. The live replay has reached frame32. No intervention or completed
replay result is claimed yet. Supervised19047 remains live, last observed at
decision receipt1516 before this replay launch.
