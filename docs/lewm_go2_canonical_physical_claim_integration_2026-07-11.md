# Go2 canonical physical-claim caller integration

Date: 2026-07-11

Status: source-only caller requalification complete after adversarial canonical-
byte, external-manifest, wrapper-plumbing, and access-ledger review;
authoritative 24-scene oracle rerun intentionally not started.

## Boundary

This migration implements gates 15-20 of
`lewm_go2_canonical_physical_claim_evaluator_binding_2026-07-11.md` without
opening held-out, G2, sealed, checkpoint, image, label, model-output, or prior
runtime payloads. Only source, synthetic manifests, and small unit-scene
simulations were used.

## Production migration map

| Boundary | Canonical implementation | Previous truth path removed or isolated |
|---|---|---|
| Raw trace construction | `go2_physical_claim_trace.py:119,182` | No caller-built pose hash, task commitment, or loose event dictionary |
| Generalization protocol | `generalization_protocol.py:279` | Distance plus caller-supplied LOS renamed `legacy_distance_los_*` and removed from canonical exports |
| Strict result scorer | `strict_result_scorer.py:166` | Reconstructs raw trace, reruns the whole evaluator, requires exact stored equality, and gives legacy rows no physical credit |
| Oracle positive control | `go2_oracle_positive_control.py:556` | Removed `_true_claim`, `update_claims`, opportunistic per-tick claiming, and retry-on-private-acceptance behavior |
| Physical eligibility | `go2_physical_eligibility.py:849` | Witnesses now include lattice yaw and pass only through the shared evaluator |
| Runtime observer | `go2_physical_claim_observer.py:21` and `benchmark_go2_memory_closed_loop.py:18459` | Controller declarations remain controller state; observer evaluation occurs only after the controller loop |
| Result finalizer | `go2_physical_claim_finalizer.py:25` | Recomputes event fields, reasons, blockers, credit, event/summary/trace hashes, feedback, and top-level physical fields |
| Promotion result reader | `go2_physical_claim_result.py:21` | Requires the external scene manifest, reruns the independent finalizer, and cannot be satisfied by stored summaries or proxy-only `success`, `claimed`, or `claimed_colors` |
| Batch and suite gates | `score_go2_result_batch.py`, `check_go2_generalized_suite.py`, `check_go2_fully_learned_demo.py`, `check_go2_teacher_dataset.py`, `check_go2_clean_demo_candidate.py`, `check_go2_wallaware_closed_loop_gate.py` | All claim/success promotion decisions now require canonical physical status |

Replay and review scripts may display `controller_beacon_claims` as diagnostic
intent. They do not promote it to physical success.

## Gate evidence

15. One synthetic trace is bit-identical through protocol, strict scorer,
    oracle, eligibility, and runtime observer adapters.
16. Strict scorer mutations cover omitted, orphaned, reordered, and changed
    attempts/evaluations plus event, summary, and trace hash changes.
17. A controller attempt with a failed physical factor receives zero credit and
    cannot set `all_targets_claimed`.
18. Runtime evaluation occurs after controller execution; all three forbidden
    evaluator-feedback counters are frozen at zero and nonzero mutations fail.
19. Generalized, fully learned, teacher, clean-demo, and wall-aware gates reject
    proxy-only success and claim-color lists.
20. Canonical JSON and evaluator import-purity remain covered by the pure
    evaluator suite. Type-exact canonical bytes reject `true`/`1` and
    integer/float substitutions throughout traces, summaries, ledgers, and
    aggregates. This document and the source hashes below form the caller
    source map; the independent finalizer mutation suite covers stored output.

Every promotion checker now requires an externally supplied scene manifest and
recomputes physical claims from the raw trace. The three live shell wrappers
derive that manifest from the same corpus, split, family, and scene arguments
used by the benchmark call. Synthetic plumbing tests cover all live checker
call sites, and the authoritative runner source map binds the checkers,
wrappers, and their mutation tests.

## Oracle execution contract

For each task object, the oracle selects one terminal pose independently of
evaluator output. It makes one route attempt. If the terminal is reached it
emits exactly one raw claim attempt; if routing fails, that object has no
replacement attempt and the regression fails. The shared evaluator runs once
after all motion. The oracle uses a conservative planned standoff and terminal
pose tolerance, but those planning values never decide claim acceptance.

## Runtime separation

The runtime controller may stop on its own declared completion or the fixed
budget. Its ordinary claim log and scheduling state are explicitly prefixed
`controller_*`. Full-precision pose commitments are appended outward to the
raw observer stream. Only after the loop ends does the observer produce
`canonical_physical_claim_trace`; top-level `claimed`, `claimed_colors`, and
`success` then derive from physical credit, with stability and clearance still
allowed to make overall success false.

## Remaining execution gate

The binding's authoritative V4 development regression remains deliberately
pending. It requires the reviewed source hashes below to be frozen first, then
the development-only 24-scene run must produce exactly 96 attempts, 96 physical
credits, 24 complete scenes, zero stalls/collisions, and exact eligibility
reconciliation. No held-out or sealed evaluation is authorized by this work.

The authoritative parent preloads all 24 scene manifests and the verified
directional collision policy. Its two fixed six-process CPU stages receive only
in-memory scene/policy objects, cap native numerical threads to one per worker,
and merge results in development-manifest order. The policy file is loaded
exactly once and the same object is reused for oracle routing and physical
eligibility, matching the declared access ledger. Its one byte buffer supplies
both file/content hashes and an independently reconstructed exact binary64
fingerprint over ordered vertices, support angles, support values, and margin;
a same-radius substituted polygon cannot pass preload validation.

## Source hashes

Changing any listed source invalidates this integration evidence.

| Source | SHA-256 |
|---|---|
| evaluator binding | `2de4ff20cff2901ab07b681f042c231f1a1e06f95a77d8c4ae2c20c9e2bb8112` |
| `go2_physical_claim_evaluator.py` | `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df` |
| `go2_physical_claim_canonical.py` | `e63a4ebe5bf615e674f2ebd06c5ba930306a330bc4d89faa0c460c6d2fddf43a` |
| `go2_physical_claim_trace.py` | `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3` |
| `go2_physical_claim_observer.py` | `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30` |
| `go2_physical_claim_result.py` | `8f17791710200b5215e497ca8aa34d08acd30b0dbbd9ad8ded647a9c3ad441b3` |
| `go2_physical_claim_finalizer.py` | `a14061641f031ee65a8dbb8235271d15a1564bd521d00948604f7e7bcd580b85` |
| `generalization_protocol.py` | `66422b8a833ba502f8e4a08c40a7463fc44e2ed7d29e903b74332676c50b4973` |
| `strict_result_scorer.py` | `d4d4fb6ddff297faaf86e0e1ec9590a35deca2f0f2b0e92fe46dfc31fdd187c2` |
| `go2_oracle_positive_control.py` | `589af532bf18e3222559868a9715f7ea2c57973cc4aa108b31d2e5cacc1061d4` |
| `go2_physical_eligibility.py` | `1c8f9c347e547890d5edb3cd83e9994886b5dc57d9c78563e606a2f878f1d3e3` |
| oracle suite finalizer | `f6d665ad20e49af7778b439699fa7f1d6a77918d56ff3ad8e86f744be5ecb306` |
| authoritative oracle runner | `3c35550c96fadd4a87c2240d9d0233917adb17d9587cf4e69e55be8c0573b373` |
| runtime | `e29e2b92ffbfcef7f8c25a80629ece32cb1cc534a96f96bffce416d3db728943` |
| batch scorer | `2d26ac190cccd934500ffa82f303543ec387bfdabaf714cf47cd21ba841f8108` |
| generalized suite checker | `cb8606918a866b7e551c9236ebff863deb5d6365711fb019b32539caebc1a725` |
| fully learned checker | `5950a194d0403199530e5a683ca00a3256951b306a5677a89dbf6ac79c4c284a` |
| teacher checker | `f325dbfec1d5335fc47c8b0e7357a9bf17d8d3f7a0d48348c26cd1bf79f4922f` |
| clean-demo checker | `21f0f28fa8f8757c9289fb39b588c8995f22199847857a9e02933d0934c4acf1` |
| wall-aware checker | `7d9cbd030faa6b6913066bbb07b969106c45fb28cabe4ec2d6c7d24e1bdffb7e` |
| generalized suite wrapper | `62f2755715df0d53eaa3afaac5d268f2ae0ed1fb7f16e3953dcc433e1b117117` |
| teacher collection wrapper | `8515d6ce12dd06460c979d34d420ee9297b2ace86cbcb1d115569f3aee3846e2` |
| fully learned wrapper | `1b9c1e18324215e2b471aca7f442494b79b452873cf63940674393186463b580` |
| replay diagnostic | `e3b2c303f967111ed7714474e59908883e1fd68f90974cb96138acf5ebb56df3` |
| review-video diagnostic | `16f7ecad9ed108bb3cf1da62a683932749bb895a413272599393526f558cbe8e` |

| Test | SHA-256 |
|---|---|
| pure evaluator | `5fd495172da6c74ad59425212c6231cd1814346a750268a746d70aa7645ab321` |
| canonical JSON/type equality | `4f26df57c4dcceab356a61d4cc38b919081a9a74be5ff5f54c98c3a2c0342cba` |
| trace and adapter parity | `935df5d0ec2df119aeeed392a0d6941a131c9bd74d9aff836f58b389e70a6dd9` |
| runtime observer | `e78b873868e7c7de099fef89b70f6b1a83cd77f36822928b84a11aa00e88494f` |
| result reader | `e876e6252a373a43759ac4eefd9d775d49ecd5b85d780e942896178d37743d06` |
| independent finalizer | `3928e4b8e09c28f7de58d83da70b6eadc51a9d7a09f83ced9d32d1970333fad6` |
| protocol | `6a0dafc5947708ce1f0e9c9da0d4d44224adf66cfecc74be0cbbb14d44da4e16` |
| strict scorer | `aebfb8cb8e9837f2bdd977622297ea85165868f6fa298ba3eba9f45da8231f51` |
| oracle | `45ed52eaef9b1ed71a5899f4b0be99a2fd40c4b8cc15968d85913b053113e416` |
| eligibility | `97975eaae03b8f3ad6490d4c83ac7727fe69523a91b6924584ce6beab9205152` |
| oracle suite finalizer | `9f3c5aec2e53eeade88e3dba1e4746f507d8a44e0a3797daabd8bef24f941171` |
| authoritative runner | `c24d7a422aff4325a39868c60544be51611aade21bbe02219104d1f1b44eda39` |
| generalized checker | `391a19e1e5aef812004db49cef094315ac2124c333ea89ef2444111c03645d4f` |
| fully learned checker | `4ef9ea98763f0e9b585cd42de1c422eb7a6f4d24bcf8d3cd084266114d9fc3ec` |
| teacher checker | `3f15444728d8579bf928b0ef69c695be262300152c28d9af325d36195dc7754c` |
| clean-demo checker | `3dc27b8afeeef9b23b7ad11e1a939b950c2a8924880042f730a95d36f58c6f18` |
| wall-aware checker | `760ec5253a0283f2af0dd1ccc13251b573091f6ed965943fad198becfa13b71b` |
| manifest-binding mutations | `e3c60ea3273e04346d8eac815f674fec8b751e601e6e779d7cc9aa05fe05bdb9` |
| wrapper manifest plumbing | `bb82bcec7c666a527a932a5f1205b6c2db2a6adbbfe67a6e48b7dfcb18a2ffcc` |

Full source-only claim verification after requalification: `394 passed`, with
no skips. Root independently reran the four oracle/eligibility files after the
full semantic preload repair: `57 passed in 6.15s`.
