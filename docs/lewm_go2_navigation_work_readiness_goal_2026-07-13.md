# Go2 navigation-work readiness goal

Date: 2026-07-13

Status: **active milestone within the end-to-end generalization goal**

## Objective

Reach the point where a learned camera observation can be produced by reviewed
code, projected into the revisioned physical/configuration memory, selected by
the viewpoint explorer, accumulated by target belief, and consumed by routing
and claiming in development simulation. This milestone starts navigation work;
it is not itself a navigation-success or held-out claim.

## Required gates

1. **V4 observable perception.** The one-shot trainer, metric verifier, and
   stage finalizer pass different-agent source review. Both frozen seeds pass
   the sequential `N=5,16,32,320` development ladder without threshold changes,
   producing a qualified fit checkpoint.
2. **V5 shared JEPA checkpoint.** Reviewed code migrates the qualified V4
   encoder/head exactly into the single-encoder V5 model, jointly trains the
   JEPA branch and mandatory four-equal V4 objectives, runs the preregistered
   matched no-JEPA ablation, performs train-role selection and calibration,
   and publishes an immutable training record. Only that shared checkpoint may
   enter and pass the one-shot G2 perception gate; a standalone V4 fit
   checkpoint is insufficient.
3. **G3 physical equivalence.** The reviewed 24-scene exact audit reports all
   24 development scenes, all 96 beacon endpoints retained, zero unsafe FREE
   cells, exact independent morphology agreement, and all deterministic route
   probes passing. The legacy strict-binary result remains reported separately.
4. **V5 execution evidence.** A reviewed one-shot runner performs and records
   every inference, an independent finalizer reconstructs per-scene and aggregate
   decisions, and publication requires complete source/checkpoint/input/outcome
   bindings. No caller-created production batch or reported metric is accepted.
5. **G5 target evidence.** Posterior behavior remains reversible and stable,
   synthetic observations remain structurally ineligible, and the production
   runner derives candidate domains, localization, visibility, and positive or
   negative evidence from fixed reviewed inputs only.
6. **Development integration entrypoint.** One command binds the qualified
   perception checkpoint and calibration to the qualified learned projection,
   revisioned memory, viewpoint explorer, target belief, route planner, and
   claim evaluator. It produces per-tick raw outcomes and an actual-open ledger.
   Physical executor/reset promotion may remain fail-closed for kinematic
   development runs, but it must not be represented as hardware-qualified.

## Execution policy

- Source review and remediation run in parallel where ownership permits.
- CPU audits use capped native threads and at most six scene workers.
- Neural training and inference use GPU0 only. GPU1 is forbidden.
- Each V4 rung is fail-fast. A failed rung stops scaling and triggers diagnosis;
  thresholds are not changed after observing a result.
- G2, held-out, runtime, and promotion inputs stay closed until their preceding
  gate explicitly licenses them.
- The sealed held-out navigation benchmark is outside this milestone and is not
  used for iteration.

## Current status

- **2026-07-14 critical-path update.** The frozen Camera V9 source received a
  canonical different-agent `PASS` (`157/157` combined checks), and its sole
  exact GPU0 attempt completed all 4,000 updates. The isolated verifier child
  then exited nonzero, but V9 discarded the child return details and bounded
  output before terminal cleanup. Only `reservation.json` and `failed.json`
  remain; checkpoint, result, completion, and all metric artifacts were
  removed. This is an infrastructure failure with no observed numerical
  result, so V9 is neither a science pass nor a science failure and cannot be
  retried. The diagnosis is frozen at `59e9036a...ec69`. A science-identical
  Camera V10 lifecycle amendment is frozen at `1d4e4e31...501f`; it requires
  phase-tagged bounded child errors, durable diagnostics before cleanup, and a
  real unmocked CPU `python -I -B` subprocess test. V10 source-only
  implementation and contract review are active; no Camera V10 exact attempt
  is authorized.
- **2026-07-14 raw-data update.** Builder and Auditor V9 both passed distinct
  source review. The nine-row authorization file was published at file SHA-256
  `7878c807...6792`, and a separate agent independently reproduced that exact
  fingerprint. The sole six-worker CPU Builder V9 execution completed: its
  immutable manifest has file SHA-256 `e102b3c6...4360`, canonical content
  SHA-256 `74ae5799...35a`, status `complete_pending_independent_audit`, 5,172
  pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 scene shards,
  and a complete 883-file inventory. The sole Auditor V9 execution then
  terminally failed before array comparison because it passed
  `SceneManifest.to_dict()` tuple fields to a raw-JSON validator that correctly
  requires decoded lists. Its failure receipt has file SHA-256
  `86363057...722f`, content SHA-256 `aaf342f7...2c72`, status
  `terminal_failed_no_dataset_authority`, and `retry_authorized=false`.
  Therefore the built bytes remain quarantined and cannot feed training. The
  audit-only V10 amendment is frozen at `02100ee0...a81`; it preserves and
  audits the existing immutable build, correcting only the raw-mapping versus
  typed-manifest representation boundary. Raw Auditor V10 source-only work is
  active; no V10 audit or dataset-use authority exists yet.
- **Latest execution update.** Camera V7 completed its exact 400-update,
  2,000-exposure training phase on GPU0, then failed closed before independent
  metric publication. The frozen V1 verifier called
  `torch.set_num_interop_threads(1)` in the already-used training process;
  PyTorch forbids changing that process-global setting after parallel work has
  begun. This is an execution-lifecycle failure, not a numerical gate result.
  The terminal V7 attempt contains only `reservation.json` and `failed.json`;
  `checkpoint.pt`, `result.json`, `completed.json`, metric, and gate artifacts
  were removed or never published. The failure receipt reports stage
  `verification`, intact journal integrity, complete owned-artifact cleanup,
  and `retry_authorized=false`; its reservation and failure file identities are
  `de5972f4...3a661` and `fec22c76...a7957`. V7 cannot be retried. Additive V8
  is being designed to run deterministic independent verification in a fresh
  isolated child process and return evidence to the lifecycle-owning parent for
  validation and publication. It requires a new pre-implementation amendment,
  frozen source, and different-agent review before any exact execution.
- **Camera V8 terminal numerical result.** The isolated-verifier successor
  passed independent review (`133/133` combined checks); its canonical review,
  report, and QA identities are `fd095eea...58f0f`, `5939c605...f19b`, and
  `14ba6f54...94d8`. The sole exact seed-`20260710`, N=5 attempt then completed
  all 400 updates and fresh-process verification on the R9700 without touching
  GPU1. The infrastructure repair therefore worked, but the unchanged numeric
  gate failed `7/26` checks. Every ground-clear distance/family/overall check
  passed (overall balanced accuracy `0.9972425`), while hit/no-hit balanced
  accuracy was `0.6633691`, hit-depth p95 error was `0.5386402 m`, raster NLL
  was `1.0748963`, raster balanced accuracy was `0.5617928`, free recall was
  `0.0007044`, unknown recall was `0.6846741`, and wrong-RGB raster balanced-
  accuracy drop was only `0.0059975`. The canonical gate and metric file
  identities are `cfe39b64...c081` and `b28cbd37...6b89`; their content
  identities are `11f02aa3...4ddf` and `c3bf90bc...7090`. The attempt is
  terminal: retry, N16, later-rung design, checkpoint use, V5 training, G2,
  held-out, runtime, hardware, production, and promotion licenses are false.
  The next camera work must be an additive scientific successor based on this
  failure, never a V8 retry or a threshold change.
- **Camera V9 source work.** The read-only V8 diagnosis showed that the ordered
  first-hit loss gives one group to no-hit rays and one group to every
  represented hit-depth bin, while the gate weights hit and no-hit states
  equally. The V9 pre-implementation amendment is frozen at
  `ccc8097b...b0a`. V9 preserves the model, five-frame panel, physical
  rasterizer, wrong-RGB controls, and all 26 thresholds; it replaces only the
  first-hit objective with equal-weight presence plus conditional-depth NLL and
  preregisters 4,000 final-only updates. A non-root author is implementing the
  source-only candidate now. The still-mutable pure loss and adversarial-test
  slice passes `9/9` CPU-hidden checks, including hand arithmetic, state/bin
  balance, invariances, zero-gradient empty groups, and finite gradients at
  logits of plus/minus 10,000. The deterministic 4,000-update/20,000-exposure
  full-panel schedule is frozen in source logic at `fb5a6c13...380`; root also
  reproduced that hash, all 41 diagnostics, final-only selection, and the new
  four-equal loss names through both the trainer helper and the production
  executor's isolated CPU-smoke relaunch. Policy and trainer now bind the V9
  amendment, diagnosis, reviewed V8 chain, terminal result/metric/gate bytes,
  and new loss/checkpoint schemas without reading the V8 checkpoint or numeric
  payload. Targeted loss/metadata/training/preflight checks pass `13/13`; the
  complete synthetic/lifecycle closure now passes `146/146`, including the
  trainer and production executor CPU smokes at exactly 4,000 updates, 20,000
  exposures, the frozen schedule hash, the new four loss names, independent
  metric reconstruction, and final-update-only checkpoint semantics. The
  frozen production source identities for loss, policy, trainer, verifier, and
  executor are `52bc99f0...ffd`, `00e0cbc7...be1`, `af8baa9a...88f`,
  `43142be5...09e`, and `94cbe45f...246`; synthetic support, loss tests,
  lifecycle tests, and handoff are `fd12a7dd...d3e`, `5bb9e1c3...27b`,
  `d7a7048d...3ee`, and `50e22a56...4f2`. Compile, pyflakes, ASCII, and both
  CPU contract smokes pass. A different agent is reviewing those exact nine
  files. No V9 data, checkpoint, GPU, or output path has been opened.
- **Latest raw-supervision update.** Builder V6 is author-frozen under the
  pre-implementation V6 amendment. Its source, CLI, focused test, and handoff
  identities are `88c36063...40d7`, `089aca48...828d`,
  `acf5ca8c...ecdd0`, and `d2cf130a...e80b`; author checks pass `56/56`,
  predecessor checks pass `61/61`, and metadata checks pass `45/45`. It opens
  and retains the complete source and staging trees inside one inotify-backed
  transaction through final validation, no-replace rename, post-rename drain,
  and poisoning on unexpected mutation. Independent review nevertheless
  reproduced one final ancestry race and issued `BLOCK`: after post-rename
  validation and parent fsync, an ancestor above the watched publication parent
  can be moved and replaced; `require_final_quiet()` sees no event, returns
  success, and leaves the retained dataset outside the canonical path. The
  independent QA is `1` pass / `1` decisive fail. Its test, BLOCK file, and
  BLOCK content identities are `2c74e331...c89a`, `55d50a38...c163`, and
  `c639170b...d74b`. Builder V6 is terminally ineligible. Auditor V6 is frozen
  only as a compile-safe successor input: its source, CLI, tests, and handoff
  identities are `cf67c993...00b8`, `de37e42d...4ffe`,
  `6cc84a49...f764`, and `f7e0c124...dc15`; focused and retained checks pass
  `60/60`. Builder/Auditor V7 is now author-frozen under the earlier
  pre-implementation amendment `ebeb552a...98fc`. Builder source, CLI, tests,
  helper, and handoff are `c79e68a2...2ab`, `9fdecaac...432`,
  `cb033519...9da`, `588e75ae...b6b`, and `b4fc0199...42d`; its focused V7,
  retained V6, V5/V4, and metadata checks pass. Auditor source, CLI, tests, and
  handoff are `3550917e...490e`, `9940d35e...949e`, `6d123d39...894f`, and
  `1351a264...fd21`; its focused V7 checks pass `25/25`, retained V1/V2 checks
  pass `37/37`, and its exact 55-row Builder V7 authority map matches. Builder
  V7 independently PASSED, but Auditor V7 independently BLOCKED after a
  deterministic terminal-boundary race: an ancestor move injected after its
  sole final event drain lets it return success while the canonical audit
  report is absent. The canonical BLOCK file/content identities are
  `180cce0d...1545` / `da2715a...b502`; combined review checks pass `68/68`.
  The additive Builder/Auditor V8 terminal-quiet amendment is frozen at
  `054de82d...6c88`. Builder V8 is a mechanical authority rebind of passing V7;
  Auditor V8 must use two drains followed by a final retained-ancestry,
  inventory/hash, and destination-identity check. A source-free scheduler
  amendment at `392745c8...698` rebinds the two implementation roles to
  already allocated, distinct non-root agents without changing science,
  source paths, review separation, or exact authority. Builder V8 is frozen at
  source/CLI/test/handoff identities `f4553335...aa35`, `f6471f1f...72d`,
  `fc1f0cf3...d9a`, and `9f4898e3...56b`. Author checks pass `69/69`, retained
  V7 checks pass `65/65`, and the 80-definition AST comparison preserves the
  passing V7 science and transaction. Different-agent review now PASSES: its
  independent QA, report, and canonical PASS-file/content identities are
  `85e4f90a...547f`, `adbef4a5...0e4c`, and `74b39df6...dd27` /
  `c12c037b...4122`; `314` applicable checks, compile, hash closure, and
  one/six-worker byte equivalence pass. Auditor V8 is now author-frozen at
  source/CLI/test identities `fb585b4e...d87`, `13c1ebed...fc2`, and
  `4270c1a1...006`; its explanatory handoff is `ed3fdf3d...18d2`. Focused V8
  checks pass `56/56`, retained V7 checks pass `31/31`, applicable V1/V2 checks
  pass `63/63`, and the six historical predecessor exploits remain explicitly
  excluded rather than reinterpreted. Different-agent review froze Auditor V8
  as `BLOCK`: after the second drain and final ancestry check, moving and
  recreating an ancestor alias at entry to the final report helper is accepted
  because both report lookups reach the retained inode and no later event read
  consumes the queued ancestor events. Reviewer QA, report, BLOCK file, and
  BLOCK content identities are `5fe390c3...229b`, `0d253de2...6489`,
  `63aa9f07...93a8f`, and `fdc52fe9...4f15`; sequential evidence passes
  `56` V8, `31` V7, `63` applicable V1/V2, and the decisive reproducer. V8 is
  terminal and cannot authorize exact work. A source-free Builder/Auditor V9
  linearization amendment is frozen at `6fba5de8...f773`. It preserves all
  science, puts a third and final event drain after every ancestry/report
  validation, defines that drain as the publication commit point under the
  governing same-user threat boundary, and makes consumer rehashing mandatory.
  The fixed Builder and Auditor V9 authors are implementing independently in
  parallel. No raw authorization, build, or audit is authorized yet.
- V4 ladder-v3 source passed different-agent review and the V2 `N=5`, seed
  `20260710` trainer published a complete immutable attempt. That attempt is
  now terminally invalid before metric publication: matched loss stores
  `0.27940133213996887`, while one quarter of its four stored components is
  `0.27940132907242515`; the `+3.067543719037502e-09` difference exceeds the
  frozen `1e-9` tolerance. A CPU-only diagnostic shows the complete frozen
  validator detects no second invariant after a forbidden in-memory repair,
  but no artifact was changed and no repair is authorized. The machine record
  is frozen at file/content SHA-256 `1744a50b...24f560` /
  `7bdaae6e...b8602`. The full-panel successor's additive structural-trigger
  amendment is frozen at `1e08aac0...490262`. Its five exact sources are now
  author-complete with `48/48` CPU/source tests passing, including transactional
  reservation fault injection and real post-training RGB rehash mutation
  coverage. Different-agent review nevertheless blocked that V1 successor:
  callers could forge/replay its importable authority marker, a fixed staging
  name could strand the sole attempt after an uncatchable pre-rename death, and
  the published directory was not parent-fsynced before a post-rename failure.
  The frozen BLOCK record is file/content SHA-256 `ccd8d979...506e` /
  `99ded56d...63e7`. Additive V2 closed those three defects but its independent
  review found a further authority-replay surface: mutable lifecycle records
  were reachable through Python function closures. Its frozen BLOCK is file /
  content SHA-256 `ddca89e...fc73` / `c4d93bba...fd8a`. Standalone V3 replaces
  caller-held lifecycle authority with one reviewed high-level operation. Its
  author suite passes `19/19`, the retained closure passes `48/48`, and the V2
  exploit remains reproduced as `1` pass / `3` deliberate failures. Policy,
  executor, synthetic-support, test, and handoff SHA-256 identities are
  `b0f5929a...46d3`, `8a8bec79...af66`, `83af899f...5c91`,
  `730513d7...beee`, and `c97b3f76...f8db`. Different-agent review froze V3
  as BLOCK: partial lifecycle writers and constructible stage values remained
  callable, claimed-directory identity was not retained through publication,
  review aliases passed, and source reads did not retain parent identity. The
  independent suite is `7` pass / `8` decisive fail; test, BLOCK file/content,
  and report identities are `b7d36691...2395`, `d1f859ae...e168` /
  `d84152d6...7f2f`, and `d28eadce...53e7`. Additive V4 is required to make
  the isolated script the only synchronous lifecycle and retain descriptor
  identities throughout. V4 is author-complete: policy, executor, synthetic
  support, author tests, and handoff SHA-256 identities are
  `ff291b94...4485`, `19cbdc56...afe4`, `01e49c30...9fb5`,
  `299fd18b...9688`, and `4e0aa7e2...925e`; focused `23/23` and retained
  `90/90` pass. Different-agent review froze V4 as BLOCK: transient replacement
  of the repository-root ancestry and claimed-output ancestry could be accepted,
  and an exception during verification/finalization after training could leave
  a completed attempt without a terminal gate result. The independent suite is
  `14` pass / `3` decisive fail; test, report, and BLOCK-file SHA-256 identities
  are `2942b232...6e9f`, `7edeff73...b18c`, and `d2224049...c507`.
  Additive V5 now retains/checks the complete source and output descriptor
  chains and terminalizes every post-training failure. Its policy, executor,
  synthetic support, author tests, and handoff SHA-256 identities are
  `cc28934b...f2c1`, `5dcc77a7...c2e1`, `7601341c...f608`,
  `80f51db2...8264`, and `df3d58ef...8b5c`; focused V5 passes `32/32`, the
  retained V1-V5 author closure passes `111/111`, compilation passes, and the
  CPU contract smoke reproduces 400 updates / 2,000 exposures. Different-agent
  review passed its distinct adversarial suite `23/23`; the independent test,
  review, and canonical PASS JSON SHA-256 identities are
  `1bd1f3d2...cefb0`, `d07407b5...ba59`, and `81345d13...cb0`. That record
  licensed exactly one fresh `N=5` attempt. The attempt ran on GPU0 from about
  13:18 to 13:20, reached the first publication assertion after training, and
  then terminalized before writing a checkpoint, result, completion, metric,
  or gate. A retained regression test running in parallel created and removed
  a temporary direct child under the shared `.generated` ancestor at exactly
  `13:20:26.803623735`; V5 bound that ancestor's size/mtime/ctime and therefore
  failed closed two seconds later despite its owned output subtree remaining
  unchanged. The immutable reservation and failure-receipt file SHA-256
  identities are `f8062f2e...501a` and `7ead7600...0af`; the failure content
  identity is `84cfa81a...58b`, cleanup is complete, and retry/later-rung/G2
  authority is false. No numerical result survived or was inspected. The V5
  lane remains terminal. A replacement infrastructure attempt requires a new
  namespace, an additive preregistration frozen before implementation, stable
  no-follow identity checks that tolerate unrelated shared-ancestor timestamp
  churn, and a fresh different-agent source review. Exact execution is now
  serialized against every other `.generated` mutation. The additive V6
  recovery amendment was frozen before implementation at `1fa4279c...aa90`.
  V6 is author-frozen in a new absent output namespace: policy, executor,
  synthetic support, tests, and handoff identities are `75b987dc...d549`,
  `79110340...8a3d`, `8df835de...96f`, `2af8b434...017b`, and
  `4ca14a5d...57e1`. Focused `40/40`, applicable retained `103/103`, compile,
  and isolated CPU contract smoke pass. Its shared-ancestor churn, persistent
  alias/identity/security replacement, exclusive-subtree mutation, claimed
  directory, source, cleanup, and terminalization cases are covered.
  Different-agent review nevertheless froze V6 as BLOCK: both its claimed-file
  and derived-directory refresh paths can absorb an unrelated create/delete
  interleaving when the final inventory is restored. The author suite remains
  `40/40`; the distinct review is three passing controls and two decisive
  failures. The review test, Markdown, and machine BLOCK file/content
  identities are `bd2379d7...6e51`, `c1ac98c3...4692`, and
  `ff1becd9...4d1b` / `98260f2b...fb1`. No exact V6 execution occurred. V7
  lifecycle-recovery authoring is active under a new absent namespace. Its
  pre-implementation amendment must precede all source, and its owned-output
  transaction journal must validate the exact declared directory events plus
  full pre/post descriptor-relative inventories before committing a captured
  fingerprint. Generic mutable-directory refresh is forbidden. Exact remains
  unauthorized.
- The legacy G5 posterior mathematics, exact writer lease, reversibility, and
  scaling result remain useful controls, but its same-grid evidence authority
  is not the downstream interface. The additive two-grid G5 evidence boundary
  is now hash-frozen and passed its author verification: `22/22` focused,
  `50/50` legacy G5, and `14/14` G3 V2 (`86/86` total). It issues exact
  single-use `0.10 m` posterior evidence from runner-owned `0.05 m` cells and
  remains synthetic/development-only. The separately implemented reversible
  two-grid posterior passes its independent review. Target-router V1 was
  blocked on a competing-hypothesis crossing and mutable rehashed authority;
  additive V2 closes both exploits and independently passes all focused,
  lifecycle, adjacent, G3/G4/G5, and world-waypoint composition tests. It
  remains development-only with production and hardware authority false.
- G3 V2 is complete and independently reproduced: `24/24` scenes, `96/96`
  endpoints, `192/192` route probes, and zero unsafe FREE, independent-label,
  component, or route mismatches. It remains an exact-control result with zero
  learned observations and no production-promotion claim.
- G4 V2 is independently passed: visibility/sweep/entropy use the `0.05 m`
  physical frame while route/history use the `0.10 m` configuration frame;
  exact live G3 components, frontiers, paths, revisions, frames, and supports
  are retained. Focused `8/8` and adjacent frozen `30/30` passed.
- World-waypoint V1 was correctly blocked because its serialized receipt did
  not explicitly deny production promotion. The additive V2 successor closes
  that sole issue and independently passes: V1+V2 `11/11`, G3 `14/14`, G4
  `8/8` (`33/33` total). Its exact high-index `0.10 m` world-centre conversion,
  exact path binding, one-use receipt, and both hash-bound authority denials
  are verified. This is development composition authority only.
- Native learned projection V1 passed its author suite but independent review
  blocked its retraction lifecycle on post-commit target mutation and a stale
  reservation that could permanently strand active evidence. Additive V2
  closed those public-path defects but is independently blocked: its reachable
  composed V1 adapter permits callers to bypass the V2 lifecycle and retract
  mutated evidence. Standalone V3 removed that adapter, but independent review
  found two remaining reachable failures: its callable core commit omitted the
  final immutable target binding, and a permanent duplicate-observation reject
  could retain a LIVE reservation forever. Additive standalone V4 closes both
  cases. Its focused suite passes `15/15`; the V4, frozen V3, revisioned-memory,
  and configuration-projection panel passes `73/73`. The frozen source and
  focused-test SHA-256 identities are `66486f70...f16a` and
  `df9b8977...0abb2`; its author handoff is `79407230...cbce`. V4 is
  independently BLOCKED: an observation identity that was used and later
  retracted remains permanently forbidden by memory, but V4 checked only the
  active identity set and left the failed reservation LIVE. The independent
  suite is `6` pass / `1` decisive fail; review, BLOCK-file/content, and test
  identities are `659b4ad6...d491`, `e52a0431...0f33` /
  `38f22ba9...d9f7`, and `e598ee44...b312`. Additive V5 is adding a narrow
  immutable view of append-only observation-ID history and using that to
  distinguish permanent release from transient retry. Its source, shared-memory
  source, focused tests, history tests, and author-handoff SHA-256 identities
  are `5ccd22e8...eca1`, `bb05f957...d483`, `e5f0d30b...d077`,
  `20860a1a...fa4`, and `6fb25e5a...61cc`. Focused/history checks pass `19/19`,
  the decisive branch subset passes `3/3`, and the adjacent panel passes
  `80/80`. V5 now independently PASSES: its distinct test and review SHA-256
  identities are `dcf0b8c1...ac77` and `a465dd61...757e`; the independent
  suite passes `9/9` and the combined retained panel passes `89/89`. This
  licenses the stated synthetic development projection boundary only;
  production, hardware, G2, held-out, and promotion remain unauthorized.
- Development coordinator V1 composes the G3/G4/G5, posterior, router V2,
  waypoint V2, and observer chain and passed its author suite, but independent
  review has reproduced four blockers: target namespace splicing, cross-scene
  snapshot/manifest splicing, evidence consumption before late input failure,
  and forgeable reconstructed observer results. V1 remains blocked. Additive
  standalone V2 closes those four defects and passes its author suites, but
  independent fault injection found one final transaction-envelope hole: a
  failure while constructing or registering the controller record can occur
  after downstream owner state and outcome consumption have committed, leaving
  no controller record and a stale coordinator seal. That decisive BLOCK is
  frozen in review / machine-receipt SHA-256 `abe943e9...3252` /
  `46fecbb0...721`; its independent probe is `3` pass and `1` deliberate fail,
  while `71/71` adjacent tests pass. Additive V3 moves record construction,
  insertion, and seal assignment inside the rollback envelope. Its source,
  tests, and author-handoff SHA-256 identities are `6d8b00aa...6523`,
  `d2af0e5a...a54`, and `df7c9234...abe6`; the focused suite passes `32/32`,
  the frozen blocker passes `1/1`, and the adjacent suite passes `71/71`.
  V3 now independently PASSES: its distinct test and review SHA-256 identities
  are `be84fba2...fb4b` and `2129e9ca...79d9`; independent `28/28`, frozen
  author `32/32`, and adjacent `71/71` pass. Frozen V1/V2 failure evidence
  remains reproduced separately. This is still observer-only development
  composition, not production or hardware authority.
- V5 model/output/loss source remains passed. The staged lifecycle successor
  uses additive V2 stage revisions, carries one exact role-manifest identity
  through G2, G3, and full publication, requires full promotion's G2 report to
  equal G3's exact G2 predecessor, preflights every output before access, and
  executes through an isolated fixed launcher that captures the shared core.
  Its focused CPU suite passes `30/30`, and the combined V5 suite passes
  `70/70`. Independent review reproduced the exact manifest/predecessor/source
  chain and returned PASS. All six production authority identities remain
  unset until a qualified checkpoint permits binding the G2 runner first.
  The exact candidate boundary is recorded in
  [`lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md`](lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md).
- The V5 architecture has exact fit-model migration and reviewed joint-loss
  arithmetic, but the repository currently has no reviewed executable that
  performs joint V5 training, the matched no-JEPA ablation, selection,
  calibration, or immutable training-record publication. This is now an
  explicit readiness blocker between the V4 fit ladder and one-shot G2; the
  staged G2 lifecycle cannot substitute for missing checkpoint production.
  The full-training execution amendment V1 is author-frozen at preregistration
  / handoff SHA-256 `b21d01d0...a4a7` / `fa0a497f...d6bc`. It fixes the primary
  V4 seed migration rule after both camera seeds pass, byte-identical promoted
  and no-JEPA initializations, 8,000 updates at effective batch 16, the
  mandatory JEPA plus four-equal current/next V4 objective, selection at the
  selected update, vector calibration, fixed per-family gates, immutable output
  and access ledgers, GPU0-only execution, and the one-shot G2 boundary. Its
  independent review froze BLOCK on three specification defects: the GPU smoke
  is simultaneously ordered before and after exact-attempt reservation; a live
  status file is byte-bound as an immutable authority parent; and a causal
  generalization claim is allowed from the same role used to select the
  promoted checkpoint, with an incomplete final precision comparator. The
  independent test passes `4/4`; test, review, and BLOCK-record SHA-256
  identities are `b2959ea1...6b38`, `2cd1bf56...125c`, and
  `c3debd1e...6273`. Additive V2 closes those three gaps and independently
  PASSES. Its amendment, author handoff, independent test, review, and PASS
  record SHA-256 identities are `b521d288...f66d`, `13102b0a...dbfa`,
  `734a140f...e0b`, `f4b22ef6...2dae`, and `6a53a3c9...89d4`; independent
  `8/8` pass. The complete V2 policy, preflight pair, exact executor, trainer,
  verifier, tests, and handoff are now source-frozen. Independent review passes
  `26/26` source/review checks plus `82/82` retained V4/V5 model/loss checks;
  its canonical review file/content identities are `2ce422c2...32a` /
  `7081c60b...6f4`. This licenses only the exact source closure and payload-free
  preflight boundary. Execution remains blocked by 19 null manifest bindings,
  Raw V8 integration, the pre-G2 V5 checkpoint-schema conflict, and the V9
  loss successor. No full-training data, model checkpoint, accelerator,
  protected role, or exact namespace was opened.
- Full development raw supervision for that trainer is preregistered across
  all 72 train scenes, 8 checkpoint-selection scenes, and 8 calibration
  scenes: 5,172 paired transitions and 9,460 exact unique endpoint identities.
  Its metadata-only planner is author-complete and reproduces all 5,172 pairs,
  10,344 endpoint instances, and 9,460 exact unique endpoint identities. It
  also reduces the 96-scene rendered source index to the exact 88 development
  scenes without opening referenced payloads; focused tests pass `9/9`. The
  ledger records zero G2 sidecar or payload, RGB, label-shard payload,
  model-output, runtime, held-out, hardware, or production opens. Independent
  review blocked V1 because lexical containment accepted an in-repository
  symlink resolving outside the repository. Additive V2 now rejects lexical and
  resolved escapes, symlink, hardlink, non-regular-file, root-alias, and
  device/inode alias substitutions while preserving the exact preregistered
  identities; its focused suite passes `17/17`. Different-agent review froze V2
  as BLOCK: parent-directory replacement and leaf hard-link replacement after
  validation could still redirect the one allowed source-index open. The review
  and adversarial-test SHA-256 identities are `376a8a76...c65a` and
  `3d9a8203...b0c`; `24` checks pass and the two exploit probes deliberately
  fail. All exact scientific identities and the zero-forbidden-open ledger still
  reproduce. Additive V3 replaces validation followed by absolute reopen with
  a descriptor-relative no-follow walk. Its source, tests, and author-handoff
  SHA-256 identities are `0adc6bfa...1247`, `f1f0bff9...aa78`, and
  `66f55b34...8160`; V3 passes `24/24` and the combined V1-V3 author suites pass
  `50/50`, including both V2 races and a FIFO replacement probe. Independent
  review froze V3 as BLOCK on two stricter continuity cases: a changed ancestor
  while opening the repository root, and a same-inode leaf whose pre-open file
  fingerprint had changed. The independent suite is `14` pass / `2` decisive
  fail; test, report, and BLOCK identities are `af32942f...3b0`,
  `95b20b53...a824`, and `f22ed2cb...c058`. Additive V4 is walking from the
  filesystem root with no-follow directory descriptors and binding the complete
  pre-open leaf fingerprint. Its source, tests, and author-handoff SHA-256
  identities are `d6282a6e...1de2`, `724f1c93...a0e0`, and
  `4753d835...1415`; V4 author tests pass `26/26`, the focused continuity subset
  passes `3/3`, and V1-V3 regressions pass `50/50`. Independent review froze V4
  as BLOCK: it rechecked directory identity/type after reading but not each
  component's complete original fingerprint. The independent suite is `9` pass
  / `2` decisive fail; test, review, and BLOCK SHA-256 identities are
  `5e079be9...0f78`, `46d44155...9757`, and `6897064f...c2d8`. Additive V5 now
  retains and rechecks every directory-component fingerprint before and after
  the read. Its source, focused tests, and author-handoff SHA-256 identities are
  `67c4d325...2921`, `384af6e2...9636`, and `b362d263...ba66`; focused V5
  passes `19/19`, continuity passes `6/6`, and the combined V1-V5 author panel
  passes `95/95`. The exact metadata-only boundary still reproduces 5,172
  pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 development
  scenes, ten allowlisted opens, and zero payload, G2, excluded-role, or
  protected opens. V5 now independently PASSES: its distinct QA and review
  SHA-256 identities are `8a50bcf5...1298` and `7d7344e4...7706`;
  independent `26/26`, combined V5 `45/45`, and the V1-V5 author panel `95/95`
  pass. This licenses builder and auditor source implementation only. The
  six-worker raycast builder is author-frozen: source, CLI, tests, and handoff
  SHA-256 identities are `3bc15597...25ec`, `df5fd60b...eeb3`,
  `15767446...2e4`, and `9d9aee5f...d28c`; focused `15/15` and adjacent
  `103/103` pass. Different-agent review nevertheless froze the builder as
  BLOCK: it opened and hashed caller-listed authorization sources before proving
  the list contained the complete fixed nine-role set, so an invalid partial
  authorization reached an arbitrary referenced-source opener. The independent
  suite is `8` pass / `1` decisive fail; test, report, and BLOCK-file/content
  SHA-256 identities are `306b02e9...2104`, `51e5a8ac...a44`, and
  `116099c2...57d` / `b3623cd9...c76b`. A successor must split complete
  authorization validation from all source opening and prove zero openers on
  every invalid authority. The auditor is author-frozen: source, CLI, tests,
  and handoff identities are `854d4330...798c`, `246a8de1...908d`,
  `6dfe991e...1557`, and `7d693902...279d`; focused `12/12` and applicable
  adjacent `115/115` pass. Different-agent review froze the auditor as BLOCK on
  four independently reproduced defects: its exported callback API could claim
  `exact=True` without the sealed loader, invalid one-role authority opened a
  caller-selected source before rejection, multiply linked dataset leaves were
  accepted, and floating-point/boolean JSON cardinalities passed as integers.
  The independent suite is `26` pass / `6` decisive fail; test, review, and
  BLOCK-file/content identities are `9684b14c...76fc`, `a61b64e3...dca8`,
  and `c427b927...8f5b` / `4a8235ed...6915`. The additive builder V2 is
  author-frozen: source, CLI, tests, and handoff identities are
  `0ae5ddd8...e71c`, `c1139687...d303`, `6755044a...0339`, and
  `7f278c5c...c04f`; focused `27/27`, retained applicable `50/50`, and broad
  applicable `228/228` pass. Different-agent review nevertheless froze V2 as
  BLOCK on four reachable authority bypasses: the imported `_v1` object exposes
  the blocked V1 exact entry, the worker pool accepts an unauthenticated caller
  callback, compatibility bridges temporarily replace the process-global V1
  authority validator, and production phase two accepts caller reader/root/
  parent-skip seams. The independent test, review, and canonical BLOCK file/
  content identities are `2c34fec9...3d43`, `e42a5876...ccad`, and
  `726e03fd...e4a` / `6696b9b7...a0d6`; the result is nine passing controls and
  four decisive failures. Auditor V2 is author-frozen with source, CLI, test,
  and handoff identities `d57aacd4...b2a`, `4502ac44...ae9`,
  `45d60db1...399`, and `6a338b7c...7ab`; focused `25/25`, retained applicable
  `63/63`, and broad applicable `193/193` pass. It validates the fixed authority
  before the dataset manifest and independently derives strict integer
  populations, rejects hard links, and exposes no exact callback. It cannot be
  licensed unchanged because its nine-role authority is cross-bound to blocked
  Builder V2. The coordinated V3 amendment froze at `501062e2...8b2b` and its
  standalone Builder V3 reached a compile-safe but intentionally unfrozen
  implementation checkpoint. Auditor V3 did not become a review candidate: a
  post-handoff partial edit changed its declared source identity and left
  unresolved legacy references. The terminal structural-invalidation record is
  `db86ea8b...e213`; the stale handoff, declared source, and changed source are
  `a3b66f15...fe23`, `08cbbc8b...9606`, and `42316470...043`. No V3 review or
  authority is eligible. The coordinated V4 successor amendment was frozen
  before V4 source at `a535ee8d...ed83`. Standalone Builder V4 was author-frozen
  with source, CLI, role-test, and handoff identities `e46f42db...93e0`,
  `db14bb15...901`, `80ca9d1d...61c0`, and `575ae2a5...3bdb`; its focused
  author suite passed `30/30`. Independent review froze V4 as BLOCK because its
  complete second metadata/source validation runs before audit-sample and
  manifest construction, manifest publication into staging, staging inventory
  checks, and fsync. A source can therefore change after the promised final
  validation and before the atomic dataset rename. The distinct review suite is
  one passing frozen-identity control and one decisive failure; test, canonical
  BLOCK file, and BLOCK content identities are `116b81f6...68e2`,
  `4c91d7ce...dc4`, and `34cfd6b1...ea6`. Auditor V4 stopped at compile-safe
  non-candidate source checkpoint `d030122e...e0d`; no CLI, test, handoff, or
  review candidate was created because its required Builder V4 authority cannot
  pass. The additive V5 amendment froze at `fe6a29a2...9e91`; Builder V5 then
  froze with source, CLI, role-test, and handoff identities
  `8d85635a...dce2`, `3116c2a5...8019`, `6b49d5d5...05d6`, and
  `a8037613...9d26`. Its author suite passed `31/31` and correctly moved all
  declared staging construction before the final source pass. Independent
  behavioral review nevertheless froze V5 as BLOCK: changing `pairs.jsonl`
  inside the final source-pass callback was published successfully because the
  post-pass code checked only the staging directory inode, not its file
  inventory or content. The test is one passing frozen-identity control and one
  decisive failure; test, canonical BLOCK file, and BLOCK content identities
  are `fc0ba7af...812`, `2687d43d...307`, and `5fd83545...1e9`. Auditor V5
  stopped as a non-candidate at source/CLI checkpoints `6df29a2f...855` and
  `3f2b99ff...b27`. Additive V6 has reached the frozen builder and active
  auditor/review state summarized at the start of this section. No exact
  construction, manifest, payload, protected role, output, or accelerator has
  been opened.

## Current closure order

1. Finish and independently review Camera V9, then run its one exact N5 gate.
   Only a full unchanged-metric pass may license the two-seed/N320
   representation ladder; never retry or reuse a failed predecessor.
2. Implement and separately review Raw Builder/Auditor V8. Only dual PASS may
   create the nine-row authorization; a human must then separately supply its
   frozen hash before the exact scene-disjoint build and audit.
3. After V9 and Raw V8 pass, freeze and review an additive full-training
   successor that carries the hierarchical loss, V8 raw authority, and valid
   pre-G2 development-checkpoint schema.
4. Run matched JEPA/no-JEPA training, independent reconstruction, selection,
   calibration, and immutable development-checkpoint publication only after all
   source, preflight, raw, and camera bindings are non-null and reviewed.
5. Bind and execute V5 G2 runner authority only for that qualified shared
   checkpoint; later stages remain sequential and unset.
6. Bind the qualified checkpoint, learned projection, G3/G4/G5 memory,
   posterior, router, waypoint receipt, and observer-only claim evaluator into
   one CPU development smoke command with per-tick raw outcomes and an
   actual-open ledger.
7. Run a different-agent complete-chain audit. Navigation work is ready only
   after that audit and the development smoke pass; held-out navigation remains
   sealed and outside this milestone.

The exact downstream changes for step 4 are frozen by the read-only
[`two-resolution integration gap audit`](lewm_go2_two_resolution_navigation_integration_gap_audit_2026-07-13.md):
versioned G4/G5 frame handling, runner-owned learned admission, a deterministic
target router, world-waypoint receipts, and one observer-only development smoke.

V5 model source readiness is independently passed, including the corrected
`(B,D,H,W)` tensor contract and mandatory four-equal raw/derived V4 objective.
The output-binding cycle and the subsequent role-chain/source-capture defects
are closed and independently passed. V5 execution source is ready, while every
production stage remains fail-closed pending the qualified checkpoint and its
strictly sequential authority binding.
