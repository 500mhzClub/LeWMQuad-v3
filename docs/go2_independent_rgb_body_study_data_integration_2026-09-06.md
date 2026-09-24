# Independent-layout prediction study: authenticated data integration

The new read-only loader is
`scripts/independent_rgb_body_study_data_development.py`. It closes the gap
between the fresh modality-specific terminal audit and the existing matched
prediction/hazard evaluators. It performs no model fitting, physics, checkpoint
selection, source export or navigation qualification.

## Receipt and full-population contract

The future study must freeze an explicit SHA-256 receipt for `launch.json` and
`rgb_body_layout_audit.json` from each of the twelve fixed fresh batches. The
loader does not discover roots or derive its own expected audit hash from current
bytes. Missing receipts, live/incomplete batches, collection/audit failure records,
different output rosters and changed bytes are rejected. No old pilot or legacy
collection root is accepted.

For each batch, it verifies the receipt before parsing, verifies the exact 127
audit-output bindings, and reuses the terminal ledger reader to authenticate
collection sources, inventory, raw artifacts, byte counts and ordered commits.
The audit-launch witness must bind those exact sources and inputs. Every terminal
episode evaluation must equal its bound raw precheck. This authenticates an
already executed raw audit; it does not pretend to reexecute physics or the raw
sensor audit itself.

The join recomputes RGB/body eligibility from authenticated coverage and
measurement diagnostics, checks all 120 episode identities and preserves setup
failures and exclusions. It checks all versus eligible window/target products,
original layout roles, all prefixes, full population counts, strict versus hard
measurement failures and materialization counts. All departures, including
excluded histories, pass target decoding and inventory joins. Boundary-only
strict depth failures remain explicit and do not become depth-navigation evidence.

`load_study` requires all twelve receipts and joins metadata in fixed batch order.
A zero-eligible layout remains in the planned denominator and reported exclusions;
it is not silently dropped or replaced. If every layout has zero eligible samples,
loading fails instead of fabricating a dataset. The resulting evaluation reports
all planned train/selection/development-evaluation layouts. No image-tensor cache
or raw packet collection is retained by this metadata join.

The existing evaluator's conservative `source_artifacts_verified_by_interface`
field remains false: that pure evaluator does not authenticate files. The outer
study report separately records verification of its receipt-bound inputs. A
dictionary supplied directly to the internal metadata helper is not provenance.

## Verification

Focused run 35703 completes exit 0: **140 passed in 10.31 s**, including 35 new
synthetic receipt/join tests, the six-action hazard tests and the existing
collection, independent-evaluation and cumulative-event suites. Tests cover
partial/selected populations, zero-eligible layouts, role/identity corruption,
changed receipt/product bytes, raw-precheck disagreement, missing artifacts,
source/input witness mismatch, inconsistent targets, eligibility and summary
counts. Synthetic receipts are not recorded scientific evidence.

Read-only 16277 completes exit 0: all **765 launched collection source bindings
remain unchanged**; the new loader is not in that live source closure. The full
233-file regression 38292 completes exit 0: **3,042 passed in 235.49 s**. This
includes the 35 new loader tests and 20 new hazard tests, not only the older
2,987-test collection regression.

Loader SHA-256:
`1c22a1acafc95a5bc6f34c4a40c97c6824825ccda6a95f47283abc59f167a9de`.
Loader-test SHA-256:
`86c3925ed4c2c201397be1fc988be3563bad687ceb0106a566ba16139dd62706`.
These are development source identities, not a frozen model-study launch.

## Separate bounded check of the live collection

Read-only 60462 completes exit 0. It verifies 3,119,124,477 raw artifact bytes
for the first 54 committed/prechecked cases and rechecks the frozen launch/source
identities. These contain 125,550 physics samples and 1,755 RGB-D frames. All
**45/45 nonreference sibling-prefix comparisons** match exactly across nine
context/history/support groups. All 54 are prospectively RGB/body eligible, with
no hard measurement failures. One strict boundary-depth failure remains recorded
in `l00_junction_quiet_nominal_a3`, frame 31. No repair or qualification is implied.
All positive-contact target counts in this bounded subset remain zero.

This early check is not a terminal batch audit, completed independent-layout
study, model result or novel-maze success. It must not be fed into `load_study`
through fabricated terminal receipts. The single l00 stage continues untouched
under handle 92647 and automatically terminal-audits its collection. Subsequent
live metadata check 04aee2 observes 69/120 raw-prechecked cases, all eligible,
with eight strict boundary-depth failed frames, no hard failures and no positive
contact targets yet. There is no collection result, collection failure or final
batch audit at that check. PID 2055172 is independently observed live at 2,021 s
elapsed. These later metadata counts do not extend the 54-case raw-hash/prefix
verification above.

## Next execution steps

1. Finish the existing stage and inspect its authoritative terminal collection
   and audit. Preserve physical outcomes, sensor failures and infrastructure
   failures. Do not restart or launch a competing audit while it is live.
2. Following valid completion, continue fixed l01 through l11 as the frozen
   preceding-batch and storage requirements permit. Freeze exact receipt hashes
   only after each terminal audit succeeds. Verify the new loader against actual
   terminal data; synthetic joins alone do not prove recorded-data integration.
3. Complete a bounded streaming sample/inference runner and the prospective
   matched experiment: direct, supervised rollout and JEPA; paired model seeds;
   identical training exposures and budgets; empirical action/time and zero-motion
   baselines; explicit RGB/history/action ablations. Preserve missing layout/action
   coverage and contact censoring. Do not silently train on a selected subset
   because collection failed or produced inconvenient outcomes.
4. Report motion, yaw and actual-horizon cumulative-event metrics alongside the
   fixed two-second six-action hazard diagnostic. Uniform outcomes supply no
   ranking contrast. The three development-evaluation layouts are three topology
   units, not hundreds of independent frame-level replicates.
5. Establish useful predictive contributions and reliable local execution before
   matched online-rollout and memory/backtracking tests. Whole unfamiliar-maze
   completion, real-time/deployment-valid sensing and bounded hardware evidence
   remain required. The previous 0/3 return result and negative learned-versus-
   empirical comparison remain unchanged. The full scientific goal is unachieved.
