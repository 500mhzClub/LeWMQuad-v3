# Bounded policy-only stream and matched study-runner preparation

This stage implements the remaining numerical and sample-materialization
mechanics needed for the independent-layout comparison. It does not launch a
recorded-data fit, select a checkpoint, change the live collection, or demonstrate
navigation. The final scientific objective remains active and unachieved.

## Implemented

`scripts/independent_rgb_body_study_stream_development.py` constructs a private
metadata snapshot from the twelve receipt-authenticated batches. It checks role
membership before sample reads, limits tensor batches to 16 samples, preserves
repeated training draws, and retains no persistent image cache. Each consumed
policy file is verified against its existing binding before and after sample
materialization; batch receipt identities are also rechecked. Errors latch the
stream instead of returning partial usable data or silently retrying.

The existing `load_route_observation` policy-only reader supplies only RGB,
body-sensor histories and past applied controls. The exact consumed roster is
the policy manifest, policy-history archive and explicitly indexed RGB files.
No depth, fast-gyro, shadow tracking, geometry, native pose, contacts or future
command tape is opened by this materializer. The already completed raw-audit
verification remains the loader's separate responsibility.

Training uses the same existing `PulseTimedDataset` target join and reads only
available future RGB observations. Inference is a separate path: it does not
call the target materializer, inspect future-target metadata, or read future
RGB frames. It constructs only the four fixed past observations and the proposed
pulse/brake plan. Although the stored sensor-history archive contains multiple
timestamps, only the four selected causal policy packets enter input tensors.
The ideal simulated sensing assumptions and hidden robot camera are unchanged;
policy-only does not mean hardware-valid.

`lewm/independent_pulse_study_runner_development.py` supplies two operations:

- `train_schedule` starts from a fresh cumulative-event trainer and verifies
  exact equality to the published layout/action-balanced train schedule, including
  every draw identity and multiplicity. Entirely absent planned training layouts
  cannot silently disappear. An explicit callback accounts for each completed
  optimizer update. Callback failures preserve the actual update count and latch
  the trainer; external mutation cannot rewrite later scheduled draws.
- `predict_heads` streams inputs without targets, requires the original proposed
  action plans, verifies exact output horizons/masks and finite active predictions,
  and returns raw arrays with identical ordered dataset indices for the existing
  motion and six-action hazard scorers. Direct is the primary head for the direct
  arm; recursive rollout is primary for supervised-rollout and JEPA arms. Auxiliary
  direct outputs remain explicit. Inference must leave parameter/buffer hashes
  and update counts unchanged and restore the previous model mode. Failures return
  no partial score set and latch the trainer.

These numerical operations do not choose a study budget, freeze a preregistration,
implement checkpoint persistence, or confer data provenance on a synthetic stream.
They are not yet a complete experiment executable.

## Tests and recorded integration

Initial stream tests 93368 pass 23 tests in 2.62 s. After adding the stricter
fixed-history check and the model runner, 52772 completes exit 0: **49 passed
in 6.69 s** (23 stream and 26 runner tests). Synthetic one-step direct/supervised/
JEPA fits are implementation tests, not recorded-data learning evidence. Tests
cover role leakage, repeated draws, missing or changed bindings, post-read changes,
future/target isolation, schedule mutation, absent training layouts, accounting
failure, exact inference row/timing contracts, model-state mutation and numerical
agreement across inference batch partitions.

Read-only 9454 completes exit 0 on the first six already committed fresh l00
episodes. After checking their artifact bindings, each new policy-only training
sample is exactly equal to the existing RGB-D replay materializer's full input
and target tensor tree, including NaN locations. The separate inference inputs
are also exactly equal to the training inputs. This tests all six action-duration
cells without fitting a model. All 765 live collection source bindings are checked
unchanged before and after. These are early integration diagnostics, not terminal
receipts or qualification of the live batch.

Full 235-file regression 63347 completes exit 0: **3,091 passed in 239.23 s**,
including all 49 new stream/runner tests.

Source SHA-256 identities:

- stream: `f880bfccc1ba34fe733b10c7f728ea39774b0fa0ff42042331949f83cda31ee5`
- runner: `bc8b8c19a93f4d27e322f68595a27ebb759a0dcafdd036172bad4f24e407f491`
- stream tests: `22044d6a9d917a847b19cebf1321743531bfb8de4a70246a1c6eb189f71adf92`
- runner tests: `71df3b072e9ccd46d49c4e1d6acee2404883061bd06c40a51f31ed5aa56d04be`

## Next execution

### First actual positive-contact and action-contrast integration

Read-only 70367 completes exit 0. It independently repeats the full raw audit of
`l00_near_wall_quiet_nominal_a1` and gets exact equality to its committed precheck,
with raw artifact bindings verified before and after. Disallowed contact occurs
at native sample 1,380 / time 2.762 s, **0.462 s after departure**. The record has
1,381 physics samples and 13 camera frames. It is a complete physical-terminal
acquisition, not a completed command schedule or a successful navigation action.

All five active cumulative-contact horizons are observed positive. All motion
and future-image targets are censored: contact occurred before the first 0.5 s
prediction horizon. The new policy-only materializer reproduces the existing
full tensor tree exactly, and its inference inputs match exactly. This is one
contact episode, not five independent collisions. There is no model fit or
depth-navigation qualification.

Read-only 10894 completes exit 0. It verifies all six near-wall/quiet/nominal
sibling artifact rosters, their complete 1,150-sample/nine-packet prefixes and
all five nonreference prefix matches. At the fixed two-second horizon, the six
observed contact labels are `[0, 1, 0, 0, 0, 0]`. Thus this one training-layout
group supplies an actual outcome contrast. Only action1 receives the separate
full raw reconstruction above; this check does not claim a full batch re-audit.

This result also exposes a limitation to retain in the study's coverage report:
positive collision labels need not provide any future-image latent targets.
Report contact/motion/future-image availability jointly by role, action and
layout. Do not manufacture post-contact observations or assume that JEPA directly
trains on an observed future image in every collision case. Hazard ranking alone
also cannot demonstrate goal-directed choice: an always-turning controller may
avoid a forward collision while never solving a maze. Keep the empirical
action/time baseline, context-sensitive probability metrics and later executed
goal-progress tests alongside the secondary hazard diagnostic.

### Remaining stages

The unchanged l00 stage 92647 is still live, with at least 89/120 cases
raw-prechecked at the last native output poll. PID 2055172 is independently
observed live at 2,671 s elapsed. No competing audit or retry should be launched.
The later bounded metadata check a0c171 observes **105/120 raw-prechecked cases**,
all RGB/body eligible, eight retained strict boundary-depth failed frames, no
hard measurement failures, and one positive-contact episode/five positive target
horizons. There is still no collection result, failure or terminal audit at that
check. PID 2055172 is again observed live at 2,907 s elapsed.
Finish its collection and automatic terminal audit, then validate the completed
receipt loader against real terminal artifacts. Continue the fixed remaining
layout batches only through their existing preceding-audit and resource checks.
A tested sequential supervisor may automate those exact stages, but must not
retry failures, resume existing outputs or launch model training.

Before a model fit, complete and freeze the scientific experiment executable:
actual terminal receipts, data/positive-contact/contrastive-action coverage gates,
matched arm/seed/exposure budgets, checkpoint persistence and reload verification,
baseline scoring and explicit RGB/history/action ablations. The same development
evaluation layouts and failed/excluded cases must stay in reported denominators.
Report paired layout effects, not frame-level independent N or a selected seed.

Reliable local physical execution still precedes meaningful online rollout and
memory/backtracking comparisons. Whole novel-maze missions, realistic real-time
sensing and bounded hardware evidence remain required. Nothing here changes the
previous 0/3 return result or establishes an advantage over the empirical motion
baseline.
