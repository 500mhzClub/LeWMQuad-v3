# Fresh model assignment for the longer comparator controllers

The existing independent-study factory builds older controllers and assigns
only primary-seed full JEPA/supervised models. The newer longer comparator
pipelines deliberately require model admission from their caller. The new
`scripts/extended_return_budget_comparator_factory_development.py` supplies a
construction helper for those comparator implementations, using the original
eighteen-model training roster rather than silently substituting an older
controller or supporting only a selected model.

This helper does not define, revise, freeze or execute a population roster.
A future launcher must authenticate the complete correction admission and
bind each case's comparator mode, model name, corrected tensor SHA and public
mission before calling it. The existing 32-case independent-study roster and
prepared single-case longer native launcher remain unchanged.

The helper accepts an explicit immutable `ModelAssignment(name,
model_state_sha256)` for predictive modes. It checks the name against the
original 18 assignments: training seeds 2026091001, 2026091401 and 2026091402,
full/no-RGB inputs, and JEPA/supervised-rollout/direct objectives. It requires
the original completed correction identity
`1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460`,
all 18 models and all 30 trained heads. It uses the native planner's existing
adapter loader on every call, requires the exact `AllPhasePlannerModel` interface,
checks the returned objective and input variant against
the original roster, and checks the assigned tensor digest before and after
construction. All model modules must be in evaluation mode on CPU, without
parameter gradients. Model inference during construction is rejected even
when a constructor catches the attempted-forward exception.
The adapter preserves the original corrected tensor keys and bytes; it adds
the interface required by the existing forecast selector without another fit.

The fixed navigation budget is 8,000 steps. Controller construction uses the
same mode-to-controller mapping as the prepared comparator collection/audit
pipeline. Predictive modes are `frozen_reference`, `nominal` and
`current_planning`. Reactive construction rejects both model assignments and
correction admissions, never loads a model and checks that the returned
controller has neither a world model nor a learned residual. Fresh controller,
memory and model instances are constructed; this is not an upgrade or resume.

Nominal inference exclusion during later navigation remains the existing
pipeline's responsibility. Nominal control retains predictive nominal
trajectories and observed residual correction. Reactive control remains a
whole-method comparator. Current-planning control removes accumulated routing
cells while retaining localization, contact, model history, residual and mission
state. This helper does not broaden those causal interpretations.

## Verification and identities

Source SHA-256:
`04a68bbc25cc844da58ce586637f2bb22109fd2ae298278eb35fdf2036ce099f`.

Test source
`lewm/tests/test_extended_return_budget_comparator_factory_development.py`,
SHA-256:
`8e7c7ec3adf115e2f3f572c6531b8ac70d28bea4e5662c837e4ad3fd29dda629`.

The revised focused test invocation, session 79338, exited 0 with **54 passed
in 3.88 seconds**. Tests construct the real prepared controllers and actual
planner-adapter classes using synthetic corrected-model fixtures. They cover all 18
assignments across the three predictive modes, fresh mutable state and separate
tensor storage, model-free reactive construction, invalid modes/assignments and
incomplete correction receipts, changed tensor identity/objective/input variant,
child-module training mode or gradients, construction-time tensor mutation or
model replacement, and swallowed root/child inference attempts. Six additional
integration cases cover every objective/input-variant combination through the
production adapter loader, replacing only its underlying checkpoint reload with
a synthetic corrected-model source. They execute the real forecast selector,
count its actual model forward, check finite predictions and unchanged state,
and verify that the unadapted source is rejected by that selector. This is not
actual trained-checkpoint reload, sensor or native-navigation evidence.

The earlier 47-test construction-only suite passed in session 16524 but did
not exercise the planner's model interface. Its source identity was
`e069511e0226db84b17c0703f73f4da5e4b52db8d4dc17727c3b1f2485375773`
and its test identity was
`bf6643f24f95c74b0587bd2fbbdcbfbeb8dd245551342f741bf3a243fb4407ee`.
Review against the actual native loader found that the factory selected the
raw corrected-model loader instead of the planner-adapter loader. The added
regression test reproduced this gap before correction: session 14916 exited 1
with one failed and 47 deselected in 2.02 seconds (`DID NOT RAISE ValueError`
for an unadapted corrected model with the correct tensor identity).
The loader correction and explicit adapter-type check address that failure;
no native experiment or prior saved result was changed to obtain the pass.

The exact focused pytest command used the original deterministic single-thread
environment with `python -B -m pytest -q -p no:cacheprovider` and the test path
above. This short test overlapped the running prospective controller-prefix
comparison; that comparison is not a timing benchmark. No second full-history
replay or native scene was started.

A new source-only preparation check, session 64752, exited 0 with 2,691 paths
in the union of prepared native, comparator, factory, compact-archive and sample
check source/test dependencies. All 2,656 live prefix source bindings and all
2,655 completed compact-sample source bindings were rehashed unchanged. The
factory source, tests and this preparation document are outside both launched
rosters. The longer native output remained absent. No source export, real-model
fitting, checkpoint selection, independent-layout execution or deployment followed.

The original prospective prefix owner remains PID 3043682, creation time
1789224352.44, tool session 67080, launch SHA-256
`49f6c5cd36fb67adc0532dd3d0971a507faa435169238e96b27dfef47c150089`.
At the latest source-check snapshot it was live beyond the frame-1700 progress
report with no completed result or failure. Its completed positive result is
still required before the already prepared longer native trial.
