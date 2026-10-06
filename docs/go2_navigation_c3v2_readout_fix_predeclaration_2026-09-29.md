# C3-v2 readout fix and C4 refit: pre-declaration, 29 September 2026

Approved by Andrew Knowles on 29 September 2026 as a change to the readout's training data, not a harness change. **Committed before any refit or acceptance evaluation.** The harness `v4_completed_support` stays frozen. The V-JEPA 2.1 encoder and the action-conditioned predictor are unchanged. C3-v1's validation result (13/20) remains the pre-registered capability result.

## 1. Sets (registered before collection)

Everything below is in `c3v2_sets_v1/registry.json` (`168f579b…`). It was built with construction seed 2026092901 and excludes every prior graph: the capability inventory, which was reconstructed in memory and hash-verified 90/90 without writing or displaying sealed contents; the audit layouts; and all earlier registries, including the maze-view training and transfer mazes.

- **Fresh check:** layouts 0–9, one episode each. This set is used only for the closed-loop check in §6.
- **Recording, fit split:** layouts 10–13, 16 cases.
- **Recording, held-out split:** layouts 14–15, 8 cases, used only for offline acceptance.

Recordings run the maze-view collector with a fixed 19-s tape: nine 1.2-s holds, each followed by a start from rest (forward, left or right arc, left or right in-place turn). Windows follow the maze-view rule: contact-free, departure ≥ 10 frames, all eight horizons present.

## 2. Design choice, stated explicitly

- **The v1 readout** (`maze_data_final.pt`) was fitted on the encoder's features of the **actual** future frame. At run time it decodes the predictor's **predicted** future features.
- **That mismatch costs about half its accuracy.** On two held-out development mazes, 700-ms translation RMSE was 27 mm with actual futures and 47–52 mm with predicted futures (progress report §7.3–7.4). On C3's validation states it predicts 10 mm for forward moves, against 66 mm from command history.
- **Coverage alone is unlikely to fix it.** v1's maze training data already contained starts from rest and in-place turns: its tape alternates holds, turns, forward runs and arcs.
- **So the approved training-data change has two parts:**
  1. new rest-start and turn recordings on unseen mazes;
  2. readout training inputs that are the frozen predictor's predicted-future features, conditioned on the executed applied command tape, which is exactly C3's run-time computation. Targets are the physics-true motion.

## 3. C3-v2 readout fit

- **Architecture and initialisation.** Same architecture, normalisation and initialisation as v1 (`go2_full_heading_readout_v1_attempt_001/mixed_data_final.pt`, `bbbb05fd…`).
- **Optimisation.** Same as v1: AdamW (lr 1e-3, weight decay 1e-4), gradient clip 1.0, 440 updates of batch 64 with balanced horizons, seed 2026092205, fixed final checkpoint.
- **Batches.** Each batch is 32 v1 "old" contexts plus 32 drawn from the v1 maze contexts together with the new fit-split contexts. Each context supplies its current pooled features and predicted pooled features at every horizon.
- **Excluded data.** No held-out, check, validation or sealed data is used.

## 4. C4-v2 refit (matched data)

- Same plan as C4-v1: architecture (about 17.4M parameters), preprocessing, 1,760 updates, seed 2026092513, fixed final checkpoint.
- The data is the C4-v1 population (the same 5,966 + 2,448 readout contexts) plus the new fit-split contexts, prepared exactly as for v1: three causal frames, the applied-command history and the executed applied command tape.
- The GPU cap still applies: at most 12 GPU-hours in total for C4 training.

## 5. Offline acceptance for C3-v2 (all must pass)

**How it is evaluated.** Evaluation runs on the held-out recordings (layouts 14–15) through the deployed pipeline: frozen encoder, then frozen predictor with the executed applied tape, then readout. Results are compared with physics-true motion. C3-v1 is evaluated on identical windows.

**A. Starts from rest.** These are windows whose preceding 1.0 s of applied commands is zero and whose 800-ms tape contains a translation command.
1. The median ratio of predicted to true 800-ms translation magnitude lies in [0.75, 1.25].
2. The 800-ms XY RMSE is at most 50% of C3-v1's.

**B. In-place turns.** These are windows whose 800-ms tape contains at least one turn command and no translation command.
1. The 800-ms XY RMSE is at most C3-v1's.
2. The median of (|predicted XY| − |true XY|) at 800 ms is at most 10 mm, meaning no spurious translation.
3. The 800-ms yaw RMSE is at most 1.05 × C3-v1's.

**C. No loss elsewhere.**
1. All other held-out windows: XY and yaw RMSE at 500 and 800 ms are at most 1.05 × C3-v1's.
2. The existing development transfer population (240 fixed windows on two mazes, §7.3–7.4), with action-predicted features: XY and yaw RMSE at 500 and 700 ms are at most 1.05 × C3-v1's.

**Reported but not gating:** C4-v1 and C4-v2 on the same windows.

**Pipeline exactness, required before any acceptance number counts:**
- The offline context construction must reproduce the deployed native contexts captured from C4 validation 11/0 exactly.
- The offline C3-v1 pipeline must reproduce C3-v1's logged run-time predictions on its validation 11/0 decisions (maximum absolute difference ≤ 1e-6).
- These checks read preserved logs and frames only; nothing is re-run on validation.

## 6. Closed-loop check and the E1 version rule

- **The check.** If C3-v2 passes §5, C1, C3-v2 and C4-v2 each run once on the 10 fresh-check episodes. They use the frozen V4 harness (only the prediction-slot model files change), the frozen readers and the prefix/reader errata.
- **The E1 rule for C3:** C3-v2 enters E1 **only if** it passes every §5 criterion **and** has zero disallowed contacts and zero hard-clearance violations in the closed-loop check. Otherwise C3-v1 enters E1. Round-trip counts from the check are reported but do not select: 10 episodes cannot rank versions reliably.
- ~~**The E1 rule for C4:** C4-v2 enters E1, per Andrew's decision. A contact or hard violation in the check is a stop for his decision.~~ Superseded by Amendment 1 (§7).
- **Model versions.** C3-v2 and C4-v2 are recorded as new model versions with checkpoint hashes. Andrew confirms before E1 launches, and the sealed set is untouched.

## 7. Amendment 1: C4 follows C3 into E1 (Andrew's decision, 29 September 2026)

**When this was committed.** After both fits had started and before any C3-v2 or C4-v2 acceptance number was computed. Both fits recorded the original text of this document (sha256 `96227b55…`, commit b72b76b9) in their `plan.json`. The amendment changes no fit setting and no §5 criterion.

**The pairing rule.** The C4 that enters E1 is the one trained on the same data as the C3 that enters. The pairs are C3-v2 with C4-v2, and C3-v1 with C4-v1.

1. **If C3-v2 fails §5, or fails the check's safety bar** (any disallowed contact or hard-clearance violation), C3-v1 and C4-v1 enter E1.
   - Both already have closed-loop validation results, so no C1 + C4-v2 check is run.
   - C4-v2 is recorded as a model version that does not enter E1.
2. **If C3-v2 passes both, and C4-v2 has any disallowed contact or hard-clearance violation in the check**, the report flags it and Andrew decides.
3. **Otherwise** C3-v2 and C4-v2 enter E1.

**Reason.** C4 was refit only to keep it on exactly the data C3's readout uses. C4-v2 alongside C3-v1 would give the supervised model data that the JEPA readout isn't using, which would confound the C3–C4 comparison.

**E1 training seeds.**
- E1's additional training seeds use the selected pair's data and training procedure.
- Every seed enters E1 whatever its offline numbers: acceptance selects the version, not seeds.
- Each seed's §5 acceptance measures are reported.

**C4 scoring (reported, not gating).** C4-v1 and C4-v2 are scored with the same §5 measures as C3 on the same windows: held-out groups A, B and C1, plus the C2 transfer population. C4-v2 is compared with C4-v1. None of these numbers selects a version.
