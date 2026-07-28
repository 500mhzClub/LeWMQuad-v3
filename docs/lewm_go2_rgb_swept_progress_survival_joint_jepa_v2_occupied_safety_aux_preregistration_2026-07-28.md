# RGB Swept-Progress Survival Joint-JEPA V2 Occupied-Safety Auxiliary — Preregistration

- Status: frozen design before implementation or V2 runtime access.
- Independent design review: PASS; no result-invalidating issue or equally small better alternative found.
- Repository goal remains a fully learned RGB-only joint-JEPA perception/navigation stack, validated later on untouched held-out mazes.

## Why one successor is warranted

- V1 completed exactly 1,000 updates / 16,000 presentations and passed every swept-progress, family, persistence, shuffled-action, wrong-RGB, and action-prior gate.
- V1 failed only occupied recall (`0.644302 < 0.70`) and rough occupied recall (`0.580587 < 0.65`).
- V1 training was stable and largely plateaued, so an identical extension is rejected. Lowering thresholds, selecting an intermediate checkpoint, reusing the V1 checkpoint, changing data, or changing the encoder is rejected.

## Sole scientific delta

- Add one parameter-free occupied-vs-rest auxiliary `A_occ` to the existing online current/next semantic logits and labels from update 1.
- Class order remains `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2`.
- For logits `z`, binary occupied log odds are `b_occ = z_occupied - logsumexp(z_unknown, z_free)`.
- Apply binary cross-entropy with logits against `label == OCCUPIED`.
- Reduce independently per row by averaging the mean loss for each binary class present in that row; an absent class contributes no term. Average current and next row means with weights `0.5/0.5`, then divide by `log(2)`.
- Coefficient is exactly `1.0`; no detach and no new parameter or head.
- Total loss becomes `L = S + P + U + R + A_occ`. Report `A_occ` separately.

## Everything preserved from V1

- Exact model architecture, accepted N320 encoder-only initialization, constructor seed `20260712`, execution/bootstrap seed `20260728`, RGB-only inputs, development roles, label files, action order, swept masks, schedule, optimizer, learning rates, clipping, EMA, four-by-four accumulation, and 1,000-update / 16,000-presentation cap.
- Exact V1 survival, ranking, semantic, and JEPA-persistence terms; all V1 controls, metrics, thresholds, paired bootstrap, and eight-family conjunctive gate.
- Fresh model from accepted N320 only. The rejected V1 checkpoint and V1 runtime state are forbidden inputs and must not be read, hashed, loaded, copied, resumed, or warm-started.

## Lifecycle and decision

- One write-once V2 attempt on the exact R9700 runtime. No retry or resume after update 1.
- A V2 PASS requires every existing V1 gate; no post-result threshold or metric change is allowed.
- If V2 passes, the next authorized experiment is the matched no-JEPA training arm before any JEPA treatment-effect claim.
- If occupied recall still fails or any swept-progress/control gate regresses, close this successor rather than add another coefficient or extend training.
- No G2, navigation, sealed, held-out, production, deployment, or promotion access is authorized by this preregistration.
