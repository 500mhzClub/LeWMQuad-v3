# RGB Swept-Progress Survival Joint-JEPA V3 Half Occupied-Safety Auxiliary — Preregistration

- Status: frozen before V3 implementation or runtime access.
- Purpose: one final midpoint falsification of the directly observed free/occupied trade-off; this is not a coefficient sweep.

## Evidence and decision

- V1 coefficient `0`: free `0.885680` passed; occupied `0.644302` and rough occupied `0.580587` failed. Every progress/control gate passed.
- V2 coefficient `1`: occupied `0.777180` and rough occupied `0.768724` passed; free `0.838621` failed by `0.011379`. Every progress/control gate passed.
- In V2, `95.426%` of free errors were predicted OCCUPIED, directly matching the boundary controlled by the auxiliary.
- A straight extension is rejected because semantic and auxiliary losses plateaued. A nonlinear semantic head was independently proposed, but is a larger change and is deferred until this single obvious midpoint test is resolved.

## Sole scientific delta

- Change only `OCCUPIED_SAFETY_AUX_COEFFICIENT` from `1.0` to exactly `0.5`.
- Preserve the exact V2 occupied logit, per-row present-binary-class balancing, current/next averaging, `log(2)` normalization, and update-1 joint route.
- Total remains `L=S+P+U+R+O`, where `O` now includes coefficient `0.5` and is separately traced.

## Frozen remainder and lifecycle

- Preserve exact V2 model, accepted N320 encoder-only initialization, RGB/data/labels, masks, action order, seeds, schedule, optimizer, clipping, EMA, cap, evaluator, controls, thresholds, bootstrap, and family gate.
- Fresh model from accepted N320 only. Never read, hash, load, copy, resume, or warm-start either rejected V1 or V2 checkpoint/runtime state.
- One fresh write-once attempt, exactly 1,000 updates / 16,000 presentations, with no retry or resume after update 1.
- If every gate passes, run the matched no-JEPA arm before any JEPA treatment-effect claim. If any gate fails, close coefficient tuning and do not run another weight.
- No G2, navigation, sealed, held-out, production, deployment, or promotion access is authorized.
