# V27 GPU-visibility integrity replacement V1 preregistration

## Purpose

The original V27 one-shot stopped during post-reservation runtime composition because ROCm exposed two devices: the authorized discrete R9700 and integrated Ryzen graphics. The frozen runtime requires exactly one visible GPU. The attempt completed zero V27 updates, zero presentations, zero optimizer or EMA steps, and produced no checkpoint, so the registered explicit-plan successor-state hypothesis remains scientifically untested.

This document authorizes exactly one science-identical infrastructure replacement. It is not a retry or resume of the consumed attempt.

## Frozen scientific identity

The replacement must preserve byte-for-byte the V27 model, dataset helper, tensor training core, and evaluator. It must preserve the RGB crop and normalization, corrected H6 V2 indexes and first 6,400 training rows, physical schedule and labels, seed, initialization, four ordered plan actions, gamma 0.9 discounted EMA target, losses, optimizer groups and learning rates, gradient clipping, EMA, observation updates, bootstrap, donor controls, hard thresholds, one joint optimizer/EMA step per update, 400-update cap, and 12,800-presentation cap.

The physical and H6 routes remain 16 presentations each per update. The sealed and held-out benchmarks remain unopened. Navigation and probability calibration remain unauthorized.

## Sole implementation changes

Only the denied-by-default launcher, executor lifecycle identity, and focused lifecycle tests may change:

- Use schema prefix `lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27_integrity_replacement_v1`.
- Use clean source root `/home/andrewknowles/Workspace/LeWMQuad-v3-v27-explicit-plan-successor-integrity-replacement-v1-source`.
- Use output root `.generated/go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27_integrity_replacement_v1/attempt_v1`.
- Require the exact environment selector `HIP_VISIBLE_DEVICES=0` before source activation, runtime composition, or reservation.
- Reject conflicting GPU visibility selectors before reservation. The launcher must not set or repair its own environment.
- Preserve the base runtime's post-reservation check for exactly one `AMD Radeon AI PRO R9700` with 34,208,743,424 bytes.
- Create the reserved attempt root as mode `0700` and the reservation receipt as mode `0444`.

The exact launch environment must set `HIP_VISIBLE_DEVICES=0` and remove conflicting GPU visibility variables. No other scientific or runtime-payload change is authorized.

## Lifecycle and terminal control

The replacement is one new attempt under fresh source and output roots. Retry and resume are false. Source closure, independent review, narrow-export certification, and a separately committed exact authority are required before launch.

On a scientific gate failure, stop with no checkpoint. On an exception, terminalize once with truthful partial-checkpoint accounting and quarantine. On a pass, publish only the update-400 development scale seed; its reuse still requires separate authority. A pass does not authorize G2, navigation, held-out, sealed, production, or promotion access.

There is no second visibility replacement. If the exact environment guard or reviewed R9700 check fails, the attempt is consumed and this replacement closes without scientific evidence.
