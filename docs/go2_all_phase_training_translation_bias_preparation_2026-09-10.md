# Expanded-data correction preparation

The original translation-intercept estimator requires every available context
to have a valid motion target. The bound expanded windows instead contain4010
available contexts,3974 with motion support and36 with no valid motion horizon.
A direct reuse would reject those36 contexts. A separate estimator now retains
their identities and schedule draw counts while assigning zero XY correction
weight. It does not delete/reassign contexts or alter neural/contact training.
For supported rows it reproduces the original estimator exactly.

The new receipt explicitly says native training targets are used, runtime
native state and transfer targets are not used, and no raw native artifact is
opened by the estimator. It avoids an unqualified inherited no-native-data flag.
Runtime correction preserves yaw/contact/unknown padding and base model state,
is evaluation-only and accepts only the original causal model inputs.

Seven focused tests passed85634 in2.36s:exact old estimator parity, motionless
row accounting, no transfer-label access, invalid clock/scope/support rejection,
and exact NumPy/runtime correction parity with an unchanged base model. Runner
and admission imports/syntax passed23710. The full coefficient runner has not
yet executed against completed expanded fits.

Actual-data accounting probe60473 closed0. It used synthetic zero predictions,
not a trained model or empirical error measurement, and the exact real target
rows/three schedules. Each schedule retains all4010 contexts and7200 draws.
Motionless draw counts are145/143/148 for seeds2026091001/2026091401/2026091402.
Valid motion counts by horizon are3974,3854,3734,3614,3494,3376,3258,3140.
No correction artifact or optimizer update resulted from this probe.

Prepared sources and hashes:
- lewm/all_phase_translation_bias_development.py:
  8e21721da0e0d7ceaf0f7c44f590b3697ccd88afaafd240cb40c973b2eb3f460
- lewm/tests/test_all_phase_translation_bias_development.py:
  175b2b67aa5b42c50c5d04dea414c9e3af37e420da2812fb7d426e8734796971
- scripts/fit_go2_all_phase_training_translation_bias_v1.py:
  6c481c81c1bc2394a3596a8b6af8d4d1be31c589972328b7a1f16889e5fa4a53
- scripts/all_phase_translation_bias_model_admission_development.py:
  66c82bd082317e00168e655094c5e6cf4518c78e17e63826537078a425b94105
- docs/go2_all_phase_training_translation_bias_v1_2026-09-10.md:
  94259c128cc48b1cfcea0a8ff8f9fc43d2ae22b2ba421c09d793c5243eff8f61

After the original full-fit run completes, authenticate all18 models with
scripts/all_phase_model_admission_development.py. The correction runner accepts
--fit-result-sha256 with that observed result, repeats full model admission and
reconstructs all30 training-only head intercepts before corrected evaluation.
Its output is exclusively go2_all_phase_training_translation_bias_v1_attempt_001.
The separate correction admission reconstructs every coefficient again before
loading an explicitly assigned runtime model. Future navigation still needs
its own bound protocol; these components make no navigation qualification claim.
