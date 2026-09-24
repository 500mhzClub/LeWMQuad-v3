# V16 integrity replacement V2 preregistration

Date: 2026-07-29

Status: preregistered science-identical controller-integrity replacement only.
No V2 reservation, scientific input access, GPU training, checkpoint,
qualification, calibration, G2, navigation, or held-out access has occurred.

## Trigger and terminal predecessor

V16 integrity replacement V1 is terminal in commit
`ed0d70962872b86fbbc3fb07d17f2517869bee80` at
`docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v1_terminal_failure_result_2026-07-29.json`,
file SHA-256
`91d60bcf461dc38c508365f9d5ccb842580fe987f5d151524e4b325907222c79`,
content SHA-256
`34359c99117359bb18832496ec1f73695fcf8c86c9f4f36f850c954f015971ed`,
and byte count `8897`.

V1 completed exactly 100 valid optimizer/EMA updates and 1,600
presentations. Every training-integrity row passed, the V16 consistency loss
had positive support throughout, and observations at updates 0 and 100 were
published. No update-400 or terminal V16 scientific gate was reached and no
checkpoint was published. V1 is consumed and may not be retried, resumed, or
spliced into this replacement.

The immutable exception-message SHA-256
`40510175845988f13f6162ed8526f0b09f73384467fa855e1e79b44a56562a58`
exactly matches `1000`. After publishing update 100, the V16 lifecycle's bare
`else` branch treated update 100 as update 1000 and accessed the absent
`observations[1000]` entry. The frozen predecessor uses an explicit
`elif update == 1_000` branch.

The update-100 snapshot was materially improving but is not a scientific
verdict: physical margins increased from 32 to 57 of 189, total shortfall fell
from 184.61066427463246 to 97.50479292980872, rough depth p95 fell from
4.869073963165282 m to 2.432741713523863 m, ground balanced accuracy rose from
0.541935563806034 to 0.6252442672254054, and pixel balanced accuracy rose from
0.6326792707919727 to 0.8383567599169721.

## Sole implementation correction

- Replace only the update-400 branch's bare `else` with
  `elif update == 1_000`, restoring the inherited lifecycle condition.
- Add one source-only regression proving update 100 cannot enter the
  update-1000 final-gate branch.
- Change only replacement evidence selectors, preregistration identity,
  schema prefix, and fresh output/source roots required to distinguish V2.

No tensor operation, model parameter, architecture, data field, seed, schedule
element, loss, coefficient, optimizer setting, EMA operation, observation,
metric, control, threshold, stopping rule, checkpoint rule, or accounting
multiplier may change.

## Frozen scientific identity

Preserve original V16 preregistration commit
`2792343e14d3376add9d6adbda7f29346a3e9e29` and V1's reviewed dtype adapter
exactly:

- unchanged V14 RGB-only model, online encoder, unified ray-survival evidence,
  semantic state, action-conditioned predictor, and EMA target;
- unchanged float32 learned hazard/offset/ground evidence and inherited
  float64 query-geometry metadata validation;
- unchanged symmetric stop-gradient metric ray-consistency loss
  `C = C_base + 0.1*M` in the same joint-JEPA update;
- unchanged training-only realized SE(2), with RGB-only inference;
- fresh N320 initialization and the same constructor, schedule, experiment,
  bootstrap, and projection seeds;
- the same 4,262-pair train and 495-pair checkpoint-selection roles and exact
  first 16,000-presentation schedule;
- four microbatches of four, float32 AdamW, parameter groups, learning rates,
  clipping, one optimizer step, and one EMA update per completed update;
- observations at updates `0`, `100`, `400`, and `1000`;
- the exact V16 update-400, final, and continuation-eligibility gates;
- milestone full-state recovery publication only after a passing update-400
  gate and an eligible or fully passing update-1000 gate; and
- exactly 1,000 maximum updates and 16,000 maximum presentations.

There is one fresh V2 seed realization under the same fixed seed and exactly
one V2 attempt. There is no loss search, topology change, warm start,
automatic retry, automatic resume, or automatic extension.

## V2 identity and authority

- Schema/evidence prefix:
  `lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v2`.
- Fresh certified source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v16-ray-consistency-integrity-replacement-v2-source`.
- Fresh attempt root:
  `.generated/go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v2/attempt_v1`.
- Original V16 and V1 outputs remain terminal evidence only. No state or
  checkpoint from either is admissible for V2 execution.
- Source implementation, focused regression, recursive closure, independent
  review, narrow clean-export certification, and one-shot authority must be
  frozen before reservation or scientific input access.
- Any V2 source, authority, reservation, custody, exception, or gate failure
  is terminal. No V2 retry, resume, or further integrity replacement is
  preregistered.

Probability calibration, G2, navigation, held-out, sealed, production,
promotion, deployment, retry, resume, and extension remain unauthorized unless
earned later under separate ordered authority.
