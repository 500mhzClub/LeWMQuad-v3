# RGB Memory-Role Factorized Joint-JEPA V1 Split-Integrity Amendment

Date: 2026-07-30

Status: frozen before implementation commit, RGB access, GPU use, or scientific output

This amendment corrects one metadata-adapter defect discovered by the real
metadata-only preflight after the original preregistration commit
`01d78284a22a52816a41f31a78411491714b4f9c`. The H6 and place indexes are each
internally scene-disjoint, but their role namespaces were composed without a
cross-index exclusion. Seven of the eight place checkpoint-selection scenes
therefore occurred in H6 train, and three place-train scenes occurred in H6
validation. Using those panels unchanged would leak scene identity across the
combined probe's development train/selection boundary.

The place-role split is derived from the same frozen raw role namespace as the
physical route: all eight place checkpoint-selection scenes are the eight
physical checkpoint-selection scenes, and all place-train scenes are a subset
of the 72 physical training scenes. Runtime must verify those identities and
apply the exclusions against the complete physical 72/8 scene inventories,
not merely infer them from the selected place rows.

The science-identical adapter correction is exact:

- Preserve the model, accepted N320 initialization, all source indexes and
  their hashes, physical schedule, data fields, seed, optimizer, losses,
  margins, thresholds, observation updates, 400-update limit, and
  12,800-presentation cap.
- For the local training route, traverse the corrected H6 V2 train index in
  frozen order, omit rows whose scene is in the frozen place
  checkpoint-selection panel, and take the first 3,200 remaining rows without
  cycling. This selects H6 source indices through 3,229, skips 30 earlier rows,
  and has ordered source-index SHA-256
  `263e72b1bfff24b059d1d46f0ec1859dbc497602e82c3f5e02f628e4f26809a5`.
- For local checkpoint selection, traverse the corrected H6 V2 validation
  index in frozen order and omit rows whose scene is in the frozen place-train
  panel. This leaves 1,994 of 2,048 rows across 147 scenes, with ordered
  source-index SHA-256
  `a9344429cdafca23cbce8e26ef18756423ac364c12bb1c4d3af78e1ab4a533b9`.
- Reindex only the runtime-safe local selection view from zero through 1,993;
  retain each source row's exact `e2`, `e3`, and `actions[2]` payload.
- Require the union of effective local/place training scenes to be disjoint
  from the union of effective local/place checkpoint-selection scenes before
  any RGB open or GPU use.

The corrected local-training family row counts are 400, 402, 404, 402, 400,
395, 401, and 396 in the preregistered family order. The corrected local
selection family row counts are 256, 256, 256, 250, 256, 219, 245, and 256;
every family retains at least 212 non-hold rows. The combined metadata census
has 1,013 training scenes, 155 checkpoint-selection scenes, and zero overlap.

This amendment authorizes no retry, resume, extra update, alternate seed,
checkpoint access, navigation, benchmark, probability-calibration, G2,
held-out, or sealed access. The one-shot attempt remains unconsumed.
