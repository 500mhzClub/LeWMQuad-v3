# Additional training seeds for the navigation comparison

The existing expanded-model admission contains 18 trained models: seeds
2026091001, 2026091401 and 2026091402, each with JEPA, direct and supervised
rollout training and full/no-RGB inputs. Current native comparisons use only
seed 2026091001. The additional neural models are already trained.

Prepare matched motion corrections for all three full-input conditions of
the two additional seeds. Use the existing fitter with explicit `--seed` and
`--condition`: `scripts/fit_matched_closed_loop_motion_residual_development.py`.
The fitting algorithm, four training recordings, validation recording, sampled
windows, targets, features, ridge penalty and stationary thinning are unchanged.
Every collected target/mask and window/group row must equal the existing JEPA
reference population. Coefficients are frozen before validation and no neural
weights change. Original seed-2026091001 outputs retain their names and files;
additional seeds use distinct output roots with their seed in the basename.

After the active native recovery run exits, fit pairs in this fixed order:
2026091401 JEPA/supervised rollout, 2026091402 JEPA/supervised rollout, then
2026091401 direct/2026091402 direct. Each pair uses the two existing disjoint
CPU groups. These small fits do not need new simulation recordings or depth
generation. Keep every result and failure; do not select the best seed.

These are preparation for prospective native comparisons, not new navigation
evidence. Report a training-method comparison as the effect of the complete
training-and-fixed-correction procedure: its outputs determine the downstream
correction fit. It does not isolate raw neural predictions from that correction.
Prediction-versus-feedback and routing-memory comparisons remain separate.

All six additional corrections completed in the specified three parallel pairs,
after the reactive native owner exited. Each took 38.4–40.4 seconds and matched
all original targets, validity masks and window/group rows: 1,207 training and
448 validation windows. The neural states were unchanged. Result, model and
fit identities are recorded in
`docs/go2_additional_training_seed_motion_fits_2026-09-15.json` (SHA-256
`0fe83678cd5c66359d156646f660b35b8de172c520c8d605e0cf9562a96564bf`).

Moving-window validation XY RMSE at 700 ms, in millimetres:

| Training seed | Method | Before motion correction | After motion correction |
| --- | --- | ---: | ---: |
| 2026091401 | JEPA | 19.927 | 8.162 |
| 2026091401 | Direct | 24.523 | 8.377 |
| 2026091401 | Supervised rollout | 15.896 | 8.218 |
| 2026091402 | JEPA | 22.277 | 8.314 |
| 2026091402 | Direct | 21.983 | 8.337 |
| 2026091402 | Supervised rollout | 18.641 | 8.257 |

The common correction procedure makes these validation errors similar, despite
larger differences before correction. This is an offline observation, not a
navigation ranking or a reason to select a seed. The next native study must
bind each assigned model to its corresponding correction, keep the same
controller/perception settings and report every assigned run. These new seeds
have not yet been tested in the current closed-loop navigation controller.
