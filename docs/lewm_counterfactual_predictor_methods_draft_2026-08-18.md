# Methods and reproducibility draft

The experiment uses the frozen Go2 simulator/robot visual contract and eight
scene families. The factorial training/selection records and the 240-branch
counterfactual qualification corpus are frozen receipts. The counterfactual
panel contains twenty states with twelve candidate action sequences per state.

RGB observations are encoded with the frozen V-JEPA 2.1 ViT-L/16-384 target
encoder using the registered crop (rows 28:196), 512×384 resize, ImageNet
normalization, and dense [768,1024] final-layer tokens. The predictor receives
RGB context plus the registered post-slew action plan and control-history
tokens; prohibited inputs include future observations, future proprioception,
oracle utility, and simulator privileged state.

The one-step arm predicts H1. The rollout arm autoregressively predicts H1 and
H2 using its own prior prediction in the next context, with the registered
objective `1.5*e1 + 0.5*e2`. The deployment-valid proprioceptive subset uses
the same objective and registered control history with proprioceptive inputs.
The four-step diagnostic predicts H1–H4 with equal per-horizon losses and is
not treated as a sample-matched replacement for the historical controls.

Training uses eight registered seeds, fixed initial weights, fixed data order,
the registered 24-epoch budget, epoch-21 retention, AdamW and the frozen
checkpoint policy. No best-epoch selection or post hoc extension is used in
the primary factorial result.

Counterfactual evaluation snapshots one common state, applies twelve
deterministic candidate post-slew action sequences, restores state between
branches, and encodes the realized future with the same frozen target encoder.
Direct fidelity uses changed-token cosine and normalized error relative to the
frozen persistence baseline. Action specificity ranks candidates by predicted
versus realized future latent similarity and reports top-1/top-3, MRR,
pairwise accuracy, rank and margin diagnostics. H1/H2 prefix degeneracy is
reported rather than silently treated as independent action evidence.

The occupancy probe is frozen, qualified only at its registered horizons, and
is a co-outcome. It is not refit and H1 is not reinterpreted. Equal-family
aggregation is primary: rows are reduced within episode clusters, clusters
within families, then families are equally weighted. Corpus-weighted estimates
are secondary. Training seed is the replication unit; intervals are two-sided
95% t intervals with df=7 and critical value 2.3646242510102993.

Technical-validity rules require complete receipts, exact checkpoint identity,
finite tensors, deterministic branch restoration, and complete metric schema.
Technical failures are preserved as non-results and are never converted into
scientific evidence. All artifacts and source/result digests are listed in the
closure report and package manifest.
