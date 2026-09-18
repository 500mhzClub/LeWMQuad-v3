# Augmented family/switch eighteen-fit result

All eighteen prescribed fresh models completed 1,200 updates each (21,600
total). Full admission reconstructed every update ledger, final snapshot and
raw score. The three seeds share initialization across their six matched arms;
the first seed exactly matches the original study's initialization. Both roles'
predictions were saved before scoring. No intermediate checkpoint was selected.

The four-worker benchmark passed exact paired ledgers and model identities and
measured 3.677x speedup. Scientific fitting took 1,313.811 seconds after launch;
maximum worker RSS was 2,703,716,352 bytes, minimum available RAM 75,065,720,832
bytes and minimum artifact space 70,967,357,440 bytes. These are CPU fits.

The fixed mixed schedule uses 600 original-source and 600 branch-source batches
per model. All old unavailable windows and new contact-censored targets remain
in the accounting. Transfer data never enter optimization. This evaluates
parameter-cluster transfer within the development family, not novel mazes.

New-source transfer first-500-ms XY error, mean and sample standard deviation
across the three optimization seeds, in millimetres:

| Predictor input | Direct | Supervised rollout | JEPA |
| --- | ---: | ---: | ---: |
| Full | 24.436 ± 5.370 | 32.707 ± 14.634 | 43.736 ± 5.449 |
| No predictor RGB | 25.644 ± 7.306 | 25.103 ± 4.653 | 35.333 ± 14.703 |

These standard deviations describe optimization variation, not confidence
intervals over independent mazes. All 72 new transfer branches have valid
first-horizon motion and negative contact labels; all-horizon scoring contains
464 motion and 576 contact targets, including 112 contact positives.

For the matched first seed, full-direct first-horizon XY error worsened from
15.617 to 21.346 mm, while yaw error improved from 0.049398 to 0.025617 rad.
Full JEPA improved from 42.606 to 38.314 mm and from 0.129178 to 0.093118 rad,
but still trails direct on motion accuracy. Full-direct mean all-horizon XY
error is 31.302 mm and contact Brier 0.063018; full JEPA is 52.747 mm and
0.079245. There is no broad motion-accuracy improvement or demonstrated JEPA
advantage. No native goal result follows from these predictive scores.

The first full-JEPA seed 2026091001 remains the primary native candidate and
the same-seed full-direct model remains its fixed comparator, as specified
before these results. All eighteen models are admitted before either is used.
The earlier observer failure, failed arrivals and timing limitations remain.

Artifact roots are direct children of the established navigation development
artifact root. Exact identities:

- `go2_augmented_family_switch_fits_v1_attempt_001/launch.json`:
  `758139a0663d900f6b166271cc0270f5ee818f377bc24d9bb70459c6b42f6943`.
- Its `result.json`:
  `d692829f385c2ba89c198cb0e7292d78f439969a384d47779f9258a609e38eda`.
- `go2_augmented_family_switch_fit_readout_v1_attempt_001/launch.json`:
  `54e49614d3b6f09092698ff88477c37c0607f0ef12b564f34871570c44d79f5f`.
- Its `metrics.json`:
  `aa6b4e1384ed2d1b546379e522e9585064b001df4ea033c73567802acd51a2df`.
- Its `result.json`:
  `335e673e02c85bb67a83bf9dd0d74a97710d5bfef9a2c43d6d31f5508ff860ab`.

Fitting binds 886 source paths and 202 artifacts (153,076,945 bytes); readout
binds 896 source paths. Native navigation, calibration, real-time operation
and deployment remain unqualified.
