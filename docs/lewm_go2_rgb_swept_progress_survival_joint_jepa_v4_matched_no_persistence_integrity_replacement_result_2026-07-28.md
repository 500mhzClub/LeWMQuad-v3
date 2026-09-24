# RGB Swept-Progress Survival Joint-JEPA V4 Matched No-Persistence — Integrity Replacement Result

- Terminal status: `COMPLETE_NEGATIVE_PERSISTENCE_TREATMENT` — a valid, complete negative result for the registered one-sided treatment predicate.
- The one allowed science-identical integrity replacement completed exactly once at 1,000 updates / 16,000 presentations. There is no retry, resume, second replacement, or further persistence-coefficient experiment.
- Three independent terminal audits found no result-invalidating blocker. They independently reproduced the canonical receipts, all trace identities and accounting, the absolute gate, and the treatment comparison.
- Result file/content SHA-256: `6d242f8dbdff90a6d46cdaa6b8c449f06897d3392854ee1aa3ac11d647f77e2f` / `a679e506f5f97cf85b7b62e002469b5737420132060a37974a2a9153eb1ef362`; byte count: 122,122.
- Training-trace file/content SHA-256: `f60e38d4ebcc693eefa96479150b912c869874121c6b0a611154157b5893d79d` / `9bbc1d773e8286eb4789c95d7263a5e289260a1901eb0344e4c8c87b7a00e9aa`; byte count: 414,447.
- The exact result and trace receipts exist and the mutually exclusive failure receipt does not. No produced checkpoint was opened, listed, statted, hashed, loaded, or otherwise accessed during review; its result-embedded metadata remains unverified and the artifact remains diagnostic, unqualified, and unusable.

## Frozen identity and integrity validity

- Replacement preregistration / source / execution-binding commits: `d5c25a3b11181aba29a2c96e9954c09c19b8f1ad` / `222550a4c26c7256b92d3d21ead03850f7b30ce2` / `6fd6bd3cb5a32f6a0e9459cb89d4e384965fd4d0`.
- Replacement executor/test SHA-256: `d2cc1781beae234df0964713b44fc74ce5baeb314a2172ebfb48903f28a9c2e0` / `abebde301c6540da3474f25f9380a2ae4f9f5f332b114e94fca83e27da581e6e`.
- The result and trace both record `science_changed=false`. Model, decoder, accepted N320 initialization, RGB/data/labels, seeds, schedule, losses, backward membership, optimizer, clipping, EMA, evaluator, gates, controls, bootstrap, thresholds, and cap matched the frozen control.
- The sole replacement delta was the duplicate terminal validator: both loss identities now use the training core's exact `math.isclose(rel_tol=2e-6, abs_tol=2e-6)` predicate instead of the failed executor's absolute-only `<=1e-6` check.
- The largest full-diagnostic identity error was `1.7695128917694092e-06` at update 658; the largest backward identity error was `4.023313522338867e-07` at update 57. Both are valid under the frozen core predicate, and the former explains the terminal attempt-V1 adapter failure.
- Reconstructed initial-state digest `181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd` and empty-optimizer digest `f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d` matched exactly.
- The update-1 pre-step witness matched exactly: `S=1.313827022910118`, `P_diagnostic=1.0`, `U=0.9792981296777725`, `R=1.0`, `O=1.026371382176876`.
- Accounting was exact: 1,000 contiguous trace rows, 16,000 presentations, 4,000 microbatch graphs/backward calls/predictor objectives, and 1,000 optimizer/EMA steps. Every traced loss and gradient was finite; encoder, lift/semantic, and predictor gradients were always positive.

## Absolute no-persistence control result

- The diagnostic no-persistence control's absolute status was `FAIL_FULL_ARM`, with exactly one failed check: free recall `0.839420 < 0.85`.
- This absolute result is separate from the registered treatment predicate. It does not qualify the control, replace V4, or establish that persistence caused the semantic difference.

| Semantic metric | No-P control | Gate | Result |
|---|---:|---:|---|
| Balanced accuracy | `0.852832` | `>= 0.80` | PASS |
| Free recall | `0.839420` | `>= 0.85` | FAIL |
| Occupied recall | `0.768565` | `>= 0.70` | PASS |
| Unknown recall | `0.950509` | `>= 0.90` | PASS |
| Rough-family occupied recall | `0.737168` | `>= 0.65` | PASS |

- Swept-progress utility `0.914823`, selected zero-prefix rate `0.025063`, and unequal-pair concordance `0.883208` all passed.
- Every utility, zero-prefix, and concordance check passed in all eight registered families.
- All four inference-control triplets passed: coordinate-matched persistence `+0.305929 / +0.238667 / 8-of-8`; shuffled action `+0.309817 / +0.245787 / 8-of-8`; wrong RGB `+0.108765 / +0.069776 / 7-of-8`; train action-mean prior `+0.078819 / +0.035278 / 6-of-8`, reported as mean delta / bootstrap lower bound / positive families.

## Registered persistence treatment result

- The registered comparison is full V4 minus the matched no-persistence control on the fixed eight development-family utilities.
- Equal-family mean delta: `-0.0028684618260227807`, failing the required strict `>0` check.
- Paired 10,000-draw bootstrap lower bound: `-0.017125088869090713`, failing the required strict `>0` check.
- Positive-family count: `3/8`, failing the required `>=6/8` check.
- All three registered positive-treatment checks are false; no positive persistence conclusion is allowed.

| Development family | Full V4 | No-P control | V4 minus control |
|---|---:|---:|---:|
| Large enclosed maze | `0.889619` | `0.898030` | `-0.008411` |
| Local composite motifs | `0.938405` | `0.973708` | `-0.035303` |
| Loop alias stress | `0.893863` | `0.900791` | `-0.006928` |
| Medium enclosed maze | `0.877259` | `0.899079` | `-0.021819` |
| Open obstacle field | `0.893483` | `0.880769` | `+0.012714` |
| Rough local dynamics | `0.943015` | `0.898872` | `+0.044143` |
| Small enclosed maze | `0.922340` | `0.932270` | `-0.009929` |
| Visual sensor stress | `0.922902` | `0.920316` | `+0.002586` |

## Interpretation and next authority

- Exact permitted interpretation: negative evidence for benefit from `P` under this fixed deterministic development schedule; no rerun or replacement is authorized.
- This one-sided test does not prove that `P` is harmful, useless in general, or unnecessary for JEPA; it does not establish equivalence, seed robustness, navigation, or generalization.
- V4's prior `PASS_FULL_ARM` remains unchanged. The control was explicitly diagnostic-only and cannot qualify, disqualify, replace, initialize, calibrate, or promote V4.
- The persistence-coefficient/control line is closed. Do not spend more runs tuning it.
- The lean candidate remains the full joint-JEPA V4 arm because it is the only absolute development pass. Moving it forward requires a new, narrow artifact/custody plan and then one untouched G2 perception qualification; this result itself does not authorize G2.
- No G2, navigation, held-out, sealed, production, deployment, promotion, or final-evaluation access occurred. Forbidden input count, G2/navigation/final-evaluation open count, and every forbidden semantic-loader counter were zero.
