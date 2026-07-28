# RGB Fixed-Teacher Dual-Domain Trajectory-H4 JEPA V1 Result

Date: 2026-07-28

## Terminal status

- Terminal decision: `STOP_MAIN_POOL_RGB_FIXED_TEACHER_DUAL_DOMAIN_TRAJECTORY_H4_JEPA_V1`.
- This was a scientifically complete STOP, not an execution failure.
- The authorized one-shot run completed 1,000 updates and exactly 16,000 training presentations. Validation used 10,240 presentations.
- The run consumed 2,453.897441713023 seconds of active GPU time.
- Update 750, after 12,000 training presentations, was selected by the preregistered minimum combined dual-domain score.
- The terminal decision evaluated 28 gates: 22 passed and 6 failed.
- The exact RGB fixed-teacher, 50/50 local-plus-cumulative, dual-domain trajectory-H4 JEPA V1 category is closed. It must not receive a retry, resume, extension, new seed, weight or threshold adjustment, additional data, checkpoint inspection or reuse, navigation run, or held-out evaluation.

This result is an intermediate perception/world-model qualification result. It does not establish a navigation policy, authorize navigation training, or evaluate held-out mazes.

## What the run tested

The probe tested whether one fixed-target RGB JEPA predictor could recover both of the useful but previously separated behaviours:

- the stronger visual prediction fit obtained from cumulative K4 supervision; and
- the stronger trained action sensitivity obtained from local trajectory supervision.

Its proper prediction score was an equal 50/50 mixture of local and cumulative objectives. The encoder target remained fixed, and the model retained the preregistered trajectory, action-ranking, and history-ranking terms. This was a deliberately bounded test of that exact mixed-score mechanism, not a general search over mixtures or coefficients.

## Selected aggregate result

| Measure | Selected value at update 750 |
|---|---:|
| Combined score | 0.778369087996 |
| Joint score | 0.774235408160 |
| H1 score | 0.816806544955 |
| H2 score | 0.829376353539 |
| H3 score | 0.820422848123 |
| H4 score | 0.852011558626 |
| H4 persistence gap | +0.147988441053 |
| H4 persistence bootstrap lower bound | +0.096762559734 |
| H4 cyclic-action gap | +0.148459141572 |
| H4 cyclic-action bootstrap lower bound | +0.130961170730 |
| H4 history gap | -0.000455686229 |
| H4 history bootstrap lower bound | -0.000610001459 |
| H4 all-hold gap | -0.026661057464 |
| H4 distribution value | 0.296850534032 |
| Combined distribution value | 0.266186228163 |
| Combined distribution bootstrap lower bound | 0.258484054417 |
| H4 spread | 1.189420243038 |
| H4 best-atom point error | 2.758235451809 |
| H4 centroid point error | 3.527136047322 |

The H1-H4 persistence gaps were +0.183193455793, +0.170623646294, +0.179577151577, and +0.147988441053. Seven of eight families were persistence-positive; `local_composite` was the only negative family at -0.000570404934, and the preregistered family floor still passed.

The H1-H4 cyclic-action gaps were +0.002560313847, +0.011597777643, +0.025392165236, and +0.148459141572. All eight families were positive, ranging from +0.090730009265 for `small` to +0.213749490142 for `rough`.

The H1-H4 history gaps were -0.000616238617, -0.000407228701, -0.000377290371, and -0.000455686229. No family had a positive selected history gap. The family values ranged from -0.0002073195 to -0.0008278205.

The H1-H4 all-hold gaps were -0.002624089768, -0.008856923043, -0.006547154299, and -0.026661057464. No family had a positive selected all-hold gap. The worst family was `small` at -0.039474751687.

The aggregate rank statistics remained healthy: target effective rank was 0.174980560939, online effective rank was 0.205727318923, both near-zero fractions were zero, and target rank and near-zero drift were zero. All eight families had positive combined distribution value. These observations rule out simple representation collapse or loss of distributional support as the immediate explanation for the failed causal controls.

## Per-family selected metrics

| Family | H4 score | Cyclic-action gap | All-hold gap | Persistence gap | History gap | Combined score |
|---|---:|---:|---:|---:|---:|---:|
| `large` | 0.7849020601 | +0.1250205267 | -0.0193487988 | +0.2150979401 | -0.0006142623 | 0.7566866536 |
| `local_composite` | 1.0005704004 | +0.1655779876 | -0.0287048788 | -0.0005704049 | -0.0005109402 | 0.8394401880 |
| `loop` | 0.8628740263 | +0.1129440460 | -0.0314124594 | +0.1371259699 | -0.0008278205 | 0.7586510764 |
| `medium` | 0.8565796051 | +0.1246412583 | -0.0217666338 | +0.1434204029 | -0.0002924859 | 0.7483296958 |
| `open` | 0.7796073635 | +0.2077322217 | -0.0224647737 | +0.2203926368 | -0.0002955944 | 0.7517169939 |
| `rough` | 0.8605919756 | +0.2137494901 | -0.0269732831 | +0.1394080242 | -0.0002073195 | 0.8003056433 |
| `small` | 0.8710586439 | +0.0907300093 | -0.0394747517 | +0.1289413534 | -0.0004623513 | 0.7951553758 |
| `visual` | 0.7999083941 | +0.1472775929 | -0.0231428804 | +0.2000916059 | -0.0004347156 | 0.7766670772 |

## Learning trajectory

| Update | Presentations | Combined | H4 | H4 persistence gap | H4 cyclic-action gap | H4 history gap | H4 all-hold gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.0000000000 | 1.0000000000 | approximately 0 | 0 | 0 | 0 |
| 250 | 4,000 | 0.8234891566 | 0.9132583707 | +0.0867416305 | +0.1854876399 | -0.0017418261 | -0.0230745731 |
| 500 | 8,000 | 0.8332389416 | 0.9354643244 | +0.0645356758 | +0.1676007049 | -0.0006964541 | -0.0195381974 |
| 750 | 12,000 | 0.7783690880 | 0.8520115586 | +0.1479884411 | +0.1484591416 | -0.0004556862 | -0.0266610575 |
| 1,000 | 16,000 | 0.7951414021 | 0.8720206307 | +0.1279793705 | +0.1401780790 | -0.0004045880 | -0.0164016777 |

Update 750 was therefore the correct preregistered selection. Fit and cyclic-action separation both regressed from 750 to 1,000. Aggregate history remained negative at every trained evaluation, and all-hold remained negative across every family. At update 1,000 only one family had a barely positive history gap while the aggregate remained negative. The completed curve supplies no scientific justification for extending or resuming this run.

## Training diagnostics

| Loss or diagnostic | Mean | Last update |
|---|---:|---:|
| Half local proper loss | 0.256144549444 | 0.277375876904 |
| Half cumulative proper loss | 0.254396984190 | 0.256457775831 |
| History alignment loss | 0.006615510036 | 0.005365388002 |
| Cyclic ranking loss | 0.030307456333 | 0.061602521688 |
| History ranking loss | 0.030448430881 | 0.030263695866 |
| Total loss | 0.577912931710 | 0.631065309048 |
| Centroid absolute future-error diagnostic | 0.522863760948 | 0.544586896896 |

The diagnostic future-error term had weight zero. The local and cumulative fit components were balanced, so one domain did not simply dominate the mixed proper score. The history-ranking loss remaining at approximately the hinge scale is consistent with the near-zero, non-positive history separation observed at validation.

## Gate failures

The six failed gates were:

1. `h4_history_gap_at_least_point03`
2. `h4_history_gap_bootstrap_lower_positive`
3. `h4_hold_gap_positive`
4. `history_positive_in_six_families`
5. `hold_positive_in_six_families`
6. `no_family_hold_gap_below_minus_point02`

The run passed its remaining 22 gates, including proper-fit, distribution-support, persistence, and trained cyclic-action requirements. Passing those gates does not compensate for the failed history and reserved all-hold controls: both are required to claim the desired compositional, action-conditioned latent dynamics.

## Comparison with the immediate predecessors

| Selected result | Combined | H4 | H4 persistence | H4 cyclic action | H4 history | H4 all-hold |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative K4, update 750 | 0.725748 | 0.742969 | +0.257031 | +0.010815 | -0.013974 | -0.006572 |
| Local-only, update 1,000 | 0.852201 | 0.953251 | +0.046749 | +0.187365 | +0.000948 | +0.001280 |
| Dual-domain V1, update 750 | 0.778369 | 0.852012 | +0.147988 | +0.148459 | -0.000456 | -0.026661 |

Relative to local-only, dual-domain V1 improved combined score by 0.073832 and H4 score by 0.101239 while retaining strong trained cyclic-action sensitivity, reduced by 0.038906. It recovered approximately 58% of the local-to-cumulative combined-score gap and approximately 48% of the H4 gap.

Relative to cumulative K4, dual-domain V1 was worse on combined score by 0.052621 and on H4 by 0.109043. It was much stronger on the trained cyclic-action control and moved history from clearly negative to approximately zero, but it made the reserved all-hold result substantially worse.

The comparison supports a narrow conclusion: changing the local-only proper score to a balanced local-plus-cumulative score recovered substantial prediction fit without destroying the trained cyclic-action signal. It does not show that the local target or a particular ranking term alone caused either predecessor's behaviour. It also does not show general action understanding: cyclic corruption was trained, whereas the reserved all-hold control failed in every family. Other unseen action corruptions were not tested.

## Scientific conclusion

The exact dual-domain mechanism achieved a useful compromise between prediction fit and sensitivity to its trained cyclic-action corruption. It did not learn a usable factual history representation, and its action sensitivity did not transfer to the reserved all-hold intervention. Point forecasts also remained too inaccurate for downstream navigation despite healthy distributional support.

The evidence therefore closes coefficient-level refinement of this category. More mixed-score weights, margin changes, seeds, updates, data, or checkpoint selection would test nearby tuning rather than repair the missing mechanism.

## Recommended next category: shared recurrent factual transition

The next justified mechanism is a fresh factual shared-transition causal-belief K4 JEPA:

- Replace horizon-specific action-prefix decoding and trained cyclic, reverse, or reset corruption hinges with one shared one-step action-conditioned latent transition.
- Apply that same transition across the factual observed sequence `e0, p0, e1, p1, e2` to construct the current latent belief.
- Recurrently apply the same transition over the factual future action sequence `p2:p5` to roll out K4 future latent atoms.
- Retain the fixed N=320 RGB teacher targets and the supported 50/50 local-plus-cumulative proper-fit objective.
- Train only from factual RGB/action pairs. Reserve cyclic, all-hold, reordered, and reset action controls exclusively for validation so that passing them measures transfer rather than memorization of a training corruption.
- Do not initialize from, inspect, or reuse any checkpoint from this stopped category.
- Do not add navigation labels, pose, depth, geometry, privileged state, or held-out maze access.

This is one coherent architectural change: it tests whether tying belief construction and future rollout to the same factual one-step dynamics produces compositional action and history use. If its reserved controls fail, that new category should also close rather than accumulating extra margins or special-case losses.

This recommendation is not execution authority. It requires a separate preregistration, source review, frozen source identity, custody review, and explicit execution authorization.

## Integrity and custody audit

- Frozen source-and-review commit: `53ffe6fc8ac188045c10c5ed726f66f9c15e8f52`.
- Execution-authorization commit: `1892f4b5735988c79071362acb8f506061160931`.
- Source review receipt SHA-256: `99d4c2349b92d2cd4eeaafe1012996bfe6cb5876b1e85622e9e64a3dc97beb7e` (9,373 bytes).
- The focused and parent synthetic suite reported 57 passing tests in 5.85 seconds before execution.
- The fixed target identity was `dd3c8f...` both initially and finally; target EMA updates were zero.
- Training-input identity began `f3f4...`; validation-input identity began `86ab...`.
- Recorded access totals were 16,000 training examples, 10,240 validation examples, 183,680 successful RGB opens from 183,680 attempts, and 6,900,398,764 RGB bytes.
- Auxiliary training access was exactly 32,000 history records and 16,000 wrong-action records. The model-source files and N=320 source were each opened and rechecked exactly once.
- The reservation's 112,000 `rgb_frame_views` cap is explicitly training-only. The 183,680 access total includes the preregistered validation views and is not a cap overrun.
- The top-level `wrong_action_training: false` field records the disabled inherited built-in centroid diagnostic. The nested additional-science and artifact fields record the authorized dual-domain controls as active. This is an intentional schema distinction, not a scientific mismatch.
- All forbidden access counters were zero, including sealed, test, held-out, label, arbitrary-checkpoint, and retry access.
- The artifact receipt inventories checkpoint filenames as metadata only. During this result audit, no checkpoint file was opened, hashed, statted, listed, loaded, copied, or reused.
- The legacy V4 sealed benchmark remains unopened, development-only, and permanently ineligible for final evaluation.
- No navigation, held-out, production, promotion, deployment, retry, resume, or replacement attempt occurred or is authorized by this result.

## Completion receipt bindings

The completion receipt binds the five prerequisite receipts and itself as follows:

| Receipt | SHA-256 | Bytes |
|---|---|---:|
| Reservation | `97bd31897cdea1b1d240925c6af2e07fbb0f847ee3f325d42dadb07cf4b9284f` | 6,052 |
| Metrics | `140616fc95fa887c3b52b3213247a823d7b4b9115cdee033a449ac6f65ae3ead` | 54,047 |
| Artifact | `e863231bff869a7f3c897e5acfecdeaf4088a241be922649c76a09164bbff71f` | 5,894 |
| Access audit | `14fcb06f2b00ce9b78b0563952943787e21f84b2927f21a251f169e009a02d24` | 1,279 |
| Result | `bfab25691e442a11e0f2a8bbd460afcb9f71b4a4540681df3257c44cf6adc93f` | 2,514 |
| Completion | `c9f900083091aae8f35897453c011abc183366988b1ca113d7265ea6e9e538f2` | 1,846 |

Canonical content hashes were independently recomputed successfully for all six receipts: reservation `0ca1f3...`, metrics `36854c...`, artifact `be9bf6...`, access audit `3a6aed...`, result `953db0...`, and completion `872fd5...`.
