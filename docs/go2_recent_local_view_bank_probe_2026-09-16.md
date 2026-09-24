# Retained JEPA tracking-failure investigation

The completed four-layout comparison has a JEPA tracking failure on layout 3.
Keep its native failure and full sensor recording. The first bounded check
tested availability for a failure-triggered old-view lookup, without changing
tracking. All 1,105 recorded poses reproduced exactly; failure remained frame
1105. No bank entry met the existing 0.20-m / 0.20-rad lookup criteria.

The eight stored views came from frames 345–460, 64.5–76 seconds before the
failure. The closest-facing view, frame 443, was 0.225 m away and 0.311 rad
different; neither existing cutoff was met. A second replay tested the normal
descriptor-based candidate fit for all eight views against both cameras,
bypassing only lookup eligibility for diagnosis. All sixteen failed with
insufficient rigid-pose matches. No candidate supplied a pose; all 1,105
recorded poses reproduced and the full original terminal-failure evidence
was equal between the two probes. Direct-flow or alternative feature matching
against those old views was not tested. These results do not support adding
the originally proposed lookup-only fallback.

Artifacts under the navigation root:
`go2_persistent_visual_learning_jepa03_failure_view_probe_v1_attempt_001`
and `go2_persistent_visual_learning_jepa03_stored_view_fits_probe_v1_attempt_001`.
Each contains the exact replay source and result. The second source revision
now also restores rejected-plane diagnostic rows after candidate-only probes;
the completed sixteen candidates produced no such changes, as confirmed by
the complete equality of the saved original failure evidence.

Next bounded hypothesis: retain the newest accepted reference in each of the
existing eight heading bins, instead of the first reference until translation
exceeds 0.30 m. Preserve the four-frame revisit cadence, all reference and
continuity checks, the 0.20-m / 0.20-rad lookup criteria, and the same sensor
packets. This changes reference selection, not measurement acceptance. It
may discard a useful older view; a successful replay is not assumed.

Run one replay on the same 1,109-packet failed recording with
`RecentLocalViewBankMotion`. Record the actual view-bank frames, selected
references and complete outcome, then evaluate pose error separately if the
variant continues. No native state enters tracking. This is an exposed
development recording and does not establish recovered navigation, timing
qualification or a general benefit. Any justified variant still needs a
prospective closed-loop trial. Broader environment tests remain deferred.

## Newest-view replay result

The fixed newest-view variant completed its replay with the same terminal loss
at frame 1105. It produced 1,105 poses. The first pose/reference difference
from the original was frame 252, with 853 differing pose records and maximum
position difference 8.830 mm. There were 32 old-view attempts and six selected
revisits, so this was an exercised treatment. The final bank contained frames
496/1095/1104/965/423/403/386/369; the failed frame had no scheduled old-view
attempt. Recent-reference fitting still failed, including an auxiliary
gyro-consensus failure against frame 1104.

Result and comparison:
`go2_persistent_visual_learning_jepa03_recent_view_bank_replay_v1_attempt_001`.
The variant is not adopted for native navigation. The replay does not show
that every possible memory policy fails, nor does it establish what a changed
controller would do online. It rules out this single newest-per-bin change as
a repair of the retained recording. Preserve its source and negative outcome.
Next investigate the observed feature/visibility decline and the executed
view-recovery commands before loss. Keep the original full sensor failure
recording; no new environment-type or native navigation trial was launched.

The original primary and auxiliary RGB images at frame 1105 were visually
inspected. Both are dominated by nearby, broad, low-texture wall patches with
few distinct corners; the auxiliary view retains a small brighter region near
the bottom. This supports examining visibility and correspondence support,
but does not establish that a feature threshold or different action would
have prevented the failure. The original images remain in the failure root as
`native/rgb_1105.png` and `native/auxiliary_rgb_1105.png`.

Follow-up: [recovery timing and two prospective missions](go2_visual_recovery_dispatch_hold_2026-09-16.md)
showed that the original recovery turn arrived after the failed image capture.
An experimental prompt hold run succeeded but never triggered the intervention;
an original-controller repeat also succeeded. Both passed physical arrival and
contact checks. Early timing-related command divergence prevents attributing
either success to a tracking repair; the original failure remains unresolved.
