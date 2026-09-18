# Accepted-reference refresh with coherent floor tracking

The completed yaw comparison exposed a raw-tracking failure on learned-yaw
layout 1 at frame 632. Exact replay matches all 632 published raw poses. Both
cameras still provide complete valid depth and good consecutive-frame fits,
but the eight retained anchors are from frames 609–616. Frames 617–621 still
use reference 616 without promoting newer imagery; frames 622–631 use the
measured-increment bridge. The bridge limit then stops tracking.

Test the existing 400 ms accepted-anchor reference-refresh rule composed with
the current coherent paired-floor tracker. This changes reference retention
only after an accepted anchored pose. It does not promote bridge-only poses,
relax matching/floor/continuity thresholds, integrate commands, or use native
pose. Source: `lewm/coherent_reference_refresh_development.py`.

Before any live follow-up, replay the complete failed learned-yaw layout-1
recording and the complete successful command-yaw layout-0 round trip. These
are fixed diagnostic inputs, not held-out qualification. Use their recorded
noisy packets and original gyro; load native pose only after estimation to
measure drift. Compare failure locations, accepted-frame counts, renewal
frames and position errors. Keep both original recordings and all results.

Seven existing reference-refresh and coherent-floor tests passed in 1.72 s.
The two replays use `scripts/replay_go2_gyro_coherent_floor_development.py`
with `--variant coherent_refresh`, writing separate directories under the
original roots. Sessions are 88254 (failed layout 1) and 47896 (successful
layout 0). No live controller assignment has yet been launched with this change.

The failed recording now accepts all 648 frames, with no tracking or floor
registration failure. The only explicit age-triggered refresh is frame 620;
621 original raw poses match. Position error median/max/final is
4.042/6.134/3.227 mm. The original maximum was also 6.134 mm. This demonstrates
continuity on the saved failed trajectory, not successful alternative navigation.
The complete successful round-trip replay remains pending.

Prepare one live follow-up on development layout 1 with learned yaw,
pose/command XY, disabled contact, the repaired frontier policy, and all
original frozen models, noise, commands and physical limits. Change only the
raw tracker's accepted-anchor reference refresh. Launcher:
`scripts/run_go2_coherent_reference_refresh_development.py`. Root:
`go2_coherent_reference_refresh_learned_yaw_noise_2mm_native_layout01_4800_v1_attempt_001`.
Run only after the complete round-trip replay has been assessed. This is a
separate exposed-maze development intervention, not another yaw-study assignment
or a replacement for its failure.

The complete round-trip replay accepts all 3,174 frames without failure. Fifteen
age-triggered refreshes occur; 2,071 raw poses match the original. Median/max/final
position error is 4.778/7.507/6.286 mm. The original median/max was 5.068/7.507 mm.
This fixed replay shows no maximum-error regression on that trajectory, while
remaining conditional on its recorded motion. Elapsed replay time was 359.32 s.
The short failure replay and long success replay are both complete, supporting
the prepared single live follow-up without further threshold changes.

The live follow-up launched in session 64871, owner PID 3623004, on the original
layout-1 CPU group (8–15,24–31). Its actual launch record confirms learned yaw,
pose/command XY, disabled contact, 400 ms accepted-anchor refresh, one planned
native assignment and measured-simulation timing. The native worker is live at
frame 0. Wait for owner exit including recording before independent physical
arrival, XY and yaw evaluation; retain a failure if one occurs. No navigation
outcome from this treatment is yet available.

## Verified live goal and return

The follow-up exited 0 after 384.31 s including archive, peak RSS 14,821,132 KiB,
zero swaps. All 2,566 camera pairs have registered poses. Independent evaluation
verifies goal frame 2069 and home frame 2564, with maximum native dwell distances
8.987/26.263 mm, maximum 100 ms speeds 0.019272/0.014080 m/s, and zero requested
motion throughout both one-second dwells. Zero disallowed contact samples.
Observed arrival uses the unchanged 20 mm threshold; physical verification uses
the unchanged 40 mm requirement. Final native home distance is 25.737 mm.
Position error median/max is 7.147/12.307 mm. Travelled path is 20.989 m.

There are 634 selected plans, 616 on time and 18 late. Maximum host/simulation
lag is 616.719 ms, and camera acquisition maximum is 78.032 ms. This remains
measured-simulation evidence. Eleven age-triggered refreshes are recorded at
frames 686, 932, 936, 956, 1285, 1289, 1629, 1928, 1957, 2503 and 2525.
The fresh asynchronous trajectory differs from the original; its refreshes need
not occur at the replay intervention's frame 620.

Its 611 executed windows have XY RMSE 6.117 mm, maximum 32.412 mm, with one
window above 30 mm. Applied learned-yaw endpoint RMSE is 3.343 degrees; command
yaw on the same windows is 1.853 degrees. These conditional errors do not alter
the verified physical result or establish a learning advantage.

The predecessor failure and the successful follow-up are summarized under
`go2_coherent_reference_refresh_layout01_summary_v1_attempt_001/result.json`.
All shared scientific settings and 138 common runtime source hashes match.
Both original failure and full follow-up recordings remain. This is one exposed
layout follow-up, not a repeated-seed or unseen-maze reliability result. It
supports reference refresh as a candidate perception improvement; further
prospective comparisons should keep its treatment explicit. No native owner
remains from this intervention; about 7.8 GiB artifact space remains.
