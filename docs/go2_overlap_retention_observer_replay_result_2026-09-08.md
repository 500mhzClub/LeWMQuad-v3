# Qualified-overlap retention survives the recorded anchor-loss trajectory

The new observer supplied all 147 current joint poses, with no terminal
failure, maximum XY error 2.852 mm and maximum rotation error 0.00280243 rad.
It passed the predeclared 20-mm/0.05-rad development thresholds. It retained 51
new anchors, including 49 half-feature-overlap promotions. These additional
compositions still have uncalibrated accumulated uncertainty.

The frozen original supplied 136 poses and failed at frame 136. Every one of
its 137 available evidence rows, including the terminal failure, reconstructed
exactly. Its maximum XY error was 5.035 mm and rotation error 0.00337297 rad.
The candidate preserved the matching/registration, reference selection,
increment/disagreement gates, eight-reference capacity and ten-frame bridge
limit. Bridge-only poses remained ineligible for promotion.

Fifteen focused observer tests passed in 1.62 seconds. The sequential original
and candidate arms had median active observer times 50.474 and 49.286 ms,
respectively, with one frame above 100 ms in each arm. This is observer-only
timing, not end-to-end control timing. Preflight recorded 82.31 GB available
RAM, 85.34 GB artifact storage free, 0.2% CPU utilization and idle GPUs.

The result authenticates 1,012 source files and the completed native trace's
full artifact/input bindings. Native pose was opened only for evaluation after
each observer replay. No new commands, physics or training occurred. The last
ten frames contain the original mission's stop commands, so survival there is
not an outcome of a changed controller. The previous native failure remains.

Artifacts in `go2_overlap_retention_observer_replay_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `107e308cb8a798020505b9d016088c5e1ae20549bafe25bd38d65f27daef69d4` |
| `full_direct_family_episode_039_original_frames.json` | `be991aa915a666953fe9d46cd4e499e0dbacd4241c4cc6476880751037809e7a` |
| `full_direct_family_episode_039_overlap_retention_frames.json` | `a7990a9e44a1719b84e06e4bbaa266cf006fdc699e5222aeff8ed09803483239` |
| `result.json` | `bf51ce8ecdde75dcc9ba6a4e82e56a9e96694046561af43a4f0008382b1762d8` |

The next step is one separately declared fresh native direct-039 case with this
observer and the existing retained-patch mission. It must recompute decisions
from new observations and pass fresh-model full raw command replay. Navigation,
physical backtracking, independent mazes, real-time execution and hardware
remain unestablished; the full goal remains active.
