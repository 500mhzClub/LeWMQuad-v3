# Complete recorded tracker comparison

The original and tiled-plane visual trackers produced equal complete public
results at all **4,740 observations**, frames 0–4739, through the collected
outbound and return arrivals. This includes 4,724 primary-camera selections,
15 auxiliary-camera selections, the initial paired reference, and five frames
using the chained-reference fallback. Neither tracker failed or diverged.

| Tracking computation across the complete sequence | Original | Candidate |
| --- | ---: | ---: |
| Total seconds | 641.816747 | 543.267671 |
| Median milliseconds | 130.215 | 110.380 |

The measured total reduction is **15.3547%**. The comparison alternated execution
order and used no profiler. It ran on a shared host alongside the native trial's
raw audit. Its 1,297.53-second wall time includes packet reconstruction, input
file reads and identity checks, result comparison and logging outside the
tracker timers. The median candidate tracker call alone still exceeds 100 ms.

The result supports reusing the existing optimized floor kernel inside visual
tracking on this recorded history. It does not establish full-controller
equivalence across that history: map, planner and learned model were absent.
The separate short comparison matched all 13 complete controller decisions and
measured a 12.3499% reduction across its ten active decisions. These percentages
describe different scopes and must not be added.

No running or queued navigation controller was changed. The independent-layout
JEPA/reactive pair retains its existing implementation. A future adoption must
be identified as such, and neither timing result establishes continuous
execution, a new navigation outcome or hardware qualification.

The process completed successfully in session 5898; owner PID 3142817 ended.
There is no failure, difference or matched-failure artifact. The 4,740 ordered
timing rows were read once after completion; their equality flags, frame
population and summed timings agree with the result.

Artifact root under the existing development artifact directory:
`go2_tiled_plane_tracker_recorded_comparison_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| `launch.json` | `f4ee868a2f030dd40145bf657d93c7f337f60e92673c0ca817d8c06a4a7c6bcd` |
| `frames.jsonl` | `a89cf5b1c1ffc6abcb9be79053a196b0301da200ad64a659e8a5b2cf5ffff15d` |
| `result.json` | `b750350bb732a64d9efada39f498a0f0c0de6771a776f50aab1ff0b305fa03d4` |

The complete raw navigation audit of the stopping-rule trial remains pending.
Its preliminary physical round trip passes, but it is a reused development
layout and does not establish independent-maze reliability.
