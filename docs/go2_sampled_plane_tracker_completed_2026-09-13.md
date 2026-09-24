# Sampled plane tracker: full recorded journey result

The sampled plane extractor reproduced all 4,740 complete recorded visual
tracker outputs, frames 0–4739 of the completed stop-conditioned maze02 journey.
This includes 4,724 primary-camera selections, 15 auxiliary-camera selections,
the initial observation and all five accepted chained-anchor recoveries.
The final observation is the recorded return-arrival observation. There were
no tracker-output differences or unavailable-pose failures.

This comparison used the actual recorded visual evidence as its reference.
It ran one fresh tracker sequentially on the original paired RGB-D and body
observations. It did not load a high-level model, update a map, select commands
or create a simulator. The earlier 13-frame full-controller comparison remains
the available controller-level equivalence evidence; this result extends the
tracker evidence, not the entire controller comparison.

| Tracker-only wall time | Result |
| --- | ---: |
| Median | 90.212 ms |
| 95th percentile | 167.374 ms |
| 99th percentile | 254.683 ms |
| Maximum | 1,033.209 ms |
| Calls exceeding 100 ms | 1,615 / 4,740 (34.07%) |

Frames 0–3062 had an 88.342 ms median and 814 overruns. Frames 3063–4739,
after the outward-arrival observation, had a 97.734 ms median and 801 overruns.
Of the 1,615 overruns, 1,614 used the primary camera and one used the auxiliary
camera. The slowest call was primary-camera frame 990, outside startup and
without an accepted chained fallback. The measured maximum is retained.

This was a shared-host run alongside the supervised native audit, without a
profiler. Per-call timing excludes packet acquisition, mapping, planning,
command execution and receipt writing. No paired original-tracker timing was
rerun, so comparison with older timing runs cannot establish an exact speedup.
The complete diagnostic took 653.665 seconds, including input reconstruction
and output comparison; timed tracker calls totalled 461.078 seconds.

The current native collector calls `controller.observe` before advancing the
next `session.command_tick`; it explicitly pauses physics during computation.
The observer/controller compositions also expect uninterrupted 100 ms sensor
timestamps. Thus moving the existing whole controller into a background worker
does not itself supply fresh, timely control. Continuous execution still needs
an explicit policy for observation processing, command age and missed
deadlines, followed by a native experiment with physics advancing during work.
This result establishes that tracking alone still misses the desired deadline
frequently; it supplies no continuous-execution or hardware qualification.

The candidate remains outside the live and queued navigation runners.
The comparison completed in session 97297, PID 3190189 (creation time
1789282483.16), exit code 0. Source:
`scripts/compare_sampled_plane_recorded_tracker_development.py`.
Artifacts are in
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_sampled_plane_recorded_tracker_v1_attempt_001`.
`frames.jsonl` retains every duration, camera choice, fallback and equality
result. `result.json` SHA-256 is
`adc5e27923ebe25464fd3b57d373b7a21130c726b192881bf2c5202ca83e4910`.
The canonical compared tracker-output stream SHA-256 is
`148b9a10ad1a5880f7582284c4ef336de0bcd7e386f3b20de7b23558f7487fd2`.
