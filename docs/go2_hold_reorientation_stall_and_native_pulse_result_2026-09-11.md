# Why the executed turn recovery stalled

The completed original hold-reorientation run issued 1,429 holds, 153 left
turns and seven right turns from observations 405 through 1993. It issued no
translation in those 1,589 observations. All three translation actions passed
the sampled-surface filter but failed the original raw 800 ms nominal-path
check throughout that population. The raw eligibility sets were hold/left-turn
at 1,574 observations and left-turn at 15. These are the original raw filters;
special residual/reentry fallbacks have separate criteria and explain why an
executed action can differ from that raw eligibility set.

The intervention forced 138 left turns. Of these, 132 were followed by exactly
ten completed holds and an admitted following observation. Both observed poses
and the simulator's complete recorded physics show that the holds usually
reversed almost the entire heading change:

| Quantity | Observed-pose median | Native-physics median |
| --- | ---: | ---: |
| First 100 ms turn, all 138 interventions | 0.022045 rad | 0.022021 rad |
| Following ten holds, 132 qualifying cycles | -0.021472 rad | -0.021750 rad |
| Turn plus ten holds, those 132 cycles | 0.000584 rad | 0.000285 rad |

Medians describe their respective populations and are not additive. The model's
median predicted first-interval yaw was 0.035064 rad. Every intervention requested
0.45 rad/s; the native applied command was approximately 0.35 rad/s throughout
the 100 ms turn interval. This was the existing execution interface, not a change
introduced by this diagnosis. The median native net XY displacement per complete
cycle was 0.0000301 m. The model/command mismatch and observed settling are
diagnostic findings, not permission to replace measured motion with integration.

The repeated short-turn strategy therefore failed to accumulate useful physical
reorientation. Merely changing which contact horizon contributes to ranking
cannot make an action pass an unchanged failed nominal-path veto at these saved
states. A different prospective trajectory could change those states; its
outcome requires fresh execution. No sustained-turn outcome is inferred here.

`scripts/diagnose_go2_hold_reorientation_stall_v1.py` reconstructed the complete
2,005-row decision identity against the preceding authenticated diagnostic and
matched all 2,004 completed command receipts. It verified the relevant immutable
source and artifact bindings before and after reading. Its 1,963-path closure
and result completed in session 44185, exit zero:
`docs/go2_hold_reorientation_stall_diagnosis_2026-09-11.json`, SHA-256
`a8357d863f59d911b3c8fb535a2f1ceff3644afaf8ae84831a9e35de63259ac6`.

`scripts/verify_go2_hold_turn_pulse_native_motion_v1.py` then verified the
100,950-sample native pose/command trace, its 2 ms clock and quaternion convention,
and every original pulse/hold interval. It used native state only for offline
evaluation. Its 1,965-path closure and result completed in session 36047, exit zero:
`docs/go2_hold_turn_pulse_native_motion_2026-09-11.json`, SHA-256
`b63bbc610f176f849a86fcb620fca12d9320b84e89d9a20b5d53f42d4033c396`.
Neither checker reran tracking, model inference, the complete raw audit or full
training ancestry. The complete raw audit and prior all-artifact verification
remain authenticated predecessor evidence; the relevant current bytes were
rehashed for these diagnoses.

The next candidate is a bounded, observation-checked continuation of the chosen
turn. Its source and prospective protocol are separate from all frozen native
experiments. Navigation, real-time execution and hardware qualification remain
unproven.
