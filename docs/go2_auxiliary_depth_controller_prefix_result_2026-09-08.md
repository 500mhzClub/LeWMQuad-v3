# Auxiliary depth controller prefix integration result

Both fixed seed-2026091001 corrected full-JEPA and full-direct controllers
admitted all 20 observations from the robot-visible paired-camera prefix.
Each reproduced all decisions exactly in two fresh runs, with unchanged model
state. Auxiliary metric depth feeds observed geometry and floor evidence only;
the learned predictor and its training-only correction remain unchanged.

For both models, the first proposed command difference from the recorded
primary-only controller is frame 17. The common causal prefix therefore contains
18 observations, including the observation at which commands diverge. The new
commands were not executed. Later rows are shadow decisions on the original
trajectory; the final observation's proposal was also not executed.

At frame 19, each auxiliary memory accounts for all 382,521 retained returns:
216,221 floor and 166,300 other/unknown. The current frame contributes 19,200
sampled returns, including 15,482 floor and 3,718 other/unknown. The memory holds
20 auxiliary floor patches and 10,126 auxiliary sample-bound voxels. Full-foot
coverage does not exempt other/unknown evidence from collision checks.

Exclusive root: `go2_auxiliary_depth_controller_prefix_v1_attempt_001` under
the guarded navigation development artifact base. Processing wall time was
54.7414023661986 seconds after admission.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 2632ea423187a9ff6860f54f7ef1c458e56ba78de92fea93cb7ea4a22c5438db |
| JEPA decisions | 08263e9daad4eafe67a9bee10a0c4421f4984a4542d07c59d91bd41b83b80acd |
| direct decisions | 24173c5c0b1d15bb730c1a27fd799b7505889775a403b8d55116ccee77e7554e |
| result.json | f80574fa6a4b4fc71898f07abafad1ad795e20a4005533bc09fbd2c95a756312 |

This establishes controller integration on a recorded prefix. It establishes
no native arrival, independent-maze performance, realistic latency, backtracking
or advantage over matched reactive/nonpredictive baselines. The next experiment
executes the new decisions prospectively using the original native goal auditor.
