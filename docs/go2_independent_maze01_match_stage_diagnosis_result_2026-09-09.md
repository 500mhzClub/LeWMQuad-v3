# Maze1 correspondence losses by matching stage

The fixed-frame diagnosis completed with all64returned correspondence arrays
byte-for-byte identical to the frozen matcher across16camera/reference pairs.
It changed no matching gate, pose, command or original outcome.

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_maze01_match_stage_diagnosis_v1_attempt_001`.
Result SHA-256:
`79295cc24a0c4fa1f20794bd5f1c01c9be1f15de0699197da44d5e455ff82424`.
Launch SHA-256:
`fd93e739567139fe4cf7bec25e43a5d794aabe89c84bfa396433a7c7506d5462`.
Predecessor diagnosis:
`15ec380fc03906fef61dbd5c9a8418856c63bdf531337152db7d8ad0ac685a56`.

Session63231closedexit0. All1,661source bindings,1,386original case input
bindings and the predecessor result/launch identities were verified before
and after execution. The measured diagnosis work took0.8026564470492303s,
excluding preceding admission. It loaded no model and ran no native scene.

## Latest accepted frame213 to failed frame214

Counts after each sequential gate:

| Gate | Primary | Auxiliary |
| --- | --- | --- |
| Reference selected features | 56 | 76 |
| Current selected features | 62 | 89 |
| Forward descriptor-ratio matches | 23 | 43 |
| Backward descriptor-ratio matches | 22 | 36 |
| Mutual descriptor pairs | 18 | 31 |
| Distinct pixel locations | 18 | 31 |
| Forward and reverse flow status | 17 | 29 |
| Finite flow coordinates | 17 | 29 |
| Forward/backward agreement within0.5px | 13 | 25 |
| Flow endpoint within1px of matched descriptor keypoint | 10 | 12 |
| Both depth endpoints liftable | 10 | 12 |

The descriptor-location agreement gate removes13of25remaining auxiliary
tracks and3of13primary tracks for this latest pair. Depth lifting removes
none. Across all16pairs, depth lifting likewise removes none of the tracks
that reached it. The prior rigid-fit diagnosis then finds only11inliers in
the best auxiliary proposal against the unchanged12minimum.

Older anchors do not supply a qualifying replacement. Mutual-pair counts for
primary references206–212 are5,9,13,14,13,15,11; final lifted counts are
2,4,5,6,4,9,8. Auxiliary mutual counts are4,2,2,3,2,5,17; final lifted
counts are0,0,1,1,2,3,8. Exact stage populations for every pair are retained
in the result. Forward and backward ratio populations are separate directions,
not a sequential population; monotonicity is checked from mutual pairs onward.

This identifies where accepted support is lost, not whether rejected matches
are physically correct. More optical-flow tracks would not by themselves
constitute a valid rigid pose. A different short-interval association method
must be declared prospectively and checked through original geometric and
continuity gates, causal replay and fresh physical evidence. Do not merely
remove the1px gate or lower the minimum inlier count to accept frame214.

## Implementation and resources

- `lewm/rgbd_match_stage_diagnostic_development.py`, SHA-256
  `7ebae5392a7e23dda617f241eb0ee2f31b6e4611ebfbb2fca9eb73ef305842f6`.
- `scripts/diagnose_go2_independent_maze01_match_stages_v1.py`, SHA-256
  `2bc38df0bca6ba84d999ca09bf71ef32869a7899dde7bfd777d73692f3c99b69`.

These sources are now frozen by the executed diagnostic launch. Runtime
verification compared all four arrays per pair with the original matcher,
including dtype, shape and bytes; checked nonincreasing sequential gate
counts; and matched all16final correspondence counts to the completed prior
diagnosis. This directly checks the actual rejected inputs without adding
tests that only repeat the instrumentation's arithmetic.

The launch measured79,290,425,344bytes availableRAM,93,244,870,656artifact
free bytes,21,360,078,848workspace free bytes,32logical CPUs/16physical cores,
CPU3.3%,bothGPUs0%. It ran one OpenCV/BLAS thread alongside the single maze2
native worker. Original collector/controller sources and parameters are unchanged.
