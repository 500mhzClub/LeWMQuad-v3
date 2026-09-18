# Fixed 45-degree camera geometry reaches the diagnosed blind region

With the same mount, intrinsics and depth interval, changing only the auxiliary
camera's downward pitch from 30 to 45 degrees puts all twelve diagnosed full
44-mm front-left-foot squares inside its hypothetical frustum. The squares come
from all six rejected candidates at each of native JEPA ticks 62 and 99.

Earliest complete-frustum frames range from 13 to 15. All twelve are visible
geometrically by frame 15 and remain so through frame 17, the observation before
the first recorded translating command. This uses retrospective target squares
and preceding observed body poses. It is not measured floor coverage or a
prospective policy result; body/wall occlusion and native raster pixels are not
tested here.

Three focused geometry tests passed, covering the fixed optical axis/mount,
full optical-pose adapter and nearer-floor visibility while preserving an unseen
region. No camera sweep, rendered-image selection, model call or fitting was
performed. The next experiment captures actual primary and 45-degree auxiliary
images over the fixed 17-command zero/turn prefix, ending at observation 17.

Root: `go2_auxiliary_downward45_depth_geometry_v1_attempt_001` under the guarded
development base, with 1,271 bound sources.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | cf802955b689a132e0488309d945b3179ab9355c1e1ed7b2d24e01e41cbf7474 |
| projection.json | d75f92c839b3889fd1e718cab82321a8678a3c0f925e07df894560873d4ce124 |
| result.json | e0a61165564fd3acec667c58e525c33fd6ba626228e25eef0a51a501704b12c6 |
