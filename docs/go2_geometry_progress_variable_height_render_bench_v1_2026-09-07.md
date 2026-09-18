# Variable-height union rendering regression bench V1

The ordered-near-field pilot failed before scene allocation because the existing
union builder requires equal wall heights. Preserve that terminal attempt. The
new visual-only provider uses exact micrometre-grid3D coordinate compression to
emit only the boundary of the original grounded, axis-aligned boxes. Physical
boxes,0.7m panel and1.4m room walls are unchanged. Uniform-height outputs retain
the existing exact face contract. No runtime monkeypatch or predecessor edits.

Before another physical collection, one fresh static native bench at
`go2_geometry_progress_variable_height_render_bench_v1_attempt_001` will render
the four original geometry/appearance strata. Use the original forward episode's
authenticated camera poses at frames0,15,16,17,18 (20captures total), including
the zero-ray and clipped-opaque-wall cases. CPU, one scene at a time, no physics
steps, no gait execution, no training and no navigation qualification.

Use the existing5mm camera, union-wall renderer, floor-first order and core-profile
precision readback. Save specifications, native geometry identities, RGB, raw
native depth, optical poses, source-pose witnesses and per-frame raster evidence.
Check exact clock/pose constancy during rendering and the unchanged physical
first-surface/raster-footprint metric/occlusion criteria. Require all20frames to
pass stable-interior1mm measurement with at least1,000rays, no near occlusion,
and no falsely valid near-surface rays. Preserve strict boundary scores too.

Freeze new source, tests, this protocol, old collection camera bindings and the
failed constructor launch/terminal identities before native rendering. One bench,
no retry/resume or outcome-driven threshold adjustment.256MiB output budget and
40GiB free reserve. A passed render bench supports preparation of a separately
frozen full24episode successor; it does not substitute for that collection or
authorize training, prediction benefit, navigation or hardware claims.
