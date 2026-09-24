# Fixed draw-order native repeatability and intervention bench

The completed union-wall bench failed exact RGB repeatability by 3 and 5 pixels.
All eight differences lie within 0.382 pixels of floor/wall edges. Its native
depth arrays match exactly. Frozen Genesis `Scene.sorted_mesh_nodes` starts from
a set and sorts by distance to node origin without a tie-breaker. Both union
visual nodes use world-space geometry and the same origin. Thus stable sorting
does not define their relative order. This is a source-supported hypothesis;
the previous run did not record actual draw order, so causation is not proved.

Run a distinct static intervention: unchanged union surfaces, colours, camera,
physical geometry and two recorded poses from the first l00 episode. Two
independent scene builds use explicit floor-first order; another two use explicit
walls-first order. Capture both poses in every build: 8 native RGB/depth pairs,
zero physics steps, CPU/software EGL, no gait, training or navigation.

The instance-local ordering helper accepts exactly two opaque visible primitives
with identity transforms and verified floor/wall extents. It supplies an explicit
order to that scene instance only and triggers native JIT list reconstruction.
Read back actual JIT node order and unchanged vertex hashes after every render.
No installed library, frozen source, geometry, material or recorded pixel is edited.

Both repeated builds within each prescribed order must have bit-exact full RGB
and native depth at each pose, with all original sampled physical visibility
checks passing the unchanged 1mm threshold. Report between-order differences
separately as the causal intervention result, not as a failed same-order pair.
No order is selected by its result; floor-first is the prospective implementation
choice only if this entire bench passes. Any failure is terminal and preserved;
no retry until lucky, pose substitution, tolerance change or automatic collection.

Output is the exclusive external navigation root child
`go2_ordered_union_rgb_repeatability_probe_v1_attempt_001`. Bind the preceding
bench launch/result/artifacts, recursive sources, native source including scene.py,
and tests before execution. Reserve 40GiB free; total budget 256MiB, serialized
launch metadata limited to one quarter before output creation. Save all meshes,
native identities, ordering witnesses, RGB, depth and score rows; verify hashes
and recompute visibility from persisted arrays at terminal.

Thirteen synthetic order tests plus twelve union geometry tests pass. They do
not establish native repeatability. This bench does not qualify mounted dynamic
capture, resolve the l00 foreground-silhouette depth failure, authorize another
layout collection or establish JEPA/memory/hardware/novel-maze success.
