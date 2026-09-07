# Native-topology correction to articulated scan geometry analysis

V1 stopped before completing any trajectory: it assumed URDF `dont_collapse`
hints matched actual retained native links. Actual audited robot topology has
only base and twelve hip/thigh/calf links; fixed heads and feet are also merged.
The empty V1 FAIL, source and input bindings remain unchanged. A calf-link
contact cannot identify the colliding primitive: it can include a foot sphere.

V2 leaves instantaneous primitive transforms, support equations, all27 shapes,
all4,504 control frames, all24 source trials and all six contact trials unchanged.
It resolves each primitive to the nearest ancestor actually present in the
audited ROBOT link roster, traversing fixed joints only. Missing movable joints,
unknown environment names, missing base and unresolved chains are rejected.
This explicit adapter supports both collapsed and retained fixed-link rosters;
it does not assume every simulator follows the same URDF retention hints.

The nominal V1 geometry output is still a URDF-based description; its grouping
hints must not be used as native contact identity. V2 adds a separate exact
shape-to-native-link mapping. No body geometry, physical trajectory, sensor
measurement, controller, contact label or task result is rerun or changed.
Tests now cover feet merged into calf groups, merged head shapes, explicitly
retained feet, and rejection of missing movable/nonrobot roster entries. Validate
all24 actual source topologies before launching the corrected analysis.

All caveats and endpoints in the [original diagnostic protocol](go2_articulated_scan_geometry_development_v1_2026-09-05.md)
remain, except its incorrect assumption of separately retained feet. V1 is bound
as failure evidence, not used to claim qualified native grouping or clearance.

Exact fresh root: `.generated/go2_articulated_scan_geometry_development_v2_attempt_001`.
