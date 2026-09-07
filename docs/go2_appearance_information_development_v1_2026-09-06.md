# Fixed geometry-preserving appearance information V1

One fresh CPU/software-EGL render-only experiment, no robot execution or B retry.
Output `.generated/go2_appearance_information_development_v1_attempt_001`.
Use the exact B camera poses4.0–5.3s (indices25–38), giving13pairs per arm.
Camera transforms place the renderer only, never the motion estimator. Pair new
rendered RGB/depth with the original co-timed measured body/gyro histories under
new counterfactual episode identities. Original files/controller results remain
unchanged. This isolates appearance, not independent motion or maze success.

Three fixed arms: neutral, repeated checker, distinctive pseudorandom grayscale
cells. Appearance seed271828 is separate from physics/topology/start/goal; no
markers, location IDs, arrows or semantic codes. All surfaces use identical
0.125m-or-smaller quad tessellation with duplicated vertices per cell and two
triangles each. Neutralfloor128/walls89; checker50/205; distinctive40–230uint8,
independent surface sub-seeds. These are reproducible procedural visual surfaces,
not realistic texture/hardware validation. No image editing of old observations.

Keep collision-only plane atz0 and the four fixed native Box primitives with
their original dimensions/poses, friction1 and default solver parameters.
Render a separate32m-square visual-only floor atz0 and six exact visual-only
faces per box. Disable mesh alignment, decimation and convexification. Do not
enable the old texture branch or modify old native sources. The scene has no
Go2 because its renderer was already robot-hidden; no robot/gait/contact-dynamics
equivalence is claimed by this static assay. No scene.step call is allowed.

Read back native collision type/data/pose/material/solver values and verify
their equality across all arms. Read back visual-only roles and actual triangle
multisets against float32 exported meshes (1micrometre canonical grid), and
verify geometry equality across arms and no terminal change. There must be no
overlapping duplicate visual surfaces. Check actual camera/calibration and
single-sample render path. Per frame, compare all eligible stride8interior rays
to independently intersected physical plane/boxes:>1000rays, maxerror<=1mm.
Do not modify rendered depth to pass. Retain partial artifacts on any failure.

Use the frozen V1 RGB-D correspondence observer unchanged in every arm. Persist
sensor predictions before native motion scoring. Report all13pairs, acceptance/
rejection, feature/match/3Dsupport counts, actual translation error and observer
latency. Checker aliasing remains a negative challenge, not assumed impossible
under every camera motion. No parameter tuning, replacement attempt, threshold
relaxation or B success relabelling. Bind all explicit source, inherited input,
native and OpenCV identities before/after, and hash every generated artifact.

If this supplies previously missing information, proceed to a fresh sensor-fusion
and continuous whole-mission implementation, retaining textureless/repeated
appearance as robustness cases and matched supervised/JEPA/rollout/memory arms.
This render test is not the final scientific goal or a learned navigation policy.
