# Auxiliary tilted-depth geometric characterization result

The fixed candidate camera places every one of the six complete 44-mm terminal
foot squares inside its frustum starting at recorded frame 17. Frames 17, 18
and 19 all precede execution of the first translating command at tick 19. The
original camera never obtained complete frustum coverage of the missing tiles.

The candidate has body-frame optical centre [0.35, 0, 0.08] m and downward pitch
30 degrees, with the existing 640-by-480 intrinsics and 0.2–5 m sensing range.
Its coordinate adapter preserves the complete optical-to-map transform while
allowing the existing floor-pixel helper to operate in a reference frame; that
reference frame is not a new robot-pose estimate.

This computation used recorded observed poses and the measured floor-height
hypothesis. It rendered no image and supplied no depth evidence. The foot
regions are retrospective diagnostic targets and must never be passed to an
online view-selection policy. Frustum containment alone does not resolve
robot self-occlusion, scene occlusion, pixel validity or physical support.
The result supports a bounded raw sensor characterization, not camera adoption
or navigation qualification.

Artifact root: `go2_auxiliary_tilted_depth_geometry_v1_attempt_001` under the
fixed navigation development artifact root.

- Launch: `2b13a578fb0d371b58cb521069b5ab9a21a4ddddb6ae1207fe78dcbe4d58aeb1`
- Projection: `06ddc091cca8ba283d8252bd46e95e39e336790138ae1955bca9f4ca2fdae2c8`
- Result: `8059e48ee6a7092fcaea51950191c2aa587aa208955889b1759b154d1be01265`

All 1,194 source bindings and predecessor input identities were verified.
Two tests passed in 0.15 s, covering rigid extrinsic composition, downward
orientation, near-region exclusion and the absence of unsupported coverage/
hardware claims. No model, optimizer, controller or native scene ran.

The following raw-capture candidate additionally enables robot visual meshes,
because the earlier renderer hides them. That capture must report whether
primary RGB pixels change; preserving primary extrinsics does not imply pixel
identity when robot visibility changes. Neither this calculation nor that
fixed-prefix sensor capture is a new closed-loop navigation result.
