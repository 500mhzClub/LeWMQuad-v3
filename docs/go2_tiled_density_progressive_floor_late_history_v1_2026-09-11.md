# Tiled-density complete-controller comparison

Compare the completed ProgressiveBatchedFloorController with a fresh
TiledDensityProgressiveFloorController on the original 1,428-observation full-RGB
JEPA late-history prefix. The candidate changes only dense floor-cell triangle
arithmetic into 32-row blocks inside registration and mapping. Both use the
same progressive patch batching and original sparse floor kernel. Preserve all
camera, threshold, witness, ownership, model, mission and command semantics.

The recorded component probe at frames 0, 100, 400 and 800 compared both cameras
with original dense, density-routed and tiled-density kernels. Twelve repetitions
per camera balanced all six execution orders. All output arrays were byte-exact
and inputs unchanged. Auxiliary total-time reductions ranged from 20.56% to
57.45%; the unchanged primary path varied from a 1.98% regression to an apparent
17.22% reduction, demonstrating shared-host and cache noise. These measurements
do not establish whole-controller performance or real-time qualification.

Use the separately tested original-body harness in
`scripts/tiled_density_progressive_floor_replay_development.py`. Preserve all
1,428 ordered raw-input and command-tape checks; complete normalized decisions
and preceding progressive-candidate hashes; alternating execution order; 1,425
forecasts; independently stored original models; unchanged model digests and
gradients; and seven original retained memory/map/residual/history witnesses.
Normalize only the existing ten type paths and the new controller name/flag.
State witnesses do not separately cover every internal registration/map field.
Preserve the original strict sensing failure at frame 1173.

Before source preflight or execution, freeze the actual completed progressive
result and completion SHA-256 values in the launcher. Placeholder bindings are
explicitly rejected. Authenticate that predecessor's source and artifact hashes,
entire reconstructed report, every timing row, original raw/model admission and
negative sensing scope. Require its recorded owner ended on the original boot.
Recheck original inputs, predecessor evidence and all sources after replay.

Use one exclusive attempt, one full CPU replay at a time, at least 64 GiB
available RAM, 41 GiB artifact space and four physical CPUs. Keep single-thread
OpenCV/BLAS/Torch and deterministic original model execution. Preserve failure
without retry or resume. Timings cover complete controller.observe calls,
excluding input reconstruction, sensor acquisition and equivalence checking.
Rebuild all four original timing windows and all 100 ms deadline counts.

Run no physics or hardware, consume no observation 1428, and change no queued
navigation experiment. Full-controller equivalence and speed must be established
by actual replay and completion verification. Reliable round trips, independent
maze comparisons and real-time/hardware evidence remain separate requirements.

The completed progressive predecessor has result SHA-256
`bc8f387a3b93aa946b77aba7756bb62f25e51e1b48d7ac2cef5720d251628888`
and completion witness
`f40606e05fb6e4dd52188875ae98739631ef1a34e22ac3e8e32256bd7099e2b7`.
Its paired total-time reduction was only 0.8703355741565555%, while the early,
repeated-hold and late ten-frame windows had greater total time. All 1,425
navigation observations still exceeded 100 ms. Preserve these mixed timings;
this tiled comparison isolates the dense-kernel change and does not promote
progressive batching or establish a navigation benefit.
