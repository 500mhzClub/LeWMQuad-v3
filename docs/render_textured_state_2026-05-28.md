# Textured Render Current State - 2026-05-28

Snapshot time: `2026-05-28T21:15:46+01:00`

## Active Job

There is an active tmux session:

```text
datagen_render_textured: 1 windows (created Thu May 28 14:04:57 2026)
```

The session was started as:

```text
CORPUS=.generated/scene_corpus/minimum_tex_20260520T211541Z
ROLLOUT_ROOT=.generated/datagen_full/rollout
OUT=.generated/datagen_full/render_textured
RENDER_VENV=.generated/venvs/genesis_render_vulkan
RENDER_BACKEND=vulkan
REPLAY_ENV_MODE=single
EGL_DEVICE_ID=1
PYOPENGL_PLATFORM=egl
WORKERS=8
CORES_PER_WORKER=4
CAMERA_HZ=10
scripts/datagen_render_resumable.sh
```

The live script instance is an older in-memory copy of
`scripts/datagen_render_resumable.sh`. It honors the Vulkan/textured env vars,
but it only treats `.render_done` as success. If the renderer writes a complete
`summary.json` and then exits nonzero during teardown, the old wrapper logs the
scene as `[FAIL]` and never creates `.render_done`.

## Output State

Output root:

```text
.generated/datagen_full/render_textured
```

Observed counts after marker backfill:

```text
scene plans:        1450
scene dirs:         24
summary.json files: 16
.render_done files: 16
png files:          1000279
disk usage:         47G
```

All completed summaries observed at this point are for `large_enclosed_maze`
test-hard scenes. The active partial scenes without `summary.json` were:

```text
large_enclosed_maze_3c3e2de610e7
large_enclosed_maze_460ca2ac5540
large_enclosed_maze_4be28cc521b3
large_enclosed_maze_59f643f7e5b9
large_enclosed_maze_5df76dc25ce2
large_enclosed_maze_64fe7e056f56
large_enclosed_maze_6eeb4ec1ec8c
large_enclosed_maze_7cf3c4775fd3
```

These eight scenes do not have frame-level resume. If the old job is stopped
before they complete, they will be rerun from the beginning.

## Memory State

System RAM at snapshot:

```text
Mem: 91Gi total, 81Gi used, 5.2Gi free, 10Gi available
Swap: 19Gi total, 1.4Mi used
```

ROCm VRAM at snapshot:

```text
GPU[0] total: 34208743424 B
GPU[0] used:  33046310912 B
GPU[1] total: 2147483648 B
GPU[1] used:  24137728 B
```

`rocm-smi --showpids` reported the eight active render PIDs, each with a small
KFD attribution, while total VRAM stayed near full. That points at driver,
renderer, or graphics-runtime allocations that are not well represented by
per-process KFD accounting.

The user's observation that VRAM rose to roughly 88% and then crept upward all
day is consistent with the current workload shape:

- 8 concurrent Vulkan render processes.
- Large scenes with 48,000 frames each.
- Single-env replay mode, so each worker can run for hours before process exit.
- No frame-level checkpointing, so killing a worker loses the current scene.

The new resume patch reduces data loss on restart, but it does not by itself
fix any within-process renderer/driver memory accumulation. Lower worker count
is still the safer operating mode.

## Fix Implemented

New helper:

```text
scripts/render_resume_markers.py
```

The helper validates that a scene is complete before writing `.render_done`:

- `summary.json` exists.
- `render_status == "complete"`.
- `scene_id` matches the job.
- `summary["plan"]` matches the job plan when present.
- `summary["frame_count"]` matches the plan `frame_count`, optionally capped by
  `RENDER_MAX_FRAMES`.

Patched wrappers:

```text
scripts/datagen_render_resumable.sh
scripts/datagen_render_resumable_v03.sh
```

New behavior:

- Existing `.render_done` markers are revalidated against the current
  `summary.json`, plan, and `RENDER_MAX_FRAMES` setting before skipping.
- Existing complete `summary.json` is backfilled to `.render_done` and skipped
  before any `rm -rf`.
- If the renderer exits `0`, the wrapper validates the output before marking
  done.
- If the renderer exits nonzero but the summary validates as complete, the
  wrapper now records `[done-after-nonzero]` and writes `.render_done`.
- Partial scenes are still cleared and rerun.

The non-v03 wrapper also now honors these restart env vars again:

```text
RENDER_BACKEND
REPLAY_ENV_MODE
RENDER_VENV
EGL_DEVICE_ID
PYOPENGL_PLATFORM
```

That matters because the active textured job was launched with
`RENDER_BACKEND=vulkan` and `REPLAY_ENV_MODE=single`.

## Textured-v03 Hybrid Implemented

The faster v03 renderer now supports a lightweight textured mode:

```text
scripts/render_replay_v03.py --textures
```

This path reuses:

- the v03 self-contained 0.3.14 renderer
- single-env egocentric replay from recorded `camera_pose_world`
- RGB-only rendering
- no depth render
- no per-frame quality validation

It adds:

- deterministic CC0 texture selection from `lewm_genesis/textures.py`
- UV-mapped cached box meshes for textured walls and obstacles
- textured floor surfaces
- `summary["visuals"] = "textured_v03"`
- `summary["textures_enabled"] = true`

The v03 resumable driver exposes this through:

```text
RENDER_TEXTURES=1 scripts/datagen_render_resumable_v03.sh
```

This is the intended replacement for the current heavy
`render_replay_genesis.py --textures --depth-validate-only` full-corpus path.
It preserves the important visual-domain shift while avoiding the depth and
validation work that made the current job run at roughly 8-9 FPS per worker.

Recommended new output root for the first full attempt:

```text
.generated/datagen_full/render_textured_v03
```

Do not mix this into `.generated/datagen_full/render_textured` until the capped
benchmark has confirmed acceptable throughput and visual quality.

### Textured-v03 Smoke Result

After stopping the old heavy textured job, a capped textured-v03 render smoke
was run on a `large_enclosed_maze` scene:

```bash
HOME=$PWD/.generated/render_home \
XDG_CACHE_HOME=$PWD/.generated/cache \
MPLCONFIGDIR=$PWD/.generated/mplconfig \
EGL_DEVICE_ID=1 \
PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/render_replay_v03.py \
  --plan .generated/datagen_full/rollout/test_hard/large_enclosed_maze/chunk_0000/plan/000023_large_enclosed_maze_7cf3c4775fd3/render_replay_plan.json \
  --scene-corpus .generated/scene_corpus/minimum_tex_20260520T211541Z \
  --out .generated/datagen_full/render_textured_v03_smoke_1780002066 \
  --resolution 224 \
  --max-frames 200 \
  --textures
```

Result:

```text
RENDER_OK large_enclosed_maze_7cf3c4775fd3 frames=200 fps=77.7
summary render_status: complete
summary visuals: textured_v03
summary textures_enabled: true
png files: 200
output size: 8.2M
resume-marker validation: ok
post-run VRAM: ~0.67 GB used on GPU[0]
```

This is about 9x faster than the heavy textured path's observed mean of
roughly 8-9 FPS per worker. It is still slower than material-color v03's
roughly 254 FPS per worker, but it puts full-corpus textured rendering back in
the practical range if multiple workers remain stable.

## Marker Backfill Performed

Command:

```text
python3 scripts/render_resume_markers.py backfill \
  --out .generated/datagen_full/render_textured \
  --rollout-root .generated/datagen_full/rollout
```

Result:

```json
{"checked": 1450, "marked": 16, "skipped_plans": 0}
```

## Safe Restart Guidance

The active tmux job is still running the old wrapper code. It will not benefit
from the patched resume behavior until it is stopped and restarted.

Recommended restart shape for the new textured-v03 path:

```bash
CORPUS=$PWD/.generated/scene_corpus/minimum_tex_20260520T211541Z \
ROLLOUT_ROOT=$PWD/.generated/datagen_full/rollout \
OUT=$PWD/.generated/datagen_full/render_textured_v03 \
RENDER_VENV=$PWD/.generated/venvs/genesis_render_vulkan \
RENDER_TEXTURES=1 \
EGL_DEVICE_ID=1 \
PYOPENGL_PLATFORM=egl \
XDG_CACHE_HOME=$PWD/.generated/cache \
MPLCONFIGDIR=$PWD/.generated/mplconfig \
HOME=$PWD/.generated/render_home \
WORKERS=4 \
scripts/datagen_render_resumable_v03.sh >> .generated/datagen_full/render_textured_v03_driver.log 2>&1
```

If `WORKERS=4` stays stable for several large scenes, increase to `WORKERS=8`.

The old heavy textured path should not be resumed for full-corpus rendering
unless depth-validation metrics are explicitly required:

```bash
CORPUS=$PWD/.generated/scene_corpus/minimum_tex_20260520T211541Z \
ROLLOUT_ROOT=$PWD/.generated/datagen_full/rollout \
OUT=$PWD/.generated/datagen_full/render_textured \
RENDER_VENV=$PWD/.generated/venvs/genesis_render_vulkan \
RENDER_BACKEND=vulkan \
REPLAY_ENV_MODE=single \
EGL_DEVICE_ID=1 \
PYOPENGL_PLATFORM=egl \
XDG_CACHE_HOME=$PWD/.generated/cache \
MPLCONFIGDIR=$PWD/.generated/mplconfig \
WORKERS=2 \
CORES_PER_WORKER=4 \
CAMERA_HZ=10 \
scripts/datagen_render_resumable.sh >> .generated/datagen_full/render_textured_driver.log 2>&1
```

Use the old path only for a targeted QA shard. Start at `WORKERS=2` if it must
be used.

## Remaining Risk

Resume is still scene-level only. If memory grows inside one long
`render_replay_genesis.py` process, the only clean mitigation is to make scenes
shorter at the render layer, for example by adding frame-range sharding and
manifest-level stitching. That is a separate change from the marker fix.
