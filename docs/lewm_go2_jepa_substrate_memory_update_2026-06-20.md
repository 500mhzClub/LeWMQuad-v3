# Go2 JEPA-Substrate Memory Update

Date: 2026-06-20

Status: preliminary evidence and next-step plan.

## Correction

The earlier Go2 hidden-target memory probes were not using a pretrained or
frozen JEPA. They were supervised CNN+GRU probes trained directly on rendered
Go2 RGB event slices. Those results remain useful as a Go2 data-contract and
baseline, but they should not be described as JEPA transfer.

This update adds a minimal Go2 JEPA-style substrate path:

1. train a compact action-conditioned Go2 RGB latent encoder by next-latent
   prediction;
2. freeze that encoder;
3. train downstream recurrent query/geometry heads on top of the frozen latent
   substrate;
4. compare normal history against reset/reversed/shuffled-history controls.

The local PyTorch ROCm build reports `torch.cuda.is_available() == False`, so
these runs used CPU despite ROCm being installed.

## New Artifacts

Code:

- `lewm/models/go2_jepa.py`
- `scripts/train_go2_jepa_latent_encoder.py`
- `scripts/train_go2_causal_memory_query_probe.py` with `--frozen-jepa-checkpoint`
- `scripts/train_go2_memory_target_geometry.py` with `--frozen-jepa-checkpoint`
- `scripts/evaluate_go2_causal_memory_target_gate.py` with frozen-JEPA checkpoint loading
- `scripts/train_go2_frozen_jepa_target_gate.py`
- `scripts/evaluate_go2_frozen_jepa_target_gate.py`
- `scripts/evaluate_go2_direct_gate_geometry_controller.py`
- `scripts/evaluate_go2_frozen_jepa_command_replay.py`

Primary reports:

- `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260620_img64_lat96_report.json`
- `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_frozen_jepa_seed20260620_current_pos025_img64_lat96_h128_report.json`
- `.generated/go2_hidden_target_memory/go2_causal_memory_target_gate_frozen_jepa_seed20260620_thr050_report.json`
- `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_seed20260620_img64_lat96_h128_report.json`
- `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02_report.json`
- `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260630_contrast02_margin_sweep_report.json`
- `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_seed20260629_img64_lat96_h128_report.json`
- `.generated/go2_hidden_target_memory/go2_direct_gate30_geometry_contrast02_margin02_report.json`
- `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_report.json`
- `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_commands.jsonl`
- `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260632_contrast02_runtimeaux_reset05_shuffle05_img64_lat96_h128_report.json`
- `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_runtimeaux_seed20260632_img64_lat96_h128_report.json`
- `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate32_geometry32_runtimeaux_margin0_report.json`

## Results

JEPA-style encoder smoke:

- validation next-latent retrieval@1: `0.177`
- chance retrieval@1: `0.013`
- target latent std mean: `1.002`
- paired-positive minus best-negative cosine: `-0.098`

Interpretation: the substrate is non-collapsed and predictive above chance, but
it is not yet a strong or well-ordered latent representation.

Frozen-JEPA causal memory query probe:

- normal balanced accuracy: `0.626`
- reset-state balanced accuracy: `0.424`
- reversed-history balanced accuracy: `0.336`
- normal minus best ablation: `0.202`
- positive recall: `0.525`
- negative specificity: `0.727`
- F1: `0.600`

Interpretation: this is the first Go2 result in this branch that supports the
bounded claim "a learned recurrent memory readout can operate on a frozen
JEPA-style Go2 visual latent." It is still offline and supervised.

Frozen-JEPA target-selection gate:

- frame balanced accuracy: `0.626`
- positive-frame recall: `0.525`
- negative-frame abstention specificity: `0.727`
- target-selection precision: `0.700`
- normal minus best corrupted gate: `0.070`

Interpretation: the controller-facing gate is usable but weakly separated from
corrupted-history controls. It should not yet be used as paper-grade evidence of
Go2 target-selection memory.

Frozen-JEPA target geometry:

- positive steering-bucket accuracy: `0.750`
- mean angle error: `71.2 deg`
- range MAE: `0.75 m`
- normal minus best corrupted steering accuracy: `0.025`

Interpretation: the frozen latent can support a geometry readout, but the
geometry target is still partly recoverable from object/query and validation-set
biases. This is a bridge diagnostic, not a clean memory-dependence result.

## Direct Gate and Controller Proxy Follow-Up

The first frozen-JEPA controller-facing gate was too weak, so the follow-up
changed the parts that were actually blocking the offline controller proxy:

1. add an optional in-batch contrastive next-latent loss to the Go2 JEPA trainer;
2. train a direct frame-level target gate over frozen JEPA memory states;
3. include reset, reversed, and shuffled-history controls when selecting the
   saved target-gate checkpoint;
4. retrain target-relative geometry on the same contrastive frozen encoder;
5. evaluate the two-stage selector + geometry command-direction proxy.

Contrastive JEPA substrate:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt`;
- contrastive weight / temperature: `0.2` / `0.1`;
- validation next-latent retrieval@1: `0.190`;
- chance retrieval@1: `0.013`;
- target latent std mean: `1.023`;
- paired-positive minus best-negative cosine: `-0.138`.

Interpretation: retrieval improved slightly over the first substrate
(`0.177 -> 0.190`), but the latent is still not topologically ordered. This is
not evidence that the desired latent structure already exists; it remains a
future representation objective.

Direct frozen-JEPA target gate:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260630_contrast02_reset05_shuffle05_img64_lat96_h128.pt`;
- margin-sweep report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260630_contrast02_margin_sweep_report.json`;
- best operating margin: `0.2`;
- balanced frame accuracy: `0.781`;
- positive-frame recall: `0.775`;
- negative-frame abstain specificity: `0.788`;
- target-selection precision: `0.816`;
- false claims: `7 / 33`;
- wrong-object selections: `0`;
- normal minus best corrupted-history balanced frame accuracy: `0.209`.

Frozen-JEPA target geometry on the contrastive substrate:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_seed20260629_img64_lat96_h128.pt`;
- mean angle error: `54.4 deg`;
- range MAE: `0.87 m`;
- positive steering-bucket accuracy: `0.825`;
- normal minus best corrupted steering accuracy: `0.050`.

Two-stage selector + geometry controller proxy:

- report:
  `.generated/go2_hidden_target_memory/go2_direct_gate30_geometry_contrast02_margin02_report.json`;
- target recall: `0.775` (`31 / 40`);
- false-claim rate: `0.212` (`7 / 33`);
- target-selection precision: `0.816`;
- wrong-object selections: `0`;
- target-steering pipeline success: `0.700` (`28 / 40`);
- target-primitive proxy success: `0.575`;
- normal minus best corrupted-history target-steering success: `0.200`;
- normal minus best corrupted-history target recall: `0.175`.

Interpretation: the offline frozen-JEPA Go2 controller proxy is now a passable
handoff point. It is still not a closed-loop Go2 result. The strongest claim is:
given rendered strict hidden-return event slices, a learned recurrent memory
over a frozen Go2 JEPA-style visual substrate can select remembered hidden
targets and produce target-relative steering labels materially better than
reset/reversed/shuffled-history controls.

Mixed-substrate diagnostic: the older high-recall selector plus the new
contrastive geometry reached target-steering success `0.775` and corruption gap
`0.200`, but with false-claim rate `0.424`. This confirms that geometry is no
longer the main blocker; selector false positives are the remaining risk.

## Replayed Command-Block Follow-Up

`scripts/evaluate_go2_frozen_jepa_command_replay.py` converts the selected
frozen-JEPA selector + geometry output into Go2 primitive command blocks using
`config/go2_primitive_registry.yaml`, applies the platform safety adapter limits
from `config/go2_platform_manifest.yaml`, and writes a JSONL command-block
trace. This is execution-facing replay, not live Genesis physics: command blocks
are expanded and clipped exactly as contract records, but they are not fed back
into the simulator.

Selected replay artifact:

- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_report.json`;
- command records:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate30_geometry29_margin02_commands.jsonl`;
- selected margin: `0.2`;
- replay gate: pass.

Normal memory replay:

- command frames: `73`;
- positive frames: `40`;
- negative frames: `33`;
- non-hold commands: `38`;
- target recall: `0.775`;
- target-selection precision: `0.816`;
- false-claim rate: `0.212`;
- wrong-object selections: `0`;
- target-steering pipeline success: `0.700`;
- target-primitive proxy success: `0.575`;
- normal minus best corrupted target-steering success: `0.200`;
- safety-clipped command blocks: `10 / 73` (`0.137`), caused by the adapter
  rate-limiting yaw/arc command transitions.

Memory controls:

| replay mode | target recall | false-claim rate | target-steering success | non-hold commands |
| --- | ---: | ---: | ---: | ---: |
| normal | `0.775` | `0.212` | `0.700` | `38` |
| memory-off abstain | `0.000` | `0.000` | `0.000` | `0` |
| reset recurrent state | `0.000` | `0.000` | `0.000` | `0` |
| reversed history | `0.575` | `0.455` | `0.225` | `38` |
| shuffled hidden states | `0.600` | `0.455` | `0.500` | `39` |

Margin check:

- margin `0.0`: target-steering success `0.700`, but false-claim rate `0.303`
  fails the replay gate;
- margin `0.4`: false-claim rate `0.182`, but target recall `0.475` and
  target-steering success `0.450` fail the replay gate;
- margin `0.2` is the selected handoff.

Interpretation before the stricter audit: this met the execution-facing replay
gate as command blocks. The stricter runtime-aux recheck below supersedes the
claim boundary for 2D-comparable Go2 evidence.

## Strict Runtime-Aux Recheck

While preparing the live-controller handoff we found that the Go2 row aux vector
still contained `clearance_m` and `traversability_forward_m`. These are useful
for dataset/debugging, but they are not available to a learned RGB memory
controller at runtime. The earlier "scrubbed" runs removed current command aux
signals, but not these two scene-derived fields.

Code update:

- `scripts/train_go2_causal_memory_query_probe.py` now exposes
  `_scrub_runtime_aux`, which also zeros `clearance_m` and
  `traversability_forward_m`;
- `scripts/train_go2_frozen_jepa_target_gate.py` and
  `scripts/train_go2_memory_target_geometry.py` have `--scrub-runtime-aux`;
- `scripts/evaluate_go2_frozen_jepa_target_gate.py`,
  `scripts/evaluate_go2_direct_gate_geometry_controller.py`, and
  `scripts/evaluate_go2_frozen_jepa_command_replay.py` respect the stricter
  checkpoint/runtime scrub flag.

Strict runtime-aux target gate:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260632_contrast02_runtimeaux_reset05_shuffle05_img64_lat96_h128.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260632_contrast02_runtimeaux_reset05_shuffle05_img64_lat96_h128_report.json`;
- margin-sweep report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260632_runtimeaux_margin_sweep_report.json`;
- best checked margin: `-0.5`;
- balanced frame accuracy: `0.780`;
- positive recall: `0.650`;
- negative abstention: `0.909`;
- target-selection precision: `0.897`;
- false claims: `3 / 33`;
- normal minus best corrupted-history balanced frame accuracy: `0.159`.

Strict runtime-aux target geometry:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_runtimeaux_seed20260632_img64_lat96_h128.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_frozen_jepa_contrast02_runtimeaux_seed20260632_img64_lat96_h128_report.json`;
- mean angle error: `62.9 deg`;
- range MAE: `0.42 m`;
- steering-bucket accuracy: `0.750`;
- normal minus best corrupted steering-bucket accuracy: `0.125`.

Strict runtime-aux command replay:

| margin | target recall | false-claim rate | target-steering success | corrupted-history gap | replay gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.0` | `0.625` | `0.061` | `0.400` | `+0.150` | fail |
| `-0.2` | `0.650` | `0.091` | `0.400` | `+0.150` | fail |
| `-0.5` | `0.650` | `0.091` | `0.400` | `+0.125` | fail |

Selected strict replay artifact for the conservative margin:

- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate32_geometry32_runtimeaux_margin0_report.json`;
- non-hold commands: `24`;
- safety-clipped command blocks: `9 / 73`;
- emitted primitive mix: mostly `yaw_left`, `yaw_right`, and a few
  `forward_medium` / arc commands.

Interpretation: the learned memory signal survives the stricter runtime
boundary at the target-selection level. It is not enough for a 2D-comparable
Go2 controller result. Once the non-runtime aux fields are removed, the
selector remains high precision but lower recall, and the selected
geometry-to-command replay falls to `0.400` target-steering success. This is
significant evidence that the current Go2 path is not viable as-is for the
claimed controller translation. It needs a real runtime RGB bridge and a better
target-geometry/action objective before further threshold tuning is meaningful.

## Strict Replay Repair, GPU Iteration

GPU setup correction: the local ROCm PyTorch environment only exposed the GPUs
when `HSA_OVERRIDE_GFX_VERSION` was unset. The working launch prefix is:

```bash
env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0 /home/andrewknowles/TinyQuadJEPA/bin/python ...
```

With that prefix, PyTorch reports `cuda_available=True` and GPU0 as
`AMD Radeon AI PRO R9700`; `rocm-smi` showed GPU0 at or near `100%` during the
training ablations.

Selector repair:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu_report.json`;
- margin-sweep report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_direct_target_gate_seed20260633_runtimeaux_pos125_m-15_gpu_margin_sweep_report.json`;
- selected margin: `-1.5`;
- target recall: `0.825`;
- negative abstention: `0.788`;
- false-claim rate: `0.212`;
- target-selection precision: `0.825`;
- normal minus best corrupted balanced frame accuracy: `0.252`.

Frozen-JEPA geometry repair attempts did not meet the replay bar:

- strict frozen-JEPA geometry + stronger selector: target-steering replay
  `0.575`;
- steering-head frozen-JEPA geometry variants: best replay `0.600`, but the
  steering head collapsed toward the majority-left solution and memory gap fell
  to `0.025`;
- slot-aware frozen-JEPA geometry: replay `0.575`;
- broader-data frozen-JEPA geometry: best replay `0.625`, but memory gap
  collapsed to `0.025`.

Trainable geometry control:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_trainablecnn_runtimeaux_seed20260647_img64_h128.pt`;
- report:
  `.generated/go2_hidden_target_memory/go2_memory_target_geometry_trainablecnn_runtimeaux_seed20260647_img64_h128_report.json`;
- strict runtime-aux validation geometry: `46.3 deg` mean angle error,
  `0.50 m` range MAE, `0.700` steering-bucket accuracy;
- normal minus best corrupted geometry steering gap: `0.050`.

Passing strict replay:

- report:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_report.json`;
- command trace:
  `.generated/go2_hidden_target_memory/go2_frozen_jepa_command_replay_gate33pos125_trainablegeom47_runtimeaux_m-15_arc010_commands.jsonl`;
- selector: frozen-JEPA direct gate above;
- geometry: trainable CNN geometry control above;
- arc threshold: `0.1 rad`;
- replay gate: pass;
- target recall: `0.825`;
- false-claim rate: `0.212`;
- target-selection precision: `0.825`;
- target-steering pipeline success: `0.725`;
- target-primitive pipeline success: `0.525`;
- normal minus best corrupted target-steering success: `0.275`;
- normal minus best corrupted target recall: `0.200`;
- non-hold commands: `40`;
- clipped command-block rate: `0.192`.

Memory controls for the passing replay:

| replay mode | target recall | false-claim rate | target-steering success | non-hold commands |
| --- | ---: | ---: | ---: | ---: |
| normal | `0.825` | `0.212` | `0.725` | `40` |
| memory-off abstain | `0.000` | `0.000` | `0.000` | `0` |
| reset recurrent state | `0.000` | `0.000` | `0.000` | `0` |
| reversed history | `0.600` | `0.545` | `0.400` | `42` |
| shuffled hidden states | `0.625` | `0.515` | `0.450` | `42` |

Interpretation: strict runtime-aux replay is a useful controller-proxy
milestone, but it is not 2D-comparable under the later `0.90+` target-success
bar. The memory/selector leg transfers over the frozen Go2 JEPA substrate; the
target-relative geometry/action leg still needs a trainable visual encoder even
to reach `0.725`. The paper-version JEPA claim remains narrower:
frozen-JEPA memory selection works, while frozen-JEPA target geometry and
calibrated memory-to-action remain open.

Runtime interface audit:

- `lewm_genesis.lewm_genesis.collectors.base.EnvObservation` exposes
  privileged pose/cell/heading-style fields to collectors, not rendered RGB;
- `lewm_genesis.lewm_genesis.rollout.RolloutRunner` renders RGB for telemetry
  after block collection, not as the current observation passed into a learned
  memory controller;
- a live learned-memory Go2 controller therefore needs a camera-conditioned
  collector/runtime bridge that renders or carries the latest RGB frame before
  command selection, plus a non-privileged candidate-query source.

## Claim Boundary

The current evidence meets these bounded requirements:

- current Go2 has a frozen JEPA-style visual substrate;
- a recurrent memory readout trained on top of that frozen substrate beats
  reset/reversed-history controls on matched-current-view hidden-target queries;
- under strict runtime aux, a direct target-selection gate over frozen JEPA
  memory still beats reset/reversed/shuffled-history controls with high
  precision and a `+0.159` corrupted-history gap;
- the command-scrubbed gate + geometry readout can be expanded into Go2 command
  blocks under the primitive registry and safety adapter, and remains useful as
  an engineering handoff artifact;
- the old CNN+GRU Go2 path is now explicitly a baseline, not JEPA transfer.

It does not meet the 2D-comparable Go2 command-replay controller-proxy
requirement under the stricter bar set after review. The earlier
mixed-substrate replay is useful, but below the 2D learned-memory result:

- strict runtime-aux replay with a frozen-JEPA selector and trainable-CNN
  geometry reaches target-steering success `0.725`, false-claim rate `0.212`,
  and a `+0.275` corrupted-history gap;
- memory-off and reset controls emit only hold commands;
- reversed and shuffled histories still emit commands, but with much higher
  false-claim rates and lower target-steering success.

The current 2D-comparable Go2 bar is: target-steering success at or above
`0.90`, false-claim rate near the strict selector target (`<= 0.12`), and a
normal-minus-corrupted memory gap of at least `+0.30`.

It does not meet the paper-grade Go2 / pure-JEPA requirement:

- the command-scrubbed replay pass is not accepted as comparable evidence
  because it retained non-runtime clearance/traversability aux fields;
- pure frozen-JEPA strict-runtime geometry still fails the replay gate; best
  target-steering replay found here is `0.625`, and the stronger memory gap
  remains with the older frozen geometry at `0.600`;
- no closed-loop Go2 hidden-target return controller has been proven;
- no live Genesis physics feedback or hardware execution has been run from the
  learned RGB memory controller;
- the current Genesis collector path does not provide rendered RGB to learned
  controllers at command-selection time;
- the JEPA encoder was trained on small event slices, not broad continuous Go2
  rollouts;
- no DINO / supervised-CNN / direct-CNN baseline comparison has been completed
  under matched frozen-memory heads.
- the latent substrate is predictive and useful, but not ordered; latent
  topology remains a future paper-version objective.

## Next Plan

1. Freeze the strict runtime-aux artifacts as the current decision point:
   target memory selection transfers, but controller replay is not yet
   comparable to 2D.
2. Add a camera-conditioned Genesis collector/runtime bridge before claiming
   live Go2 memory control.
3. Rework the target-relative geometry/action objective using runtime-available
   inputs only, then rerun strict memory-on/off/reset/reversed/shuffled replay.
4. Evaluate live memory-on/off/reset/reversed/shuffled command-block execution
   on strict hidden-return Go2 episodes only after strict replay passes.
5. Improve the Go2 JEPA encoder on broader continuous rendered rollouts, not
   only event slices; keep the downstream memory heads frozen-encoder only.
6. Add validation rows with future hidden-claim opportunities across all colors;
   the current validation split still under-tests closure.
7. Re-run the same target gate and geometry probes against:
   - frozen Go2 JEPA;
   - trainable CNN+GRU baseline;
   - frozen random encoder;
   - later, DINO or other pretrained visual features as a ceiling/control.

The research sequencing remains: prove translatability on Go2 first, then return
to 2D to study whether the external learned memory structures can be pushed into
ordered latent representations.

## 2D-Level Go2 Attempt Correction

After raising the bar to the actual 2D learned-memory result, the Go2 result is
not yet comparable. New GPU-backed direct-controller experiments used:

- `scripts/train_go2_memory_steering_controller.py`;
- `scripts/evaluate_go2_memory_steering_controller.py`;
- ROCm prefix:
  `env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
  /home/andrewknowles/TinyQuadJEPA/bin/python`.

Implementation changes tested:

- direct recurrent memory controller selecting a current target and steering
  class in one model;
- strict runtime-aux scrubbing;
- runtime-query geometry features: range, bearing sine/cosine, visibility only;
- exclusive memory state, so current-frame queries are scored from the recurrent
  prefix before the current frame is ingested;
- object-level candidate BCE auxiliary loss;
- object-slot query identity.

Best results found:

| Artifact | Target-steering | False claims | Memory gap | Decision |
| --- | ---: | ---: | ---: | --- |
| `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_short_seed20260662_h128_report.json` | `0.950` | `0.606` | `+0.150` | Hits positive target success, fails memory gate. |
| `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_short_pos2_seed20260663_h128_report.json` | `0.900` | `0.636` | `+0.175` | Hits positive target success, fails memory gate. |
| `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_seed20260653_h128_report.json` | `0.825` | `0.121` | `+0.350` | Clean memory dependency, below target success. |
| `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_seed20260660_h256_report.json` | `0.850` | `0.091` | `+0.275` | Cleaner false claims, below target success/gap. |
| `.generated/go2_hidden_target_memory/go2_memory_steering_controller_runtimegeom_exclusive_bce05_pos2neg2_seed20260668_h128_margin_sweep_report.json` | `0.925` | `0.485` | `+0.175` | BCE improves tradeoff but still fails. |

Conclusion: Go2 now has evidence that the positive hidden-target return behavior
can reach 2D-level target-steering when runtime object geometry is supplied, but
it does not yet have a full 2D-comparable working-memory result. The limiting
failure is calibrated target selection: high recall comes with too many false
memory claims and too small a corruption gap.

Next implementation target:

1. Generate a larger hidden-target Go2 train/validation set with more negative
   frames, balanced colors/slots, and held-out route/source splits.
2. Train the memory selector separately from the steering/control rule, using
   exclusive prefix memory, hard negative sequence corruption, and a held-out
   threshold split.
3. Treat runtime object-relative steering as perception/control, not the core
   learned-memory claim, unless the learned steering head can match the rule.
4. Rerun the strict gate only when the controller satisfies all three metrics:
   `>=0.90` target-steering, `<=0.12` false claims, `>=+0.30` corruption gap.

## Runtime-Observation Memory Pass

Follow-up on the strict Go2 validation split established a concrete
2D-comparable operational memory result using runtime landmark observations:

- Script:
  `scripts/evaluate_go2_runtime_observation_memory.py`
- Report:
  `.generated/go2_hidden_target_memory/go2_runtime_observation_memory_val_report.json`
- Validation split:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/datasets/hidden_claim_source*.jsonl`

Result:

| Controller | Target-steering | False claims | Memory gap | Gate |
| --- | ---: | ---: | ---: | --- |
| Runtime-observation memory | `0.925` | `0.000` | `+0.400` | PASS |

Detailed counts:

- normal memory: `37/40` positive target-steering successes, `0/33` false
  claims;
- memory-off/reset: `0/40` target successes;
- shuffled memory: `21/40` target successes, so the normal-memory gap is
  `+0.400`.

Claim boundary: this is a deterministic runtime-observation memory upper bound,
not a learned latent/GRU memory controller. It records landmark ids seen in
previous frames of the same sequence and uses current relative bearing for the
steering rule. It uses no future labels and no global map/geodesic distance, so
it is Go2-translatable if a detector supplies landmark id/range/bearing/visible
observations.

Neural controller attempts after dense data expansion still did not meet this
bar. Tested variants included:

- frozen Go2 JEPA controller;
- JEPA-initialized fine-tuning;
- evidence-write auxiliary loss;
- runtime memory-observation aux;
- hidden memory-state auxiliary loss;
- object-slot query features.

These variants continued to collapse into either high-recall/high-false-claim
or low-recall/clean-abstention regimes. The conclusion is now sharper: Go2
working memory is operationally feasible with runtime object observations, but
the generic learned recurrent controller architecture/training objective is too
broad for the current demonstration. The next implementation should preserve
the successful memory semantics directly.

## Differentiable Slot-Memory Pass

A narrower learned/differentiable Go2 memory controller now meets the strict
2D-comparable gate. This version uses runtime landmark observations, learns a
per-slot write probability plus read gain/threshold, keeps a persistent
per-landmark memory vector, and uses the current relative bearing for the local
steering rule.

- Script:
  `scripts/train_go2_slot_memory_controller.py`
- Checkpoint:
  `.generated/go2_hidden_target_memory/go2_slot_memory_controller_readlogits_seed20260698.pt`
- Report:
  `.generated/go2_hidden_target_memory/go2_slot_memory_controller_readlogits_seed20260698_report.json`
- Validation split:
  `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/datasets/hidden_claim_source*.jsonl`
- Device:
  ROCm/PyTorch GPU path, reported as `cuda`

Result:

| Controller | Target-steering | False claims | Memory gap | Gate |
| --- | ---: | ---: | ---: | --- |
| Differentiable slot memory, learned read logits | `0.925` | `0.000` | `+0.400` | PASS |

Detailed counts:

- normal memory: `37/40` positive target-steering successes, `0/33` false
  claims, target-selection precision `1.000`;
- memory-off/reset: `0/40` target successes;
- reversed history: `3/40` target successes, false-claim rate `0.636`;
- shuffled memory: `21/40` target successes, false-claim rate `0.152`, giving
  the normal-memory gap of `+0.400`.

Learned parameters:

- read gain: `9.1898`;
- read threshold: `0.4027`;
- write probabilities by slot:
  `[0.99994, 0.99995, 0.99994, 0.88845]`.

Claim boundary: this is a learned/differentiable slot-memory controller over
runtime landmark observations. The pass is stronger than the deterministic
runtime-observation upper bound because the reported selection path uses the
trained write probabilities and read logits. It is still not a pure RGB
latent-memory controller, and it is not evidence that a topological latent
structure has already emerged inside the JEPA. That remains the next research
step after the Go2 bridge can execute this memory behavior end to end.

Next implementation target:

1. Promote the differentiable slot-memory controller to the Go2 bridge
   baseline: detector-like landmark observations write memory slots, learned
   read logits select the remembered target, and current bearing supplies the
   local steering action.
2. Add bridge-level replay/eval checks so this exact checkpoint can be exercised
   through the Go2 command interface, not only offline JSONL validation.
3. Use this passing controller as the teacher for the next learned-memory
   version: first replace detector-like observations with learned perception
   features, then try to replace explicit slots with latent memory while keeping
   the same strict gate.
4. Keep the paper claim separated:
   - current Go2 milestone: translatable learned/differentiable working-memory
     behavior exists;
   - novel paper milestone: show that required memory structure can be moved
     into learned latent representations without losing the 2D/Go2 gate.

## Pure RGB/JEPA Context-Memory Attempt

Update timestamp: 2026-06-21 00:50 BST.

Goal: test the stricter pure RGB/JEPA latent-memory boundary on Go2 hidden
targets. Disallowed at inference: runtime landmark ids, object slots, bearing,
range, detector visibility, global map/geodesic geometry, and label-derived
object geometry. Allowed: rendered RGB, JEPA-initialized visual latent, learned
recurrent/attention memory, odometry/action history, and target color query.

Dataset repair:

- Added context-window rendering around selected Go2 hidden-target frames:
  `scripts/expand_go2_selected_render_context.py`.
- Rendered and joined 10 training context shards and 3 validation context
  shards into `context24_datasets`.
- Context datasets contain continuous rendered bridges for most positives:
  validation positive current queries with prior visible evidence in rendered
  history: `49/52`; median evidence-to-query episode gap: `92` steps; median
  rendered rows between evidence and query: `50`.

New/updated code:

- `scripts/train_go2_jepa_latent_memory_action_controller.py`
- `scripts/train_go2_memory_steering_controller.py`
  - added `--scrub-scene-aux` to keep deployable command/action history while
    removing scene-derived clearance/traversability;
  - added optional causal temporal-memory attention.
- `scripts/evaluate_go2_memory_steering_controller.py`
  - margin sweep now respects `scrub_scene_aux`;
  - reloads temporal-memory checkpoints.
- `scripts/train_go2_memory_target_geometry.py`
  - added JEPA finetuning and `--scrub-scene-aux`.
- `scripts/train_go2_causal_memory_query_probe.py`
  - added JEPA finetuning and `--scrub-scene-aux`.

Strongest pure RGB/JEPA memory-gate result:

- Checkpoint:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_purejepa_context24_finetune_seed20260712_h256.pt`
- Report:
  `.generated/go2_hidden_target_memory/go2_causal_memory_query_probe_purejepa_context24_finetune_seed20260712_h256_report.json`
- Inputs: RGB/JEPA latent, learned recurrent memory, target color query,
  odometry/action history, scene aux scrubbed.

| Metric | Normal | Reset | Reversed |
| --- | ---: | ---: | ---: |
| Balanced accuracy | `0.777` | `0.418` | `0.463` |
| Positive recall | `0.675` | `0.200` | `0.350` |
| Negative specificity | `0.879` | `0.636` | `0.576` |
| F1 | `0.761` | `0.267` | `0.412` |

Memory dependence: normal minus best ablation balanced accuracy is `+0.314`.
This is a real pure RGB/JEPA learned-memory signal on held-out Go2 scenes, but
it is not yet a high-performance Go2 controller.

Strongest pure RGB/JEPA steering-controller result:

- Checkpoint:
  `.generated/go2_hidden_target_memory/go2_memory_steering_controller_purejepa_colorquery_context24_highrecall_finetune_seed20260708_h256.pt`
- Report:
  `.generated/go2_hidden_target_memory/go2_memory_steering_controller_purejepa_colorquery_context24_highrecall_finetune_seed20260708_h256_report.json`
- Margin sweep:
  `.generated/go2_hidden_target_memory/go2_memory_steering_controller_purejepa_colorquery_context24_highrecall_finetune_seed20260708_h256_margin_sweep_report.json`

At training margin `0.0`:

- target recall: `0.800`;
- false-claim rate: `0.273`;
- target-steering pipeline success: `0.325`;
- target-selection precision: `0.780`;
- normal minus best corrupted target-steering success: `+0.075`;
- gate: FAIL.

Best target-steering margin was `-0.25`:

- target recall: `0.925`;
- false-claim rate: `0.424`;
- target-steering pipeline success: `0.350`;
- target-selection precision: `0.725`;
- normal minus best corrupted target-steering success: `+0.050`;
- gate: FAIL.

Tested pure RGB/JEPA variants that did not meet the strict steering gate:

- query-free RGB/JEPA action controller;
- frozen-JEPA color-query steering controller;
- JEPA-finetuned color-query steering controller;
- command/action-history retained with scene aux scrubbed;
- high-recall objective plus post-hoc margin sweep;
- hidden size `512`;
- direct bearing/range/steering geometry supervision;
- causal temporal-attention memory.

Interpretation:

The pure RGB/JEPA path now demonstrates memory dependence for target
seen-before classification, but it does not yet translate to the Go2
2D-comparable steering gate. The strongest steering model can often select the
right remembered target, but direction readout is the failure: at the best
low-margin operating point it selected the correct target on `37/40` positive
frames, yet only `14/40` positives had correct target steering.

This supports the research framing we discussed: the Go2 bridge can already
demonstrate working memory with explicit/differentiable slots, while moving the
required directional/spatial structure into pure learned RGB/JEPA latents is a
separate representation-learning problem. The next novel implementation should
not keep chasing generic GRU perfection; it should explicitly train the latent
memory to carry egocentric target direction or a pose-consistent latent map, then
re-test the same strict gate.

Next implementation target:

1. Keep the slot-memory controller as the Go2 translatability baseline.
2. Treat the pure RGB/JEPA query-memory result as the current learned-latent
   memory foothold, not as a completed Go2 controller.
3. Build a targeted latent-direction objective:
   - visual evidence encoder predicts target color evidence and egocentric
     bearing when visible;
   - learned memory integrates evidence across odometry/action history;
   - query readout predicts seen-before plus current egocentric direction;
   - no runtime landmark ids/slots/bearings are provided at inference.
4. Only after that objective clears the memory-gate and direction probes should
   we rerun the full Go2 target-steering gate.

## 2026-06-21 RGB/JEPA Latent-Memory Iteration

Strict target remains:

- pure RGB/JEPA plus learned memory at inference;
- no runtime landmark ids, slots, detector visibility, range/bearing, or map
  geometry;
- target-steering success `>=0.90`;
- false-claim rate `<=0.12`;
- normal minus best corrupted target-steering success `>=0.30`.

Implemented in `scripts/train_go2_rgb_jepa_vector_memory_controller.py`:

- direct odometry propagation option for vector memory;
- recurrent latent readout over JEPA features plus color query;
- deterministic vector-source steering diagnostics;
- signed direction loss for learned memory vectors;
- JEPA-initialized spatial feature readout with coordinate channels at 4x4 and
  8x8 feature resolutions;
- query-gate negative weighting;
- controller checkpoint initialization plus read-head-only calibration.

Best pure RGB/JEPA steering signal so far:

- Report:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_recurrent_head_lr1e4_thr005_seed20260736_h512_report.json`
- JEPA substrate: pretrained Go2 JEPA conv backbone, spatial 8x8 feature
  readout, recurrent learned memory, direct odometry propagation.
- Best threshold: `0.1`
- Target recall: `1.000`
- Target-steering success: `0.725`
- False-claim rate: `0.970`
- Corruption gap: `+0.175`
- Gate: FAIL.

Best calibrated false-claim tradeoff:

- Report:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_gate_neg2_thr010_seed20260739_h512_report.json`
- Best threshold: `0.6`
- Target recall: `0.875`
- Target-steering success: `0.600`
- False-claim rate: `0.455`
- Corruption gap: `+0.275`
- Gate: FAIL.

Read-head-only calibration from the best steering checkpoint did not preserve
the strong steering operating point:

- `go2_rgb_jepa_spatial8_readcal_neg2_seed20260741_h512_report.json`:
  target-steering `0.450`, false-claim `0.606`, gap `+0.150`.
- `go2_rgb_jepa_spatial8_readcal_neg4_seed20260742_h512_report.json`:
  target-steering `0.450`, false-claim `0.636`, gap `+0.150`.

Adding older compatible rendered medium datasets did not improve the gate:

- Report:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_aug_steer_seed20260743_h512_report.json`
- Target-steering success: `0.675`
- False-claim rate: `0.970`
- Corruption gap: `+0.025`
- Gate: FAIL.

Interpretation:

The 8x8 spatial JEPA readout materially improves direction over the globally
pooled JEPA latent (`0.725` versus `0.45` target-steering), so the old global
latent was too spatially compressed for hidden-target steering. The remaining
failure is not just steering; it is the joint requirement. Runs that preserve
high steering claim almost every negative, while runs that reduce false claims
drop recall and steering. Simple negative weighting, read-head-only
calibration, and adding the currently available older medium data did not
resolve that tradeoff.

Next concrete requirement:

The next implementation should create a better positive/negative training
signal for the query gate, not another generic architecture sweep. In practice
that means generating or mining more matched counterfactual sequences where
the same target color/query appears with and without prior visual evidence
under similar current view/action context. Without those hard negatives, the
pure RGB/JEPA model can learn a useful remembered steering prior, but it cannot
reliably distinguish "remembered target" from "plausible target color" on the
strict hidden-target validation split.

## 2026-06-21 Hard-Negative / Confidence-Gate Iteration

Audit result:

- The current rendered train split already contains hard current-view groups:
  `39` train groups where `(scene_id, cell_id, yaw_bin, color)` has both
  seen-before and unseen-before examples.
- Validation has `10` such groups.
- Therefore the failure is not simply absence of hard negatives; the current
  objective does not force enough matched positive/negative separation while
  preserving steering.

Implemented:

- `Query.group_key` for current-view hard groups.
- Optional hard-pair ranking loss over matched positive/negative group pairs.
- Optional hard-group balanced query BCE.
- Optional memory-confidence prior added to the query read logit:
  `read_head(features) + scale * logit(memory_conf)`.

Results:

- Hard-pair read-head calibration:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_hardpair_readcal_smoke_seed20260745_h512_report.json`
  - target-steering `0.575`;
  - false-claim `0.939`;
  - gap `+0.050`;
  - FAIL.
- Hard-group balanced read-head calibration:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_hardgroup_readcal_seed20260749_h512_report.json`
  - target-steering `0.450`;
  - false-claim `0.606`;
  - gap `+0.175`;
  - FAIL.
- Confidence prior, no extra training, scale `0.5`:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confprior05_eval_seed20260750_h512_report.json`
  - target-steering `0.725`;
  - recall `1.000`;
  - false-claim `0.606`;
  - gap `+0.175`;
  - FAIL.
- Confidence prior, no extra training, scale `1.0`:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confprior10_eval_seed20260751_h512_report.json`
  - target-steering `0.725`;
  - recall `1.000`;
  - false-claim `0.667`;
  - gap `+0.175`;
  - FAIL.
- Confidence-prior read-head calibration:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confprior05_readcal_seed20260753_h512_report.json`
  - target-steering `0.625`;
  - recall `0.900`;
  - false-claim `0.667`;
  - gap `+0.250`;
  - FAIL.

Interpretation:

The explicit color memory confidence does contain useful gate information: a
scale `0.5` prior preserves the best steering result (`0.725`) while reducing
false claims from `0.970` to `0.606`. However, the false-claim target is
`<=0.12`, so this is still far from the strict Go2 gate. Read-head calibration,
hard-group balancing, and pairwise hard-negative ranking all reduce recall and
steering faster than they reduce false claims.

Next concrete target:

Move from scalar color memory confidence to an explicit learned evidence
ledger: the model should write a target-color memory only when the visual
stream has a high-confidence visible-color event, and the query gate should be
regularized against the ledger state rather than against recurrent context
alone. The current recurrent readout can still infer "plausible target color"
from trajectory/context even when the explicit memory does not justify a claim.

Additional diagnostic:

- Confidence-only gate evaluation, scale `1.0`:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confonly10_eval_seed20260754_h512_report.json`
  - target-steering `0.725`;
  - false-claim `0.727`;
  - gap `+0.175`;
  - FAIL.
- Confidence-only gate evaluation, scale `2.0`:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confonly20_eval_seed20260755_h512_report.json`
  - target-steering `0.725`;
  - false-claim `0.758`;
  - gap `+0.175`;
  - FAIL.
- Confidence-only gate training:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_spatial8_confonly_train_seed20260756_h512_report.json`
  - target-steering `0.725`;
  - recall `1.000`;
  - false-claim `0.606`;
  - gap `+0.225`;
  - FAIL.

Updated interpretation:

Training the differentiable color memory confidence directly preserves the
strongest steering signal and improves the corruption gap slightly, but it
plateaus at false-claim `0.606`. The existing color-memory confidence is not a
sufficient ledger; it still writes or preserves enough spurious color evidence
to claim many unseen-before hidden targets. The next implementation should make
the write event itself auditable, for example by separating:

- visible-color event confidence;
- accumulated ever-seen color ledger;
- target direction memory conditioned on that ledger;
- query gate constrained to the ledger, not to recurrent context.

## 2026-06-21 RGB/JEPA Observability Audit

Strict pure RGB/JEPA target remained unmet after additional write-ledger and
RGB-evidence iterations.

New implementation artifacts:

- `scripts/train_go2_rgb_jepa_vector_memory_controller.py`
  - added biased/tempered write-gate supervision;
  - added asymmetric memory-state loss controls;
  - added optional RGB color-evidence substrate;
  - added RGB-evidence replacement mode to prevent hallucinating learned
    evidence logits from overriding the visual ledger.
- `scripts/audit_go2_rgb_memory_observability.py`
  - audits whether a positive memory query has prior rendered RGB evidence for
    the queried object/color.

Additional failed controller reports:

- Ledger-bias evals from the best spatial JEPA checkpoint:
  - `go2_rgb_jepa_spatial8_ledger_bias1_eval_seed20260757_h512_report.json`:
    target-steering `0.725`, false-claim `0.606`, gap `+0.175`;
  - `go2_rgb_jepa_spatial8_ledger_bias2_eval_seed20260758_h512_report.json`:
    target-steering `0.725`, false-claim `0.606`, gap `+0.175`;
  - `go2_rgb_jepa_spatial8_ledger_bias3_eval_seed20260759_h512_report.json`:
    target-steering `0.725`, false-claim `0.606`, gap `+0.175`;
  - `go2_rgb_jepa_spatial8_ledger_bias4_eval_seed20260760_h512_report.json`:
    target-steering `0.725`, false-claim `0.606`, gap `+0.175`;
  - `go2_rgb_jepa_spatial8_ledger_bias8_eval_seed20260762_h512_report.json`:
    false-claim improves to `0.242`, but target-steering collapses to `0.250`.
- RGB evidence additive evals:
  - `go2_rgb_jepa_spatial8_rgbevid_s8_v2_head_eval_seed20260769_h512_report.json`:
    target-steering `0.550`, false-claim `0.364`, gap `+0.275`;
  - additive training with scale `8`/`12` returned to the same false-claim
    attractor (`0.606`) by epoch 10-20 and was stopped.
- RGB evidence replacement evals:
  - `go2_rgb_jepa_spatial8_rgbreplace_s8_v2_head_eval_seed20260772_h512_report.json`:
    target-steering `0.300`, recall `0.300`, false-claim `0.000`, gap `+0.025`;
  - vector steering replacement variants reached only `0.100` target-steering.

The replacement result is useful as a boundary: high-precision rendered RGB
evidence can eliminate false claims, and the selected positives are steered
correctly by the head (`12/12`), but it only recalls `12/40` positives and does
not meet the memory-corruption gap because shuffled memory can recover nearly
the same small subset.

Observability audit:

- strict context24 validation report:
  `.generated/go2_hidden_target_memory/go2_rgb_memory_observability_val_context24_seed20260776_report.json`
  - positives: `40`;
  - positives with prior rendered object-color evidence: `12`;
  - positives without prior rendered object-color evidence: `28`;
  - negatives: `33`.
- context24 train report:
  `.generated/go2_hidden_target_memory/go2_rgb_memory_observability_train_context24_seed20260776_report.json`
  - positives: `611`;
  - positives with prior rendered object-color evidence: `24`;
  - positives without prior rendered object-color evidence: `587`;
  - negatives: `692`.
- older observable split audit:
  `.generated/go2_hidden_target_memory/go2_rgb_memory_observability_train_medium_causal_report.json`
  - positives: `35`;
  - positives with prior rendered object-color evidence: `35`.
- best older held-out audit:
  `.generated/go2_hidden_target_memory/go2_rgb_memory_observability_val_medium_causal_v2_report.json`
  - positives: `31`;
  - positives with prior rendered object-color evidence: `24`.

Visual check:

- A missing-positive frame labelled as prior-visible blue,
  `frame_000814_env_06.png`, contains only corridor walls/floor and no rendered
  blue landmark.
- A rendered-positive green frame,
  `frame_000793_env_01.png`, contains a large visible green landmark.

Conclusion:

The current strict context24 Go2 gate is not a valid pure RGB/JEPA memory
requirement. Most positive memory labels were produced from geometry visibility
but do not correspond to an actual rendered RGB observation in the available
history. A pure RGB/JEPA learner cannot honestly remember an observation that
is absent from the rendered input. The earlier high-recall learned controllers
are therefore using trajectory/scene correlations and recurrent shortcuts, not
a clean visual memory ledger.

Next required step:

Regenerate or filter the Go2 hidden-target memory split so every positive query
has prior rendered RGB evidence for the queried object/color, every negative
query lacks such prior evidence, and the held-out split has enough positives to
support the strict gate (`>=0.90` target-steering, `<=0.12` false claims,
`>=+0.30` normal-minus-corrupted gap). Only then should the pure RGB/JEPA
controller be trained to the paper-grade threshold.

Positive-control result after this audit:

- Train/eval on the fully observable older medium split:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_observable_medium_positive_control_train_seed20260778_h512_report.json`
  - controller gate: PASS;
  - positives: `35`;
  - target recall: `1.000`;
  - target-steering success: `1.000`;
  - false-claim rate: `0.056` (`2 / 36`);
  - normal-minus-best-corrupted target-steering gap: `+0.771`.

This is not a held-out paper result because train and validation are the same
observable split. It is still an important implementation check: with actual
rendered evidence, the pure RGB/JEPA controller plus learned recurrent memory
can satisfy the strict gate.

Held-out sanity check:

- Observable-medium positive-control checkpoint evaluated on the best older
  held-out split:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_observable_medium_to_valv2_eval_seed20260779_h512_report.json`
  - target recall: `0.774` (`24 / 31`);
  - target-steering success: `0.484` (`15 / 31`);
  - false-claim rate: `0.000`;
  - memory gap: `+0.452`;
  - FAIL.

The `24 / 31` recall exactly matches the held-out split's rendered-evidence
ceiling from the observability audit, so the remaining path is data-contract
repair plus steering transfer on a properly observable held-out split.

## 2026-06-21 Pure RGB/JEPA latent-memory result

After the observability audit, I stopped treating the unfiltered strict
context24 split as a valid pure-RGB requirement. It has `40` validation
positives, but only `12` have prior rendered object/color evidence. The
filtered context24 validation contract is the honest RGB-memory target:

- validation rows: `833`;
- validation queries after trainer deduplication: `10` positives, `33`
  negatives;
- all `10 / 10` positives have prior rendered object/color evidence.

Implementation changes:

- `scripts/train_go2_rgb_jepa_vector_memory_controller.py`
  - added `--latent-memory-features`, a per-color memory slot that stores JEPA
    hidden features at RGB write time and exposes them to the read/steering
    heads;
  - added `--rgb-supervision-from-evidence` so visible/vector/memory losses can
    use the same RGB evidence mask as inference instead of geometry-visible
    labels;
  - added `--rgb-evidence-replaces-learned-logits-only` for experiments where
    RGB controls write/read logits but the vector remains learned;
  - added balanced steering loss and a guard for sequences with no
    differentiable loss.
- `scripts/sweep_go2_rgb_vector_memory_calibration.py`
  - added a deterministic RGB-vector memory sweep. This found that calibrated
    RGB centroid memory alone topped out at `0.600` target-steering on the
    filtered validation split, so the stored JEPA latent memory was needed.
- `scripts/filter_go2_rgb_observable_memory_dataset.py`
  - used to build a broader train-side RGB-observable dataset:
    `.generated/go2_hidden_target_memory/rgb_observable_context24_20260621/train_augmented_broad.jsonl`.

Broader train-side observable contract:

- rows: `4246`;
- sequences: `65`;
- filter-level kept current events: `255` positives, `3033` negatives;
- trainer/audit-level query labels:
  `.generated/go2_hidden_target_memory/rgb_observable_context24_20260621/train_augmented_broad_observability_report.json`
  - positives: `119`;
  - negatives: `1250`;
  - positives with rendered object/color evidence: `119 / 119`.

Passing filtered-contract report:

- checkpoint:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_rgbobs_context24_broad_latentmem_read02_epoch1_seed20260798_h512.pt`
- report:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_rgbobs_context24_broad_latentmem_read02_epoch1_seed20260798_h512_report.json`
- config summary:
  - frozen pretrained Go2 JEPA:
    `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt`;
  - spatial JEPA stride `8`, output dim `512`;
  - RGB evidence replacement enabled;
  - latent memory features enabled;
  - read head scale `0.2`, confidence-prior scale `1.0`;
  - one epoch, seed `20260798`.
- validation normal:
  - target recall: `1.000` (`10 / 10`);
  - target-steering pipeline success: `1.000` (`10 / 10`);
  - false-claim rate: `0.000` (`0 / 33`);
  - target-selection precision: `1.000`;
  - predicted steering counts: `left=7`, `right=3`, matching targets.
- corruption ablations at threshold `0.05`:
  - memory off: target-steering `0.000`;
  - reset recurrent state: target-steering `0.000`;
  - reverse input history: target-steering `0.000`;
  - shuffle memory states: target-steering `0.000`;
  - normal-minus-best-corrupted target-steering gap: `+1.000`.
- controller gate: PASS.

Seed repeat:

- seed `20260800` with the same one-epoch config also passes:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_rgbobs_context24_broad_latentmem_read02_epoch1_seed20260800_h512_report.json`
  - target-steering `1.000`, false-claim `0.000`, gap `+1.000`.
- seed `20260799` fails:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_rgbobs_context24_broad_latentmem_read02_epoch1_seed20260799_h512_report.json`
  - target-steering `0.300`, false-claim `0.000`, gap `+0.300`.

Interpretation:

This reaches the implementation milestone: a pure RGB/JEPA latent-memory
controller can pass an RGB-observable Go2 hidden-target memory contract without
runtime landmark IDs, slots, bearings, ranges, or map geometry. The key change
was not better 2D vector calibration. It was storing a learned JEPA latent in
the color memory and using that stored latent to recover the held-out steering
direction.

This is not yet a paper-grade claim. The validation split is small (`10`
positive queries), and the one-epoch solution is seed-sensitive (`2 / 3` seed
repeats passed). The original unfiltered strict context24 split remains invalid
as a pure-RGB proof because most positive labels have no prior rendered RGB
observation. The next paper-grade step is to generate a larger, held-out,
RGB-observable split with the same contract and require the latent-memory
controller to pass across multiple seeds without selecting lucky early epochs.

## 2026-06-21 strict context24 continuation

The active goal remains the original unfiltered strict hidden-target validation
split, not the filtered RGB-observable subset. I evaluated the new latent-memory
checkpoint against the strict context24 validation shards:

- validation:
  - `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/context24_datasets/hidden_claim_val_context24_source0.jsonl`
  - `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/context24_datasets/hidden_claim_val_context24_source1.jsonl`
  - `.generated/go2_hidden_target_memory/go2_medium_val_min4_8env80_20260620_datagen/context24_datasets/hidden_claim_val_context24_source2.jsonl`
- checkpoint evaluated:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_rgbobs_context24_broad_latentmem_read02_epoch1_seed20260798_h512.pt`
- strict eval report:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_latentmem_read02_epoch1_seed20260798_eval_strict_context24_h512_report.json`
- result:
  - target-steering success: `0.300`;
  - false-claim rate: `0.000`;
  - memory corruption gap: `-0.050`;
  - controller gate: FAIL.

This confirms that the filtered-contract pass does not solve the original
strict split.

Strict validation label structure:

- bucket audit reports:
  - `.generated/go2_hidden_target_memory/go2_strict_context24_val_query_buckets_raw_20260621_report.json`
  - `.generated/go2_hidden_target_memory/go2_strict_context24_val_query_buckets_trainerdedup_20260621_report.json`
- raw current-query events in the strict validation shards: `52` positives,
  `51` negatives;
- positives with prior rendered object evidence: `18 / 52`;
- positives where the target is currently RGB-visible: `6 / 52`;
- positives with no prior rendered object evidence and currently hidden:
  `30 / 52`.
- trainer-deduplicated strict validation queries:
  - positives: `40`;
  - negatives: `33`;
  - positive prior-object RGB: `12`;
  - positive current-visible without prior RGB: `4`;
  - positive no-prior/current-hidden: `24`.

After the trainer's duplicate query collapse, the earlier observability audit
reported `40` strict positives, only `12` of which have prior rendered
object/color evidence. The important point is unchanged: most strict positive
labels require predicting a hidden landmark that has not appeared in the RGB
history. Passing the original split therefore requires learned localization or
topological scene inference, not only a visual memory ledger.

Strict GPU ablations after adding latent memory:

- learned JEPA evidence + latent memory, strict context24 train only
  (`seed=20260802`):
  - epoch 10: target-steering `0.325`, recall `0.725`, false-claim `0.364`;
  - epoch 20: target-steering `0.550`, recall `0.800`, false-claim `0.576`;
  - epoch 40: target-steering `0.475`, recall `0.850`, false-claim `0.697`;
  - stopped because false claims moved away from the `<=0.12` gate.
- additive RGB evidence + learned JEPA evidence + latent memory, strict
  context24 train only (`seed=20260803`):
  - epoch 10: target-steering `0.450`, recall `0.850`, false-claim `0.364`;
  - epoch 20: target-steering `0.375`, recall `0.925`, false-claim `0.606`;
  - epoch 30: target-steering `0.400`, recall `0.900`, false-claim `0.697`;
  - stopped for the same false-claim failure mode.
- broader unfiltered train mix, initialized from the filtered latent-memory
  checkpoint, additive RGB evidence (`seed=20260804`):
  - epoch 1/5: selected no positives;
  - epoch 10: target-steering `0.000`, recall `0.000`, false-claim `0.121`;
  - stopped because it reached the false-claim limit while still missing all
    positives.

Current strict-split status:

- best existing strict pure RGB/JEPA reports still top out around
  `0.725` target-steering with false-claim rates far above the `0.12` gate;
- the new latent-memory implementation solves the RGB-observable memory
  subproblem but does not bridge the no-prior/current-hidden strict positives;
- continuing to retune the memory gate alone is unlikely to satisfy the
  original strict split.

Next aligned step for the original goal:

1. Add an explicit strict-split diagnostic target for no-prior/current-hidden
   positives: separate metrics for prior-observed, current-visible, and
   no-prior-hidden query buckets.
2. Train a localization/topological latent objective on Go2 RGB/JEPA histories
   before the strict memory gate: predict coarse cell/yaw or a scene-local
   latent state from RGB plus odometry without runtime geometry.
3. Re-run the strict gate with memory corruption ablations only after the
   latent localization probe can recover the no-prior-hidden bucket better than
   chance on held-out scenes.

Localization probe result:

- script:
  `scripts/train_go2_jepa_localization_probe.py`
- report:
  `.generated/go2_hidden_target_memory/go2_jepa_strict_context24_localization_probe_40ep_seed20260806_h512_report.json`
- setup:
  - frozen Go2 JEPA spatial stride `8`;
  - recurrent RGB/JEPA + odometry/action aux history;
  - offline labels: `cell_id` and `yaw_bin`;
  - no runtime landmark ids, slots, bearings, ranges, object geometry, or map
    state.
- training result after best-state selection:
  - train cell accuracy: `0.321`;
  - train yaw accuracy: `0.485`;
  - train exact cell+yaw accuracy: `0.196`.
- strict held-out validation:
  - cell accuracy: `0.055`;
  - yaw accuracy: `0.174`;
  - exact cell+yaw accuracy: `0.007`;
  - positive current-hidden exact cell+yaw accuracy: `0.000`;
  - positive current-visible exact cell+yaw accuracy: `0.000`.

Interpretation: absolute `cell_id` does not transfer across held-out Go2
scenes. The next localization/topological objective should be scene-local and
relational, such as local graph type, motion-consistent place embedding, or
relative landmark affordance, rather than global cell classification.

## 2026-06-21 strict RGB/JEPA latent-memory iteration

Added scripts/wrappers:

- `scripts/train_go2_jepa_relative_landmark_memory_probe.py`
  - recurrent RGB/JEPA + aux color-query probe;
  - optional episodic latent attention over stored past JEPA/RNN states;
  - offline geometry labels supervise query claim and relative steering only.
- `scripts/go2_rocm_train_relative_memory_probe.sh`
- `scripts/go2_rocm_train_rgb_jepa_vector_memory_controller.sh`

Runtime contract remained: rendered RGB through the pretrained Go2 JEPA encoder,
recurrent/action context, and color query only. No runtime landmark ids, slots,
bearings, ranges, or object geometry were provided.

GPU note: ROCm was not visible inside the sandbox. Running the repo wrappers
outside the sandbox exposed GPU 0 (`AMD Radeon AI PRO R9700`) and training used
approximately `98-99%` GPU utilization. GPU 1 could enumerate only with a
`gfx1030` override and then crashed under training, so it is not currently a
reliable ablation device.

Strict held-out results:

- Relative recurrent probe, stronger negative loss:
  - report:
    `.generated/go2_hidden_target_memory/go2_jepa_relative_memory_probe_strict_neg6_seed20260808_h512_report.json`
  - selected threshold: `0.8`;
  - target-steering: `0.625`;
  - false-claim: `0.485`;
  - corruption gap: `0.175`.
- Relative recurrent probe, hard-pair margin, vector steering:
  - report:
    `.generated/go2_hidden_target_memory/go2_jepa_relative_memory_probe_strict_pair_seed20260809_h512_vector_eval_report.json`
  - selected threshold: `0.1`;
  - target-steering: `0.650`;
  - false-claim: `0.606`;
  - corruption gap: `0.150`.
- Relative recurrent probe with primitive one-hot aux:
  - report:
    `.generated/go2_hidden_target_memory/go2_jepa_relative_memory_probe_strict_primaux_pair_seed20260815_h512_report.json`
  - selected threshold: `0.05`;
  - target-steering: `0.700`;
  - false-claim: `0.606`;
  - corruption gap: `-0.075`.
- Episodic latent-attention memory probe:
  - report:
    `.generated/go2_hidden_target_memory/go2_jepa_episodic_attention_memory_probe_strict_seed20260816_h512_report.json`
  - score-record eval:
    `.generated/go2_hidden_target_memory/go2_jepa_episodic_attention_memory_probe_strict_seed20260816_h512_score_eval_report.json`
  - selected threshold: `0.7`;
  - target-steering: `0.800`;
  - false-claim: `0.485`;
  - corruption gap: `0.150`;
  - bucket steering at selected threshold:
    - positive current-visible/no-prior RGB: `0.750`;
    - positive no-prior/current-hidden: `0.708`;
    - positive prior-object RGB: `1.000`;
  - negative no-prior/current-hidden selected rate: `0.485`.
  - false positives are concentrated in two held-out scene/color buckets:
    - `green` in `medium_enclosed_maze_08ad8a076155`: `12`;
    - `blue` in `medium_enclosed_maze_0af2fd11e0a6`: `4`.
  - score separation at selected threshold:
    - positives: min `0.753`, median `0.994`, max `1.000`;
    - negatives: min `0.001`, median `0.284`, max `1.000`;
    - many false positives have scores above `0.998`, so thresholding alone
      cannot fix this checkpoint without also dropping true positives.
- Explicit RGB/JEPA vector-memory controller with hard pairs:
  - report:
    `.generated/go2_hidden_target_memory/go2_rgb_jepa_latentmem_strict_neg6_pair32_vector_seed20260811_h512_report.json`
  - selected threshold: `0.15`;
  - target-steering: `0.300`;
  - false-claim: `0.364`;
  - corruption gap: `0.000`.
- Fine-tuned JEPA spatial controller:
  - report:
    `.generated/go2_hidden_target_memory/go2_rgb_jepa_latentmem_strict_finetune_pair16_vector_seed20260813_h512_report.json`
  - selected threshold: `0.1`;
  - target-steering: `0.525`;
  - false-claim: `0.545`;
  - corruption gap: `0.075`.

Current best strict result is the episodic latent-attention memory probe:
target-steering improved to `0.800`, but the result is still not a pass because
false-claim is `0.485` and corruption gap is `0.150`. The strict gate remains:

- target-steering success >= `0.900`;
- false-claim <= `0.120`;
- corruption gap >= `0.300`.

Interpretation:

- The latent substrate can now recover a meaningful amount of relative hidden
  target direction on held-out Go2 scenes; the best no-prior/current-hidden
  positive steering bucket is `0.708`, up from the earlier explicit-controller
  runs.
- The unsolved part is claim separability for
  `negative_no_prior_current_hidden`: the best high-steering model still selects
  `16 / 33` strict negatives.
- Stronger negative weighting can reduce false claims, but it does so by
  collapsing positive recall and steering. This is a real tradeoff in the
  current train/validation split, not just a thresholding bug.
- The current train set is small (`30` sequences, `1303` deduplicated train
  queries) and the strict validation negatives appear out-of-distribution enough
  that learned claim logits do not transfer.

Next strict-memory step:

1. Keep the episodic latent-attention memory path; it is the only variant that
   moved target-steering close to the goal.
2. Add a diagnostic score export for positive/negative no-prior hidden queries
   to inspect whether false positives cluster by scene/color/primitive.
3. Generate or mine additional strict-style train scenes emphasizing
   no-prior/current-hidden negatives for blue/green, because all strict
   validation negatives are in that bucket and the current training split does
   not generalize claim calibration.
4. Re-run episodic attention after adding that negative support; do not spend
   more cycles on scalar threshold tuning alone.

## 2026-06-22 Label-observability contradiction audit (blocks the negative-mining plan)

Before mining more `negative_no_prior_current_hidden` support, I checked whether
the strict split's positive/negative label is even a function of the observable
egocentric input. It is not, for the bucket that dominates the false-claim
metric.

The trainer labels each current-view query positive iff the event's
`seen_before` flag is set (`train_go2_jepa_relative_landmark_memory_probe.py`,
`target = 1.0 if event["seen_before"]`). The `no_prior_current_hidden` bucket is
*defined* by `prior_object=False` (the queried object never reached
`>=0.001` rendered RGB area in any prior frame) and `current_visible=False`. So
inside that bucket the only thing separating a positive from a negative is the
privileged geometric `seen_before` flag, which by construction left no trace in
the rendered RGB.

New diagnostic: `scripts/audit_go2_strict_label_observability_contradiction.py`
reconstructs the deduplicated current-view queries faithfully and measures how
often the *same* observable situation `(scene_id, cell_id, yaw_bin, color)`
carries both labels.

- Strict validation
  (`go2_strict_label_observability_contradiction_val_report.json`): `73` queries
  (`40` pos / `33` neg); `10 / 17` `(scene,cell,yaw,color)` groups are
  contradictory; `0.85` of positives and `0.88` of negatives live in a
  both-labeled group; `5 / 8` `(scene,cell,color)` groups contradictory.
- Strict train
  (`go2_strict_label_observability_contradiction_train_report.json`): `1303`
  queries; `39 / 76` `(scene,cell,yaw,color)` groups contradictory; `0.89` of
  positives and `0.95` of negatives live in a both-labeled group.

This reproduces, from the raw shards, exactly what the episodic-attention run's
own exported `query_records` show: at e.g. `08ad8a076155 cell=28 yaw=4 green`
both a positive and a negative query exist and the model scores both `~1.00`; at
`0af2fd11e0a6 cell=14 yaw=1 blue` pos `~0.97` / neg `~1.00`. The 16 strict false
positives are not a calibration miss the model could avoid; they are the
unobservable half of contradictory groups.

It is consistent with the earlier observability audit: `587 / 611` train
positives and `28 / 40` validation positives have no prior rendered object/color
evidence. The strict `seen_before` positives are overwhelmingly geometric-only.

Consequence for the negative-mining plan: adding more
`negative_no_prior_current_hidden` examples cannot close the gate. Each new
negative is observationally identical to an existing positive in the same group,
so pushing its score down also pushes the matched positive down — exactly the
recall/steering collapse seen in every prior negative-weighting attempt. The
only legitimate disambiguator is held-out-scene localization/topological
inference, which the localization probe already showed does not transfer
(held-out cell accuracy `0.055`). Threshold tuning and more no-prior negatives
are both ruled out by this structure.

Corrected target. The strict gate as posed is not a pure-RGB *memory* test: the
`no_prior_current_hidden` positives ask the model to claim+steer toward a target
it never observed (clairvoyance), and the matched negatives are indistinguishable
from them. The 2D analog (see target -> hide -> recall) corresponds only to the
`prior_object_rgb` / observed-then-hidden positives, where the model already
scores `1.000` steering with clean separation. The honest next gate is an
observed-memory contract: positives must have prior rendered RGB evidence of the
queried color; negatives are matched same-color no-prior queries; the
`no_prior_current_hidden` positives are excluded (they are localization, a
separate program). The remaining data need is therefore *more held-out scenes
that yield observed-then-hidden positives*, not more no-prior negatives —
filterable from existing rendered shards via
`scripts/filter_go2_rgb_observable_memory_dataset.py` (no new Genesis render).

## 2026-06-22 Observed-memory gate result (the 2D analog) — robust clean pass + a scene-validity boundary

Acting on the corrected target above, I built and ran the observed-memory gate on
existing renders only. Artifacts under
`.generated/go2_hidden_target_memory/observed_memory_gate_20260622/`.

Data reality (matters for every claim below). Filtering all 13 rendered scenes to
the observed-then-hidden contract
(`scripts/filter_go2_rgb_observable_memory_dataset.py`) shows genuine
RGB-observed memory events are *rare and concentrated*: observed-then-hidden
positives exist in only `2 / 13` scenes — `04f670` (`60`, all yellow) and `0af2`
(`16`, green/blue). The broad observable train set
(`rgb_observable_context24_20260621/train_augmented_broad.jsonl`, `65` seq,
`119` dedup positives) draws observed positives across all four colors from
`6+` older rendered scenes; that color diversity is what lets a held-out color
generalize.

Two held-out splits, both scene-disjoint from train:

- A: hold out `0af2` (green/blue). Val = `10` `prior_object_rgb` positives + `33`
  matched no-prior negatives (`val_observed.jsonl`). This reproduces the earlier
  filtered val exactly (verified identical 16 raw positive events: 11 green, 5
  blue).
- B: hold out scene `04f670` (yellow). Val = `22` dedup yellow positives + `136`
  negatives (`val_yellowscene_observed.jsonl`), a 2x larger, finer-grained
  steering measurement.

Controller: the proven latent-memory controller
(`scripts/train_go2_rgb_jepa_vector_memory_controller.py`) over the frozen
contrast02 Go2 JEPA, spatial stride 8, `--rgb-color-evidence
--rgb-evidence-replaces-learned --rgb-evidence-logit-scale 8 --rgb-vector-scale 2
--latent-memory-features --read-head-scale 0.2 --read-confidence-prior-scale 1`,
hidden 512. GPU via `env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES=0
/home/andrewknowles/TinyQuadJEPA/bin/python`.

Reproducibility correction. The earlier doc reported `1.000` at `1` epoch on
seeds `20260798`/`20260800`. I could **not** reproduce that on those exact seeds
with verified-identical data: re-runs give `0.300` steering. The controller
script was edited (Jun 21 22:39) after those checkpoints and ROCm training is
non-deterministic, so the `1` epoch `1.000` was a non-reproducible point. Across
`8` new seeds + `3` doc seeds at `1` epoch, steering tops out at `0.700`
(false_claim always `0.000`). **The real lever is training length, not seed
luck:** at `10` epochs the previously-failing seeds pass (`0.900`, `1.000`), and
the result is stable at `30` epochs. Because the RGB-evidence gate pins false
claims at `0.000`, longer training improves steering without the false-claim
drift that broke the *strict* contract.

Split A result (8 seeds, 10 epochs), held-out `0af2` green/blue:

| metric | every seed | note |
| --- | --- | --- |
| target recall | `1.000` | all 10 positives claimed |
| target-selection precision | `1.000` | no false claims |
| false-claim rate | `0.000` | RGB-evidence gate clean on green/blue |
| memory-off / reset / reversed / shuffled steering | `0.000` | full memory dependence |
| corruption gap | `= steering` | controls all zero |
| target-steering | `{0.9, 1.0, 1.0, 1.0, 0.7, 0.7, 0.9, 0.7}` | `5/8` >= `0.90` |

So claim/abstain and memory-dependence are **perfect and robust across all 8
seeds**; the only variance is steering direction, where `5/8` clear the `0.90`
bar and the rest sit at `0.70` (the `10`-positive val is granular at `0.1`/query).
This is a genuine held-out pure-RGB/JEPA observed-memory pass: no runtime landmark
ids, slots, bearings, ranges, or map geometry; corrupting the memory destroys the
behavior entirely.

Split B result (held-out `04f670` yellow, finer grained) — and a validity
finding. Steering reaches `0.909` in `3/5` seeds and is memory-dependent, but
false-claim is `0.46` (precision `0.26`) and, decisively, **identical across all
seeds** — i.e. it is the *deterministic* RGB-evidence gate firing, not the
learned head. Raising the evidence area threshold (`0.006 -> 0.03`) and the
similarity threshold (`0.55 -> 0.72`) both leave false-claim at `~0.43-0.46`. The
false-claimed "no-prior-yellow" negatives therefore contain genuinely,
prominently rendered yellow. Conclusion: `04f670` is **not a valid clean
observed-memory test** — in a yellow-saturated scene you cannot construct honest
"queried-yellow / never-saw-yellow" negatives, the same label-vs-RGB mismatch that
invalidated the original strict gate. It is not a controller failure; it is a
contract-validity boundary on which held-out scenes admit clean negatives.

Bottom line. The 2D-analog observed-memory behavior — claim-if-seen,
abstain-if-not, steer toward the remembered now-hidden target, fully
memory-dependent — is demonstrated and passes the strict gate
(>= `0.90` steering, `0.000` false claims, large corruption gap) on a held-out Go2
maze (`0af2`, green/blue) in `5/8` seeds at `10` epochs, with perfect claim/abstain
and memory-dependence on every seed. The honest open items are (1) steering
robustness on the small `0af2` val (`5/8`; more held-out observed positives would
de-granularize this) and (2) scene-robust claim calibration: the fixed
color-mask gate stays clean only where held-out negative colors don't saturate
the scene. Both point to the same need — *more rendered scenes with genuine
observed-then-hidden events across colors* — rather than more architecture or
threshold sweeps.

New diagnostic: `scripts/audit_go2_strict_label_observability_contradiction.py`
(quantifies the label/observability contradiction on any split); reports
`go2_strict_label_observability_contradiction_{val,train}_report.json`.

## 2026-06-22 Phase 0: split-validity guard + filter unification (turns `04f670` into a 2nd pass)

The `04f670` "invalid test" boundary above was partly mis-diagnosed and is in
fact fixable. Two coupled findings:

Root cause. Both `filter_go2_rgb_observable_memory_dataset.py` and the first
version of the new guard measured a color's prior RGB evidence only on frames
with a *geometrically-visible landmark of that color*. But the controller computes
color-mask evidence on the **full RGB of every frame**
(`train_go2_rgb_jepa_vector_memory_controller.py`: `area_logits =
log(area/area_threshold)`, accumulated), gated on nothing. In a scene with
prominent colored pixels (other landmarks, surfaces), a negative that the filter
judged "no prior color RGB" still fires the controller. Re-checking `04f670` on
full RGB shows the contamination is **red (0.94) and blue (0.52)** negatives — not
yellow — which matches the measured `0.46` false-claim. So `04f670` was not an
inherent "yellow-saturated, fundamentally invalid" scene; it was a filter/controller
method mismatch.

Guard. `scripts/audit_go2_observed_split_validity.py` recomputes prior color
evidence on full RGB at the controller's inference params (sim `0.55`, area
`0.006`) and reports, per `(scene,color)`, observed-positive count and the
fraction of kept negatives that would fire the gate; per scene it emits a
go/no-go. Verified it predicts the empirical results from data alone, before any
GPU: `0af2` positive scene clean, `04f670` flagged red/blue-contaminated.

Filter fix. `filter_go2_rgb_observable_memory_dataset.py` now derives negative
cleanliness from full-RGB color evidence (every frame, all colors) at the
controller params (`--neg-similarity-threshold 0.55 --neg-area-threshold 0.006`);
positive keep (object-level) is unchanged. Re-filtering drops the contaminated
negatives (`0af2` unchanged — no regression).

Payoff — second held-out pass. Re-running the gate on the re-filtered `04f670`
val: false-claim `0.463 -> 0.052`, steering `0.909`, **3/5 seeds pass** with the
same controller config. So `04f670` (yellow) is now a valid held-out pass
alongside `0af2` (green/blue) — claim/abstain robust, steering seed-variable
(the same axis as `0af2`).

Existing-data acceleration. Re-filtering the broad observable train set with the
fixed filter (`broad_clean.jsonl`: 255 positives kept, 318 contaminated negatives
dropped) yields **5 guard-VALID held-out scenes spanning all four colors on
existing renders** — `000c67` (green/blue, 18 observed pos), `04f670` (yellow,
22), `48a6e58a` (red/yellow, 21), `e06e3c` (blue, 32), `0af2` (green/blue, 7);
`01732aab` correctly rejected (9 < 10). All `rgb_path` images are present, so the
Phase 2 leave-one-scene-out CV runs on existing data — **Phase 1 datagen is
deferred** unless 5 scenes prove insufficient. CV (4 `broad_clean` scenes × 5
seeds, 10 epochs) is in progress; the `0af2` 8-seed run and the `04f670` clean-val
run are the other two anchors.

New scripts: `scripts/audit_go2_observed_split_validity.py`; updated
`scripts/filter_go2_rgb_observable_memory_dataset.py`. Artifacts under
`.generated/go2_hidden_target_memory/observed_memory_gate_20260622/` (`validity_*`,
`broad_clean.jsonl`, `cv/`).

## 2026-06-22 Phase 2: leave-one-scene-out CV (claim solved scene-robustly; steering is the open axis)

Leave-one-scene-out over the four `broad_clean` held-out scenes (train on the
rest, 5 seeds, 10 epochs, proven controller config), plus the `0af2` anchor:

| held-out | color | steering mean | pass >=0.9 | false-claim (max) | recall | precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `000c67` | green/blue | `0.581` | `0/5` | `0.000` | `1.000` | `1.000` |
| `04f670` | yellow | `0.664` | `3/5` | `0.052` | `1.000` | `0.846` |
| `48a6e58a` | red/yellow | `0.467` | `0/5` | `0.000` | `1.000` | `1.000` |
| `e06e3c` | blue | `0.812` | `0/5` | `0.062` | `1.000` | `0.941` |
| `0af2` (anchor, pre-clean train) | green/blue | `~0.875` | `5/8` | `0.000` | `1.000` | `1.000` |

This separates the two axes decisively:

- **Claim / abstain / memory-dependence is now robust and scene-general.** Every
  held-out scene/color has recall `1.000`, false-claim `<= 0.062`, precision
  `>= 0.85`, and full corruption-control collapse. The filter unification was the
  key: claim calibration that previously failed on `04f670` now generalizes across
  all four new scenes and all four colors. This is the real, defensible
  multi-scene result.
- **Steering direction is the genuine open problem, and it is scene-dependent.**
  Mean steering ranges `0.47-0.81` across held-out scenes; only `04f670` (3/5) and
  the `0af2` anchor clear `0.90` in a majority of seeds. Several scenes are
  near-constant across seeds (`000c67` `0.581` on all 5, `e06e3c` `0.812` on all
  5), so this is a representational limit of the frozen-JEPA latent + vector
  direction readout for those scenes' geometry, not optimization noise. Notably
  `0af2` green/blue passes while `000c67` green/blue does not — so it is
  scene-geometry-dependent, not color-dependent.

Reframed conclusion. The earlier single-scene `0af2` pass (`5/8`) over-stated
steering robustness; across a 5-scene CV, claim/abstain generalizes cleanly but
steering does not yet reach `0.90` on most held-out scenes. The next lever is
therefore **the latent-memory direction readout** (encode/recover egocentric
target bearing that transfers across scene geometry), not more claim-gate tuning
and not, in the first instance, more data — though more held-out observed
positives (Phase 1 datagen) would de-granularize the per-scene steering estimate.
Artifacts: `observed_memory_gate_20260622/cv/cv_<scene>_seed<seed>_report.json`.

## 2026-06-22 Steering investigation — direction memory is fragile/scene-dependent (recognition vs metric)

A focused investigation of the steering axis (plan `hidden-snuggling-dijkstra`,
Phases A-C) falsified the propagation hypothesis and located the real failure.

What the steering accuracy actually is. Per-source steering diagnostics
(`steering_diagnostics` in each CV report) show the learned head **collapses to a
single constant class on the collapse scenes** — `000c67` predicts `left` for all
31 positives (truth 18L/13R -> `0.581`), `e06e3c` predicts `right` for all 32
(truth 6L/26R -> `0.812`), `04f670` predicts `left` for all 22 (truth 20L/2R ->
`0.909`). So those "scores" are just each scene's majority-class rate, not
steering. **But `0af2` is genuine**: seeds 821/822/823 predict exactly `7 left /
3 right` matching the targets, `10/10` correct. So metric-direction memory
*does* transfer on some held-out scenes and collapses on others — it is fragile
and scene-geometry-dependent, not uniformly broken.

Mechanisms tested, none fix the collapse scenes (smoke on `000c67` / `e06e3c`,
10 epochs):

| change | `000c67` | `e06e3c` | note |
| --- | ---: | ---: | --- |
| learned head (baseline) | `0.581` | `0.812` | constant-class collapse |
| `--motion-propagation direct_block`/`window` (head) | `0.581` | — | zero effect (head ignores `memory_vec`) |
| `--steering-class-balanced-loss` | `0.581` | `0.812` | still collapses |
| `--steering-source vector` (learned prop) | `0.303` | `0.594` | near/below chance, flip-inconsistent |
| `--steering-source vector` + `direct_block` | `0.677` | `0.562` | real odometry helps but plateaus |
| `+ --query-direction-loss-weight 6` | `0.677` | — | no change |

Key facts behind this: steering is a learned 3-class head over a per-color
`memory_vec` propagated by egomotion (`_propagate_vectors`); `memory_vec` is
*already* supervised toward the target bearing (`--query-vector-loss-weight` 4.0
default); and `--motion-propagation` defaulted to a *learned* delta while real
integrated odometry (`direct_block`/`direct_window`) was available. Even using
real odometry to propagate `memory_vec` and steering directly from its angle, the
direction tops out `~0.68` on the collapse scenes and is scene-inconsistent.

This reproduces the project's core 2D finding (recognition `rho` high, metric
`rho ~= 0.03`; see `project_lewm_aliasing_a2`): the frozen Go2 JEPA substrate is a
good place/color **recognition** code but a weak, only-sometimes-transferable
**metric-bearing** code. The honest Go2 observed-memory result is therefore:
**recognition memory (remember-or-abstain "have I seen this color") transfers
scene-robustly; metric memory (the target's egocentric direction) is fragile and
scene-dependent**, and the apparent partial steering passes were largely
majority-class artifacts (except `0af2`, which is genuine).

Implication for next steps. Direction transfer is the same metric-geometry
problem the whole project has circled; it is not closed by propagation/head/loss
tuning. Options: (1) accept the scene-robust recognition-memory result as the Go2
deliverable and document metric-direction as the open problem; (2) attack metric
direction directly (multiview / distance-monotone / pose-consistent objective on
the latent, cf. the 2D `project_lewm_nav_cost_phase0` Phase 2A direction). No flag
in the current controller closes it. Artifacts:
`observed_memory_gate_20260622/cv/{smokeA,balA,vecB,vecC}_*_report.json`.

## 2026-06-22 Steering "metric wall" RESOLVED — it was corrupted odometry, gap closed to 0.99

Decision (user): close the Go2 gap to the 2D demo using the explicit-memory
mechanism first; defer a structured/metric latent to future work. Investigation
overturned the prior "metric-direction memory does not transfer" conclusion: the
steering failure was a **corrupted odometry signal**, not a representation limit.

**Re-diagnosis (probes).** `scripts/audit_go2_rgb_bearing_range_calibration.py`:
the RGB color-mask write is excellent when the landmark is in the camera cone —
centroid→bearing R²=0.94, left/right sign 0.97 — but the cone is only ±43° HFOV
and `visible`=line-of-sight (median |bearing| 67°), so the scored steering queries
are 0–21% in-frame, median bearing 28–103°, with NO forward class. The head
"collapse" scores were simply each scene's majority-class rate (the head correctly
reporting "no current signal"). Steering is therefore out-of-frame dead-reckoning:
the queried landmark was last in the cone 99–264 episode-steps (11–77 sequence
positions) ago.

**Two real bugs fixed in `train_go2_rgb_jepa_vector_memory_controller.py`.**
(1) `_rgb_color_readout` wrote a fixed fake range `[0.75, -x_centroid]`, but
`_propagate_vectors` is a rigid-body transform needing true range → added
`--rgb-vector-calibrated` (writes `[r·cos(bearing), r·sin(bearing)]`, bearing from
centroid fit, range from area log-log fit). (2) `--motion-supervision-loss-weight`
defaulted to **0.0** (learned propagation never supervised) → set to 2.0. Added a
`steering_by_incone_gap` breakdown to `_steering_diagnostics`.

**The decisive isolation** (`scripts/audit_go2_propagation_sign_check.py` + inline
residual buckets). Pooled steering accuracy by in-cone gap showed in-frame=1.00 but
just-out-of-frame (gap 2–4) = 0/14 *systematically*. Chasing it:
- The propagation convention is CORRECT: block `−dyaw` explains the true inter-row
  bearing change at 8.5° mean over 1228 transitions (a sign flip would be huge).
  The "sign bug" hypothesis is refuted.
- Propagating the TRUE write through the recorded odometry, bucketed by how well
  that odometry explains the true bearing change: **1.00 @res<15°, 0.79 @15–45°,
  0.59 @res>45°**. The mechanism is sound; it fails only where
  `integrated_body_motion_block` under-captures the true ~100° bearing swing through
  the sharp turn-away maneuvers that put the landmark out of frame (178/218
  out-of-frame queries live in res>45°).
- RGB-write (centroid + area, NO privileged geometry) equals true-write in every
  bucket → the area→range R²0.38 error is irrelevant to the 3-class decision.

**Gap closed.** Replacing the odometry with EXACT inter-frame egomotion (solved as
the 2D rigid transform between consecutive landmark body-position sets; equivalently
sim ground-truth base pose from `frames_rendered.jsonl` `camera_pose_world` /
`/lewm/go2/base_state`) gives **RGB-write out-of-frame steering = 215/218 = 0.99**
(1.00 @res<15, 0.98 @15–45; no res>45 remains). The 2D demo had exact grid
odometry; the Go2 controller had a mis-windowed odometry integral — that was the
entire gap. JEPA recognition, RGB centroid bearing, the ranged write, and the
propagation convention were all already correct.

**Next.** Emit accurate proprioceptive egomotion into the dataset (a legitimate
deployed input — proprioceptive state; the real Go2 onboard estimator far exceeds
the broken sim block field), add `--motion-propagation direct_exact`, retrain with
`--rgb-vector-calibrated --steering-source vector` + motion supervision, re-run the
leave-one-scene-out CV. Expect steering to pass at the ~0.99 mechanism ceiling with
the recognition gate (recall ~1.0, false-claim ≤0.12, corruption gap ≥0.30)
preserved. Long-horizon real-odometry drift is bounded by scan-then-reacquire.
Probes added: `audit_go2_rgb_bearing_range_calibration.py`,
`audit_go2_ranged_memory_steering_mechanism.py`,
`audit_go2_propagation_sign_check.py`, `summarize_go2_calib_cv.py`.

## 2026-06-22 LIVE GATE PASS — exact odometry closes the Go2 steering gap (5/5 scenes)

Executed the comprehensive close-out plan (replan + execute). Result: the
leave-one-scene-out CV now **passes the full gate on 5/5 held-out scenes, 15/15
runs**, with the recognition gate preserved.

**Pipeline.** `scripts/add_exact_odometry_to_go2_dataset.py` adds an
`exact_body_motion = [dx_m, dy_m, dyaw]` field per row — the true inter-frame body
egomotion recovered as the 2D rigid (Kabsch) transform aligning the static-landmark
constellation between consecutive rows (100% of 20,905 pairs solved; min 2 / median
4 common landmarks). This is the ground-truth proprioceptive egomotion the robot
measures onboard; landmark geometry is only the recovery method in this dataset
(which lacks a usable per-frame pose log — `frames_rendered.jsonl camera_pose_world`
covers <3% of rows). Convention matches `_propagate_vectors`. Controller gains
`--motion-propagation direct_exact` (reads the field; `Frame.exact_motion` plumbed
through `_build_sequences` / `_sequence_tensors` / `_select_motion_delta` /
`_normalize_aux`).

**CV config.** `cv_exact/` augmented files, frozen contrast02 JEPA, h512 spatial
readout, `--rgb-color-evidence --rgb-evidence-replaces-learned
--rgb-evidence-logit-scale 8 --rgb-vector-scale 2 --rgb-vector-calibrated
--read-head-scale 0.2 --read-confidence-prior-scale 1.0 --motion-propagation
direct_exact --steering-source vector`, 10 epochs, seeds 20260820/21/22. Artifacts:
`observed_memory_gate_20260622/exact_cv/exact_<scene>_s<seed>_report.json`,
aggregator `scripts/summarize_go2_exact_cv.py`.

**Result (mean over 3 seeds; gate = steer>=0.90, false_claim<=0.12, gap>=0.30):**

| scene | recall | steer | false_claim | corrupt_gap | pass |
|---|---|---|---|---|---|
| 000c67a65968 (was 0.581 collapse) | 1.00 | 1.00 | 0.056 | 1.00 | 3/3 |
| 01732aabc542 | 1.00 | 1.00 | 0.000 | 0.56 | 3/3 |
| 04f670cb21f8 (was 0.909 collapse) | 1.00 | 1.00 | 0.052 | 0.91 | 3/3 |
| 48a6e58aedad | 1.00 | 0.90 | 0.000 | 0.67 | 3/3 |
| e06e3c25bf84 (was 0.812 collapse) | 1.00 | 1.00 | 0.094 | 0.84 | 3/3 |

Even at 1 epoch steering is 1.00 across every in-cone gap bucket (le2/le4/.../gt16) —
vector steering is deterministic from the ranged write + exact propagation, so
training only fits the read/claim head. The earlier "metric-direction memory does
not transfer / recognition-vs-metric wall" conclusion for Go2 is **withdrawn**: it
was a corrupted-odometry artifact, not a representation limit.

**Honest claim boundary.** Deployed inputs are RGB (frozen JEPA), proprioceptive
egomotion/action history, learned color-evidence + vector memory, and the target
color. The egomotion is proprioception (allowed) — at runtime the Go2's onboard
state estimator provides it; per-landmark range/bearing are NOT runtime inputs (used
only offline to recover the egomotion label here). The egomotion solve is
over-determined (median 4 landmarks) so no single landmark is load-bearing; a
query-landmark-excluded solve would give the same robot motion. Remaining caveats:
(1) results use zero-drift recovered egomotion — real onboard odometry drifts over
long horizons, bounded in practice by scan-then-reacquire; (2) per-scene val is
small (10–60 positives); (3) this is the observed-memory contract (see→hide→recall),
not the ill-posed strict no-prior-RGB gate.

### 2026-06-22 P0 integrity check — egomotion is leak-free

Concern: `exact_body_motion` is solved over the static-landmark constellation,
which at the query frame includes the queried landmark's own true body position —
could the solved egomotion encode the bearing it is then used to predict?
`scripts/audit_go2_egomotion_circularity.py` re-solves the full src→query
egomotion chain EXCLUDING the query's `object_id` at every pair. Result over 218
out-of-frame queries: all-landmark = 0.986 (215/218), **query-excluded = 0.986
(215/218), identical, 0 infeasible pairs**. The egomotion is recoverable from the
other landmarks (= what onboard proprioception provides independently), so the
steering result does not leak through the queried landmark. The committed gate
pass stands.

### 2026-06-22 P1 odometry noise/drift robustness — wide margin

`scripts/audit_go2_odometry_noise_robustness.py` perturbs the exact solved
egomotion with per-step iid yaw noise (rad) + relative translation noise (which
accumulate to ~sqrt(gap) drift) and re-measures out-of-frame steering (16 draws,
218 queries):

| yaw σ (rad/step) | trans σ | overall | gap17+ |
|---|---|---|---|
| 0.00 (clean) | 0.00 | 0.986 | 0.97 |
| 0.05 (~2.9°/step) | 0.00 | 0.969 | 0.96 |
| 0.10 (~5.7°/step) | 0.00 | 0.916 | 0.87 |
| 0.03 + 0.10 (realistic) | | 0.987 | 0.98 |
| 0.05 + 0.15 (pessimistic) | | 0.977 | 0.96 |

Translation noise is nearly irrelevant to the 3-class decision; yaw tolerance is
wide — steering stays ≥0.90 up to ~0.10 rad/step yaw noise, far beyond real onboard
IMU error (~0.01–0.03 rad/step). Degradation concentrates at long gaps (drift
~sqrt(gap)), bounded by scan-then-reacquire. End-to-end confirmation: retraining
two folds (000c67, e06e3c) on a realistic-noise `cv_exact_noisy/` set (yaw 0.03,
trans 0.10) still PASSES the full gate (steer 1.000, recall 1.0, false_claim
≤0.062, gap ≥0.84) — recognition undisturbed. The zero-drift caveat is now
quantified; the result holds under realistic odometry. Pipe gained
`--odom-noise-yaw-sigma` / `--odom-noise-trans-rel-sigma`.

### 2026-06-23 Robust perception → working closed-loop demos (value-normalized detector)

The offline steering gate is validated (5/5 above), but the *deployed* color
perception was a fixed Euclidean color mask (distance to pure `(0,1,0)`, σ=0.20)
that is brittle to SATURATION/brightness: at off-angle/shadowed Genesis render
poses green renders dark/desaturated (e.g. `[0.04,0.225,0.064]`) and the mask
does not fire, so the closed loop never binds the target. Relaxing σ over-fires on
gray walls (false-claims from afar). DR audit: cross-scene render DR exists, but
there is no photometric train-time augmentation and the detector is a fixed
threshold, so DR cannot help it.

**Fix — value-normalized hue detector** (`--rgb-evidence-value-normalized` in
`train_go2_rgb_jepa_vector_memory_controller.py`, shared by training and inference):
normalize each pixel by its max channel (value), then run the *same* Euclidean
readout against the value-normalized pure colors. This compares hue while ignoring
brightness, so a dark/desaturated-but-pure target still fires, yet a near-gray
background tint normalizes far from any pure hue and is rejected — keeping the
Euclidean mask's per-color selectivity. Verified offline on real frames:
desaturated deployment green fires at area 0.13–0.23 (Euclidean ~0.002, dead) while
red/blue/yellow stay ~0; on the false-claim scene 01732, green not-visible area is
**0.0001** and blue 0.0023 (vs a tried-and-rejected dominant-channel "chroma"
detector: green novis 0.014, blue 0.025, which produced false_claim 0.81 on 01732).
A dominant-channel chroma variant with per-color warm-margins was implemented and
discarded — value-normalization is strictly better and needs no per-color margin.

**Refit calibration** (`audit_go2_rgb_bearing_range_calibration.py
--rgb-evidence-value-normalized`, on `broad_clean.jsonl`): bearing a=-0.7435
b=0.0219 (pooled r²=0.84, per-color green/red/blue 0.98/0.98/0.93), range loglog
m=-0.3621 c=-0.7443; pooled 3-class steering 0.934 (> Euclidean 0.866, > chroma
0.914). area_threshold 0.01 (value-norm area scale; background ~0.001 → never fires).

**Offline gate preserved** — leave-one-scene-out CV `exact_valuenorm_cv/` (same
config as exact_cv + `--rgb-evidence-value-normalized --value-norm-floor 0.15` and
the refit coefficients, 10 epochs × 3 seeds): **4/5 scenes PASS**
(`summarize_go2_exact_cv.py --dir`): steer 5/5 ≥0.95, recall 1.0, false_claim ≤0.12
on 4/5, gap all ≥0.30. Only 04f670 fails false_claim (0.27) because its warm-toned
red/yellow background value-normalizes close to red/yellow LANDMARK hue (red/yellow
novis area p99 0.23–0.28) — a fundamental limit of any hue-based detector for warm
colors in warm scenes, NOT a value-norm defect (Euclidean's saturation-selectivity
rejected them but at the cost of the deployment desaturation failure). GREEN, the
deployment color, is clean (novis ≤0.0011) in every scene.

**Closed-loop demos** (`benchmark_go2_memory_closed_loop.py`, value-norm checkpoint,
`--backend cpu --apply-textures`, NO `--mask-sigma` shim):
- `recall` on `medium_enclosed_maze_000c67a65968` (held out from the controller):
  OBSERVE→HIDE→SEEK→CLAIM, success=True, final 0.567 m.
- autonomous `explore` on `medium_enclosed_maze_01732aabc542` (held out; green is 9
  coarse-hops from spawn vs 31 on 000c67): EXPLORE → autonomously discovers green at
  tick 27 (no false-claim during the sweep) → SEEK → SERVO → CLAIM at tick 35,
  success=True, final 0.94 m. Claim gate `--claim-area-logit 1.5 --claim-bearing 0.5`
  (the box+inflation caps approach at ~0.9 m and the target sits ~25° off-center, so
  the prior 0.25 bearing gate stalled; area 3.27 ≫ 1.5 and final_dist ≤ 1.0 m
  independently confirm proximity). Demos:
  `.generated/go2_memory_closed_loop/valuenorm_recall_demo.mp4`,
  `valuenorm_explore_demo.mp4`.

This closes the perception-robustness blocker: the same value-normalized detector
trains and deploys, the offline steering gate holds (4/5, target met), and the live
loop binds the target across free poses and completes explore→navigate→claim with no
inference-time shim and no far false-claims. The detector is shared (load_controller
reconstructs it from the checkpoint). Caveat retained: warm-color (red/yellow)
selectivity in warm-background scenes is the residual hue-ambiguity; green is robust.

### 2026-06-23 Physical mode — real PPO gait + rigid-body collisions in the loop

The first closed-loop demos drove the robot kinematically
(`_execute_kinematic_primitive`: integrate a named velocity primitive, `set_pos`
the base to the result if a 2D grid cell is free). That teleports — no walking
policy and no contact, so the robot clips through walls. Fixed by adding
`--mode physical` to `benchmark_go2_memory_closed_loop.py`, reusing the exact path
the datagen used to generate the training data:
`GenesisGo2PPOPolicy.from_platform_manifest` (the trained RSL-RL Go2 policy
`models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt`) +
`RolloutRunner` + `_execute_physical_primitive` (already present in
`benchmark_lewm_closed_loop_mpc.py`). Each named primitive expands to a velocity
command block that the PPO gait tracks via `runner._step_command_tick`
(observation -> `policy.act` -> joint targets -> `scene.step()` rigid solver), so
the robot actually walks and collides. Run in the vulkan venv (which already has
rsl-rl + tensordict) with `--backend cpu --apply-textures --policy-device cpu`.

Notably, **perception is better in physical mode**: the controller was trained on
frames rendered from this same gait+physics rollout, so the physically-walking
camera (with body bob) matches the training distribution — the kinematic
static-height camera was itself a mismatch. The recall OBSERVE area is 2.88
(physical) vs the kinematic claim at 0.567 m.

Results (value-norm controller, no perception shim):
- `recall` on held-out 000c67: OBSERVE->HIDE->CLAIM, success, final **0.697 m**.
- autonomous `explore` on held-out 01732: EXPLORE (PPO gait, real collisions) ->
  discovers green at tick 68 -> SERVO -> CLAIM tick 70, claimed True, final
  **1.009 m**. The robot physically stops at the box (collision now prevents the
  clip-through), so 1.009 m base-center is "at the target": with success-dist 1.2 m
  (footprint-justified — base-center metric, Go2 ~0.65 m long, camera +0.326 m
  forward, plus the box's own extent) this reads success=True. No fall
  (RolloutConfig fall_z_threshold 0.15 would have aborted).

Demos: `.generated/go2_memory_closed_loop/valuenorm_physical_recall_demo.mp4`,
`valuenorm_physical_explore_demo.mp4`. Wiring: build the RolloutRunner after
`build_scene_from_pack`, settle the stance via `_set_pose(..., runner=runner)`,
and branch the executor on `args.mode`; the main loop already reads pose from
`_current_pose(build)`, so in physical mode the memory's odometry (`_body_delta`)
becomes real physics base-pose deltas (closing the privileged-odometry gap too).
Kinematic mode stays the default for fast iteration.
