# Prospective reactive nominal-route native comparison V1

Run `scripts/run_go2_reactive_nominal_maze_pilot_v1.py --prefix-result-sha256 HASH`
once at exclusive `go2_reactive_nominal_maze_pilot_v1_attempt_001` only after
the full declared storage/RAM envelope is available. Bind completed reactive
prefix `0653ca664495f8113296e0958bd2a1d2a3b2b6e3da536516e7efa70f17f8394a`,
waypoint native `d5ba1136c969cd6591028f435a93f0e35a4dae7b9cbf217524887b459271ace8`,
and waypoint readout `1bb9ae412452fd68a99fa827ad304d9201f5dfdc2b7bb24f57e003390ba7799c`,
their artifacts and compatible frozen source/native dependencies before/after.
Also retain the reactive prefix's complete predecessor input bindings.

The optional `--preflight-only` mode verifies those inputs and the complete
source union, reports hardware/envelope availability, and exits without creating
an output root or native scene. It does not bypass the normal launch's resource
checks or count as an experiment. A later real launch revalidates all inputs.

Collect one fresh CPU scene on the same development maze 0 with the same physical
Go2, pretrained PPO locomotion policy, gains, physics, primitive first commands,
paired public RGB-D/body/gyro and auxiliary depth, visibility requirements and
raw native guards. The high-level controller is `ReactiveNominalRoundTripController`:
no world-model state is loaded, no candidate future outcome is evaluated, and no
learned forecast feasibility filter is reused. Original model artifact bindings
retained transitively are provenance, not baseline policy inputs. Persistent
observer/map and exact outbound/return mission, three warm-up ticks, shared 3,000
navigation ticks, arrival dwell and ten zero drain commands match the learned run.

The reactive rule and its explicit unknown connector policy are frozen by the
prefix. It turns to the observed route target, then drives forward subject to
current footprint and known-obstacle connector checks. These geometry gates differ
from the predictive policy's learned future surface/path checks and nominal-reentry
exception. This is a matched scene/sensor/actuator/mission whole-controller
comparison, not a claim that predictive ranking is the only changed mechanism.
Persistent memory is present in both; a separate memory ablation remains necessary.

The native collector is a narrow source copy changing only high-level controller
interface, truthful reactive role/status labels and explicit resource identity.
The fresh-controller raw audit retains original sensor reconstruction, actuator
slew/command verification, visibility, native pose, arrival/backtracking and safety
checks; it omits world-model replay/state checks because the baseline has no such
model. Nine scope and physical/public-prefix tests passed in 1.73 s.

Require exact physical/public/observer/map/mission agreement with the completed
recovery attempt through command 7 (eight observations), where baseline forward
differs from learned left arc. Independently require agreement with the executed-
waypoint native attempt through command 3 (four observations), where baseline
forward differs from waypoint-policy left arc. Both requested-command differences
are bound by actual completed input tapes and prospective baseline prefix output.
Do not infer any unexecuted outcome. Preserve audit/prefix/infrastructure failures.

Assess hardware immediately before launch. Use the same 10 GiB collection plus
1 GiB persistence headroom above the unchanged 40 GiB artifact reserve, and require
32 GiB available RAM. One ordered scene/process; no new training or GPU request.
Record CPU/RAM/storage throughout. Resource admission is not an enforced OS RAM
limit. Do not shrink the collection envelope to force this comparison into current
free space or delete evidence to make room. Any separately proposed cache cleanup
requires its own explicit user authorization and revalidation.

Record actual resource stops and every failed navigation outcome. Navigation
success requires the complete native arrival, return/backtracking, visibility and
raw sensor/command gates. This reused maze adds zero independent layouts. Paused
physics and ideal sensors remain explicit; no hardware or real-time qualification
or causal JEPA/RGB/planning/memory advantage is inferred from this pilot alone.
