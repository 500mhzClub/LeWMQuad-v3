# Online learned-model goal probe: replay verified, 0/2 goals reached

The learned model now runs inside a native Go2 control loop. Both fresh
episodes completed collection and raw audit, including exact replay of all
sensor-driven decisions and all49 actual six-candidate model selections.
Neither reached the1.2m downstream goal. These known mirrored development
panels are not independent mazes. The overall navigation goal remains active.

The first case lost visual support; the second maintained tracking but only
turned. These are separate failures. No checkpoint, score or observer threshold
was changed during the attempt. Both attempts and their failures are preserved.

| Case | Online selections | Terminal | Native final goal distance |
| --- | --- | --- | --- |
| cluster00 left opening, family_episode_052 | 1 right turn | Visual support lost at observation5; ten zero ticks drained | 1.209431m |
| cluster00 right opening, family_episode_039 | 30 right turns,18 left turns | All240 navigation ticks exhausted; ten zero ticks drained | 1.145692m |

Neither case had a physical contact, native guard stop or acquisition failure.
The closest native goal distances were1.199770m and1.139180m. The second case
began alternating left/right every replan from tick68 onward. It never selected
a translation candidate. This establishes an online integration failure, not
evidence of useful predictive navigation or an RGB/JEPA advantage.

The controller used the fixed first preregistered seed2026091101, full-input
JEPA checkpoint from the completed independent-pulse parallel study. That
historical model was trained on short pulses. Its predictions and uncalibrated
contact scores have a known domain gap to four-second plans and moving replans.
The existing distance-reduction-minus-contact-cost selector and1.2m contact
penalty were frozen before either new outcome. At the first left-case selection,
right turn had utility−0.060757m versus hold−0.121816m, right arc−0.119802m,
left arc−0.342156m and forward−0.839365m. The audit reproduces these forecasts
and their selected command exactly; it does not validate their accuracy.

## Tracking diagnosis

A separate source-bound readout reconstructs all four rejected reference pairs
at left-case frame5. The unchanged original registration rejects each pair.
The failure is specifically image-grid coverage: the21–23 accepted inliers
occupy only3 of the required6 current-frame grid cells. Inlier fractions are
0.913–0.958, above the0.6 requirement; fitted reference displacements are
1.44–3.22mm, below the3m reference limit. The previous-frame pair has21/22
inliers,5 reference cells,3 current cells and1.994mm residual RMS.

This decomposition grants no pose and does not weaken the grid gate. It gives
a concrete target for a new feature-support design. The source currently uses
600 global SIFT features. A separately named, bounded feature-selection change
that improves measured spatial support is a possible next experiment; simply
accepting the failed three-cell fit would change the qualification criterion.

Maximum native-versus-observed XY errors on accepted poses were0.777mm and
5.703mm. Those are conditional measurements on these traces, not calibrated
error bounds or support for observations after failure.

## Verification and timing

The probe recorded270 RGB-D frames and14,900 physics samples. All sensor,
contact, actuator, command, setup and stop audits passed. Both episodes pass
strict sampled physical visibility and the separate stable-interior/near-
occlusion measurement gate. No hard measurement failures occurred. Evaluation
replay left the model state unchanged and produced no gradients.

The prospective native goal criterion required actual arrival, an observed
candidate, a complete drain, no contact/stops, and501 consecutive physics
samples within.06m at speed at most.05m/s, with500 zero requests. Both fail.
The old150mm short-progress threshold was not used.

Median complete observation/control/command iterations were150.724ms and
229.695ms;14/15 and252/253 exceeded the100ms command interval. Maxima were
228.988ms and288.930ms. Median observation/control portions were119.870ms
and200.015ms. Physics remained paused during computation. These are CPU
simulation measurements, not real-time or hardware qualification.

The hardware preflight found32 logical/16 physical CPUs, about82GB available
RAM,97GB artifact storage,0% GPU activity and no substantial competing Python
job. Serial fresh workers avoided a concurrent native simulation. The measured
collection/audit phase took174.827s. Eleven resource samples showed maximum
CPU activity6.6%, minimum available RAM80,445,394,944 bytes and minimum free
artifact space96,598,597,632 bytes. The terminal binds1,146 artifacts totaling
474,957,268 bytes, excluding the terminal result itself.

Focused verification:24 controller/selector/interface/command tests passed,
then2 separate consensus-diagnostic tests passed. The latter preserve original
acceptance and explicitly test grid-only rejection. A test-fixture terminal
representation mismatch was corrected before native source freeze; no native
attempt was rerun. All source changes in this continuation are new named files.
Tracked-file diff remains empty; AGENTS.md and.ignore hashes are unchanged.

## Artifact identities

All roots are direct children of
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

Probe root: `go2_learned_goal_bootstrap_probe_v1_attempt_001`,861 bound sources
and59,665 inherited bound inputs.

| File | SHA-256 |
| --- | --- |
| launch.json | 962f9d461648d5aaa353c2a7a20e0a90245724cbf66cac223b60a89466422254 |
| result.json | 71601c660272b579db520ba631af661d76f869267cf1c4a26fb60b17cdfbb7f1 |
| family_episode_052_audit.json | 0bf21cf6f5ba693f490cb881df046672371f3cf55e53a6f49c7df48e38d7588d |
| family_episode_039_audit.json | 26aa2ef1ec6183d4212200d0095ba75f6e891e745d569e8f79fc4ffd6119b963 |

Separate readout root: `go2_learned_goal_bootstrap_readout_v1_attempt_001`,
865 bound sources. Its result records both full selection sequences and the
four unchanged-consensus rejection decompositions.

| File | SHA-256 |
| --- | --- |
| launch.json | 4bf83458cab7ba98719cfdbb504ec14e229ea2005fdc21e3abd17b185f5aa3e1 |
| result.json | a386f091390ba803f076a7c66ae2bbe573fa61673dbf31832c9e89b475976a77 |

Historical checkpoint root: `go2_independent_pulse_parallel_study_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| result.json | 588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122 |
| seed_2026091101_full_jepa_fit.json | 9d30bd4bcce402d897279fd23001103cc6db31935c22281ce3e3f0b74d23925b |
| seed_2026091101_full_jepa.pt | 59cec8efec26a2318bdfecb0315f84a17a244f7d03f4029d9f408cdcd3b6abc7 |

Its fitted model-state hash is
`d4b122bfc14910ef854686b37e0addc51f19955c095b259c23212cbea8f3d3bb`.
Whole-study definition SHA-256 is
`8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8`.
The full36-fit study was authenticated before the native launch. Reload was
evaluation-only with exact experiment/data/schedule/configuration binding.

## Continue from this evidence

1. Preserve both frozen probe/readout roots and all861/865 bound sources.
   Do not rerun this controller unchanged or relax its failed visual gate.
2. Address the measured image-support loss with a separately named sensor-only
   feature-support candidate. Test spatial support and pose accuracy on complete
   recorded traces, including failures, before a new prospective native run.
3. Address the turn-only policy separately. A model trained for actual moving
   candidate plans and a goal objective that handles orientation/continued
   passage require their own prospective definitions and matched comparisons.
   Do not interpret retrospective score changes as successful navigation.
4. Keep the failed family-design gate in
   `docs/go2_geometry_progress_family_result_2026-09-08.md` unchanged. Its
   policy-stream checker still requires a false readiness flag and remains
   unlaunched. Any new scientific use of those valid transition measurements
   needs an explicit new scope; it cannot turn the old transfer test into a pass.
5. Require actual downstream goal-reaching before expanding to independent
   mazes, exploration, memory/backtracking and matched planning/JEPA/RGB
   ablations. Resolve the observed timing deficit before real-time claims.

New implementation entry points are
`scripts/run_go2_learned_goal_bootstrap_probe_v1.py`,
`lewm/learned_goal_probe_development.py`,
`scripts/learned_goal_probe_episode_development.py`,
`scripts/learned_goal_probe_audit_development.py`,
`scripts/learned_goal_probe_checkpoint_development.py`, and
`scripts/read_go2_learned_goal_bootstrap_probe_v1.py`.
Their prospective protocols and focused tests are included in the source
bindings. No training, independent-maze comparison, hardware motion or goal
completion occurred. No native or diagnostic job remains running.
