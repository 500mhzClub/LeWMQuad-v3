# Live automatic correction handoff and next native definition

The previous storage/status turn was a verified wait: the original fitting and
native processes were observed live. This turn made progress by implementing,
testing and launching the single correction handoff. The full navigation goal
remains active:30 completed raw-audited native episodes, zero verified round
trips, with the next original native episode still running at this observation.

Correction waiter handle95563/PID2639661, creation1789019633.03, is live:
scripts/await_go2_all_phase_translation_bias_v1.py.
Launch4329326dc8b3d07983bde26f7149141d9aee7f2eccb57343949a197b4b935238,
1165 sources, exclusive root go2_all_phase_translation_bias_wait_v1_attempt_001.
It owns the future launch of the existing correction runner. Do not manually
launch that runner or another waiter. It waits for original waiter2635725 and
parent2636352, admits their completed exact results, then derives corrections
and executes full correction admission. No fit or native scene is launched by
this waiter. A bounded wait expiration preserves the running original work.

Frozen handoff sources:
- scripts/await_go2_all_phase_translation_bias_v1.py:
  5e4b5f8e44360ff0399584c5e7158204a2898eb1212ac25a4d9108fed955b829
- lewm/tests/test_all_phase_translation_bias_wait_development.py:
  a75007aedfbb127e7f254cdc17046bdc56e4e3dd0c80515b8a0c99a7c8deb57e
- docs/go2_all_phase_translation_bias_wait_v1_2026-09-10.md:
  7111dbda4f0cac7c101b3c315be516a047307db7ea540f3873210ba8e820bda6

Tests81685 closed0:22 passed in2.82s, covering original owner identity/liveness,
reboot, failed/missing results and changed fit/source/benchmark chains.
Preflight55065 closed0: original1144 fit and1146 waiter sources verified; all1160
correction sources are included unchanged in the1165-source handoff union;
both original owners live; exclusive outputs absent before registration.
Post-launch14180 closed0 independently observed this waiter's PID, command,
creation time and launch/source count. No waiter result or failure yet.

Latest full-fit logs: direct worker reached1200 updates on its sixth/final fit
(seed2026091402 no_rgb); supervised worker reached700 on its sixth; JEPA reached
1200 on its fifth (seed2026091402 full). These are update observations, not
completed snapshot/prediction/admission claims. Full-fit/waiter results remain
pending. Preserve original handle6946 and all original child processes.
Native handle25801/PID2636286 and worker2637549 remain the sole native attempt;
no terminal result/failure was present. Do not overlap a new native scene.

The next prospective native science is now fixed before inspecting new fitted
scores or coefficients:
docs/go2_all_phase_residual_maze02_matched_native_v1_2026-09-10.md,
SHAaa239cd4933e30f6e6af64f564250a44f99233d3639ee99946481fb506aac84e.
Use the six preassigned primary-seed full/no_rgb JEPA, supervised and direct
models on development maze2 with unchanged ResidualAnchoredContinuationController.
This targets the observed stopping drift through expanded training while
retaining the original controller. It does not establish independent-maze,
online-planning or persistent-memory advantage. The no_rgb treatment affects
the model's RGB input, not the controller's RGBD localization and map.

Remaining concrete work:
1. Preserve/poll the original live fit, correction and native owners. Admit all
   terminal artifacts; never restart on an observation timeout.
2. Complete the separately checked six-case native launcher and common physical/
   public startup comparison. Require complete new correction admission, exact
   old development reference, authenticated current native completion, unchanged
   controller/environment and fresh resource/native-idle checks. The protocol
   alone is not a launched or verified experiment.
3. Execute all six assigned cases without outcome-based case/model substitution,
   preserving negative outcomes and full raw audits. Then pursue independent
   layouts and the still-missing planning/memory controls and sensing/timing/
   hardware qualification required by the active goal.

Current artifact free capacity is about621GiB; workspace free is about20GiB.
Storage is not the current blocker. Do not delete verifier-required input data.
