# Expanded models admitted; fixed six-case native comparison queued

This goal turn made progress: all18 expanded-context fits completed, training-only
correction and complete model/correction admission finished, and the six-case
native launcher was implemented, tested and registered behind the original
running native pilot. The navigation goal remains active. There are still30
completed raw-audited native episodes and zero verified round trips. The original
31st native episode is running; no expanded-model native result exists yet.

## Completed fitting and correction

Full-fit result44c4cd65812b021b29cfb0aff33e2058cfaaa71dcc627d36eced30dfac17ea35,
root go2_all_phase_matched_fits_v1_attempt_001:18 models,21600 optimizer updates,
1144 sources,156 outputs, selected workers3. Each worker completed7200 updates.
Reported fit phase wall1937.33473261795s, including cache warmup/fitting/predictions
after launch; it is not a claim about prelaunch admission time. Worker walls:
direct1775.572895860998s, supervised1841.7456821519881s, JEPA1931.9014684751164s.
No checkpoint selection or reuse of benchmark weights. Original waiter6946
closed0 with resultc9e9c7db52b31087d1ebfd17391aaf525f01da32788d65ca7acc85d8102c2898.

Correction result1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460,
root go2_all_phase_training_translation_bias_v1_attempt_001, launch
e2ade21e600a8d1c38ddb3eac3c2504f388c79a666b7054dec8780d051f14735.
All18 models/30 heads/480 scalars fit from4010 training contexts, retaining all36
motionless contexts in accounting.1160 sources,3 outputs. Reported coefficient
phase wall26.07074808399193s excludes its prerequisite full model admission.
Correction parent2640832 exited0. Original correction waiter95563/PID2639661
closed0 with result4bf13d2e00fb318fa836d02bad93784fbb1a9c5ccef8792bd19dcfd503f657a0,
1165 sources,4 outputs. It executed complete correction admission, including
reconstruction of the coefficients and full18-model/21600-ledger/raw-score/
snapshot admission, and saved correction_admission.json. Do not relaunch it.

Independent25868 closed0: all sources/artifacts of the completed fit, correction
and waiter were checked before and after six fresh evaluation-only model loads.
All six treatment assignments matched; their corrected state identities are:

| Assigned primary-seed model | Corrected state SHA-256 |
| --- | --- |
| full_jepa | 35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a |
| full_supervised_rollout | 755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5 |
| full_direct | 8cf81bec6c67261df9d25b56d2f6546abd887735ab8977021bb54a3c1963f1df |
| no_rgb_jepa | fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f |
| no_rgb_supervised_rollout | 7d9a53c6477884548f687cb9a26184e11621304b927cb924838795badb226035 |
| no_rgb_direct | 56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd |

All names have prefix seed_2026091001_. The primary JEPA base state is
940d21dd221f22fb3e37adcf9a9e31534c4ac0b7743ba22f9b5b1037a65ac3aa.
These are identity/admission results, not native performance claims. Corrected
geometry-transfer evaluation remains unperformed; no corrected score has been
used for selection. The native assignment/protocol preceded score inspection.

## Live native handoff

Waiter14784/PID2641948, creation1789020925.89, is live:
scripts/await_go2_all_phase_residual_maze02_native_v1.py.
Launch6e96b6bc78f08f8dd7b5af3ad25e48b78ca915a5c3b5ede37efe0bc8a7be9b5e,
1890 sources, root go2_all_phase_residual_maze02_native_wait_v1_attempt_001.
It owns the future six-case launch. Do not launch another waiter or manually
start the cohort. It waits for exact original native owner2636286; correction
owner2639661 has completed and is absent. It will pass the observed completed
native result and fixed correction-waiter result into the checked launcher.
The cohort output root remains absent; no new native scene has been launched.

Source preflight25552 closed0:1887 native sources unchanged within1890 waiter
sources; original native live; exclusive outputs absent. Memory available
76,661,657,600 bytes, artifact free661,918,040,064 bytes, workspace free21,356,756,992
bytes,16 physical/32 logical CPUs,5.8% busy, GPU0%. The complete six-case allowance
is113,816,633,344 bytes (106GiB), with32GiB available RAM and one native worker.
These are capacity checks, not OS reservations. Actual admission runs again
after the original owner finishes and before each new case.

Post-launch67405 closed0 observed exact new waiter PID/creation/command, verified
all1890 source bindings, confirmed the original native owner live, no new cohort
root and no waiter failure. Preserve original native handle25801/PID2636286 and
worker2637549. Its latest observed live timing ledger reached tick1450; that is
an execution observation, not a completed raw audit or navigation result.

The six models use unchanged ResidualAnchoredContinuationController on reused
development maze2. All negative scientific outcomes continue through the fixed
roster; incomplete collection/raw replay stops and preserves the attempt. The
direct condition still predicts candidate outcomes. The no_rgb treatment removes
model RGB while retaining controller RGBD localization/mapping. Neither isolates
online planning or persistent memory, and this is not independent-maze evidence.

## Verification and frozen new sources

Final tests61440 closed0:47 passed in2.48s, covering source-assigned model states,
complete six-case accounting, negative outcomes, joint success criteria,
pre-command physics/public startup, original-owner lifecycle and failed/missing
terminal evidence. Earlier39/42-test passes were superseded after adding the
compact scientific readout and waiter tests.

Actual-original-artifact probe44536 closed0: all old maze2 source/output bindings
checked before and after startup self-comparison and compact readout. Startup
is900 physics samples/four public observations/three completed zero commands,
physics fingerprint8419be1a3143128fcef2a1cf843d6476063177cc52f83316c42fd9b2bc7789fa,
public fingerprintc70015e14262cc01a4e733d999676b38931748d936c57732b4f5a170f4419280.
It reported three crossings of one distinct open edge, no arrivals, no contact
in14000 physics samples, strict visibility pass and no verified round trip.
Observed median observation/control938.776559ms, median iteration with receipt
990.6766494999999ms; all266 observations exceeded100ms. This is the old fixed
development reference, not a new native run or performance optimization result.

Frozen under the live waiter:
- scripts/await_go2_all_phase_residual_maze02_native_v1.py:
  7a50eb991ab7b9bb5d3d9c6d7d24885a0fa918dece8075fc3c5d70b60ebd4541
- scripts/run_go2_all_phase_residual_maze02_matched_native_v1.py:
  50c7e7df972428a732ec806316819b4ffe84163d8f6d49e7915a035f8b24e187
- scripts/all_phase_residual_maze02_native_inputs_development.py:
  9da96d7b7cd46a0741bd2d1cf37c1602693547bd9afa3e8b72e312a0ffb899c3
- scripts/all_phase_residual_maze02_startup_development.py:
  1e5c6ea044b87feb82fb0d8eeb0ab37b8c40b1356c1c7c0d0f762cfbcbbbe50e
- lewm/all_phase_residual_maze02_study_development.py:
  e92166bf3c20951bd17d365d963bdb7a3076d25efafe194d107eebece6713740
- lewm/all_phase_residual_maze02_readout_development.py:
  132856160e62ffdcd8c938ed6ebd29a484f8b685816bd0a0468ff4b298d3eeed
- docs/go2_all_phase_residual_maze02_native_wait_v1_2026-09-10.md:
  4edb8962c4e7ab372ac8614ac50803db032e6137a75dc8fcbced58257b2237ab
- docs/go2_all_phase_residual_maze02_matched_native_v1_2026-09-10.md:
  aa239cd4933e30f6e6af64f564250a44f99233d3639ee99946481fb506aac84e

Keep all these files and their imported closure unchanged. All test-file hashes
are bound in the1890-source waiter launch. No source export, commit, data
deletion, new simulator variant or hardware deployment occurred.

## Next work

1. Preserve/poll native25801 and waiter14784. Authenticate the current pilot's
   terminal result when it exists. The waiter owns subsequent execution; do not
   duplicate it. Preserve a failure rather than retrying implicitly.
2. If useful while native work continues, perform a separately bound complete
   corrected prediction readout for all18 fixed models without selection or
   changing the queued assignments. This is distinct from native success.
3. Audit all six assigned fresh native outcomes, including scientific failures,
   startup, strict visibility, distinct edges, return and timing. Follow with
   independently specified layouts and matched planning/memory controls toward
   the full goal. Realistic sensing, real-time feasibility and bounded hardware
   evidence remain unestablished; storage is not the current blocker.
