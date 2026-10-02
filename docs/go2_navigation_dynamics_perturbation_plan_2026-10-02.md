# Plan: dynamics perturbation (draft, not run) — 2 October 2026

**Status: DRAFT plan only. Nothing here has been started.** Option F of the [options note](go2_navigation_benchmark_discrimination_options_2026-10-02.md), drafted per Andrew (2 October) after the forecast-sensitivity experiment. Moving obstacles (E2) wait until this is done.

## Question

When commands no longer determine motion, does vision-based prediction (C3 JEPA, C4 supervised) carry decision-relevant information that command history (C1) lacks?

In the preliminary run, C1's kinematic forecast was as accurate as C4's (6 mm) and drove as well. The forecast-sensitivity experiment ([results](go2_navigation_forecast_sensitivity_2026-10-02.md)) measures how much forecast error this harness tolerates before driving degrades. A perturbation only discriminates if it pushes C1's error past that point while the visual predictors stay inside it.

**Hypotheses:**
- **H1.** Under perturbed dynamics, C1's forecast error grows (it maps commands to nominal motion), while C3 and C4, which see recent frames, track the true motion more closely.
- **H2.** That error gap turns into a driving gap (success, SPL, time, contacts) once C1's error passes the harness's tolerance.
- **H0.** All three degrade alike: C3 and C4 learned the nominal command-to-motion mapping and use vision mainly for what they also get from commands.

## Perturbations

All are applied in the simulator only. The controllers, harness, sensors and evaluator are unchanged, and C0 stays a perfect-forecast reference because its oracle forecasts come from the perturbed physics.

| Perturbation | Mechanism (Genesis 0.4.6) | Effect on command → motion | Candidate levels |
|---|---|---|---|
| **Floor friction** | The maze spec's `friction_mu` (default 1.0) feeds `physics_randomization.floor_friction_mu`, the floor's Rigid material (clamped to 0.01–5). Earlier room-return experiments used 0.15 through the same path. | Foot slip: less translation and yaw rate than commanded, with more variance, and drift on turns. | μ = 0.6, 0.4, 0.25 |
| **Payload** | `robot.set_mass_shift(Δm, base link)`, plus optional `set_COM_shift` for an off-centre load. Go2 base link 6.92 kg, total 15.02 kg (Genesis `go2.urdf`). | Slower acceleration and lag; reduced speed on turns. | +2, +4, +6 kg (centred); +4 kg with COM shifted 5 cm |
| **Motor strength** | Scale the PD gains (`set_dofs_kp/kv`, nominal kp 20, kd 0.5 from the locomotion checkpoint) or the torque limits (`set_dofs_force_range`). | The locomotion policy tracks its joint targets less well: lower realised speed and sluggish turns. | kp × 0.7, 0.5; torque limit × 0.6 |
| **Mixed, within an episode** (stage 2) | Low-friction floor patches with the existing `slick_patch` visual marker (`lewm_genesis/textures.py`). | Motion changes where the floor looks different: vision could anticipate it, command history cannot. | μ = 0.25 patches over 20–40% of route cells |

**Picking levels.** Before any navigation run, a short open-loop characterisation drives each candidate level with fixed command tapes on a flat floor. It measures:
- realised over commanded speed and yaw rate;
- their variance;
- whether the locomotion policy stays stable, with no falls.

The levels kept are those whose realised/commanded ratio falls where the forecast-sensitivity curve shows driving starts to degrade for a uniformly biased forecast, plus one beyond. Levels where the robot falls or cannot walk are dropped; they test locomotion, not prediction.

## Evaluation

- **Mazes:** the 20 preliminary mazes 30–49, the same as the sensitivity experiment, so results line up with its dose-response.
- **Setting:** recovery off (the default) with the coverage-rule fix.
- **Controllers:**
  - **C0:** the perfect-forecast ceiling under the same physics. It separates harness and locomotion limits from prediction.
  - **C1:** unchanged command-history kinematics, fitted on nominal dynamics.
  - **C3:** the large past-frames decoder, which sees frames 0.5 s and 1 s ago.
  - **C4:** the matched supervised predictor, which sees three frames plus command history.
  - **C2:** a reference only; it uses no forecasts, so it shows the perturbation's effect on the shared harness and on locomotion alone.
- **Primary measures, per controller and condition:**
  - **Mechanism:** the forecast error the planner acted on while driving (median 700-ms error and ratio, by movement type), from the closed-loop scorer.
  - **Outcome:** success (Wilson 95%), SPL, median time, hold rate, contacts and clearance.
  - **Paired:** against C1 on the same mazes (bootstrap difference, discordant counts), and C3 against C4.
- **Reading:**
  - H1 holds if, under perturbation, C1's measured error rises well above C3's and C4's.
  - H2 holds if the driving gap follows, roughly where the sensitivity curve predicts.
  - If C0 also degrades, that share of the loss belongs to locomotion and the harness, not prediction.

## Retraining

1. **Stage 1, zero-shot (no retraining).** C3 and C4 as trained, on nominal dynamics. This tests whether their visual inputs already let them follow the real motion. C1 is also used as fitted, which is its fair default.
2. **Stage 2, adaptation (only if stage 1 is ambiguous or H0 holds).**
   - Record C1-driven missions under each kept perturbation on 16 development layouts (the C3-v3 round's on-policy fit layouts, not the evaluation mazes).
   - Build a small feature cache.
   - Refit the C3 decoder and C4 on the same mix plus the perturbed data.
   - Refit C1's command-history model on the same data.

   If an adapted C1 matches the adapted visual predictors, vision adds nothing under *stationary* perturbations. The discriminating case is then the mixed, within-episode patches, where only vision can tell in advance.

## Cost (based on the preliminary run's throughput)

| Step | Engineering | Compute |
|---|---|---|
| Perturbation hooks (spec override for friction; mass and COM shift and gain/torque scaling bound into session setup), with checks that the harness validators and replay still pass | about 1 day | — |
| Open-loop characterisation of about 9 candidate levels, stability checks | — | about 2 h CPU |
| Stage 1: 3 perturbations × 1–2 levels × C0–C4 × 20 mazes | — | C3 dominates at 0.8–1.2 h per mission (20 C3 missions per condition is about 8–10 h on the one GPU). 3 conditions is about 1.5 days; 6 is about 3 days. C0, C1, C2 and C4 add about 0.5 day. |
| Stage 2 (optional): recordings, cache, refits, re-evaluation | about 0.5 day | about 2 days |
| Mixed within-episode patches (stage 2): textured slick patches in the scene builder | about 1 day | about 1 day per evaluation |

**Recommended first step** (about 2 days in total):
- Hooks, the characterisation, then **friction only** at the two levels picked by characterisation, zero-shot, on C0–C4 × 20 mazes.
- Friction is the simplest to apply (one spec field, already used in this project). It acts on both translation and turning, and it fails the command-to-motion assumption most clearly.

## Risks

- The locomotion policy may be unstable on low friction or with heavy payloads. The characterisation filters such levels out, and falls are counted as failures, not hidden.
- The harness's own safety margins (clearance reserves, a 0.48-m requirement, replanning every 400 ms) may absorb even large forecast errors. The sensitivity experiment tells us how much before we spend compute here.
- C3 and C4 were trained only on nominal dynamics. A null zero-shot result does not show that vision cannot help, only that these models did not learn to use it for that. Stage 2 addresses this.
