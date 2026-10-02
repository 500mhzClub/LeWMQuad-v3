# Options: making the static benchmark discriminate (2 October 2026)

**Options only; nothing has been started.** The basis is the preliminary results ([report](go2_navigation_preliminary_results_2026-10-02.md)). On static mazes:
- C0 (oracle), C1 (kinematic), C3 (JEPA) and C4 (supervised) all reach 95–100% success with SPL 0.81–0.85.
- C3's closed-loop forecasts are about 1.7–2× less accurate than C1's or C4's, and it drives the same.
- Speed is limited by the selector and routing, not by prediction.
- Recovery flattens C2 and causes failures for the others.

The benchmark currently tests the shared harness (mapping, routing, clearance gates) more than the predictor.

**Cost basis** (measured this run): C3 takes about 0.8–1.2 h per mission on the single GPU, which saturates at 2–3 C3 runs; C4 about 0.5 h; C1/C2 about 0.12 h. One pass of C1–C4 on 60 mazes takes about 20–24 h wall time; on 20 mazes, about 8 h. A new maze family takes about half a day to generate and register (structural checks only).

| Option | Hypothesis it tests | Expected cost | Expected to separate C1/C3/C4? |
|---|---|---|---|
| **A. Recovery off as the main setting** (recovery thresholds as a separate ablation) | The controller's own choices (predictor plus selector) carry performance. Recovery flattens differences and adds failures. | No engineering. Re-uses the current harness; about 1 day per 60-maze pass. | **No, on its own.** Recovery-off results were 20/20, 19/20 and 20/20. It does remove recovery's distortions (C2 60/60 → 11/20; the two recovery-caused failures) and should be the default for every option below. |
| **B. Forecast-sensitivity check** (do first). Run C3/C4 with their forecasts degraded (scale bias, noise, or C1's kinematic forecast swapped in) and measure driving. | Whether driving outcomes depend on forecast accuracy in this harness at all. If success holds until forecasts are grossly wrong, no static variant of this harness can discriminate predictors. | Small: a model-output wrapper (half a day), then about 1 day on 20 mazes at 2–3 degradation levels. | It measures the harness's tolerance, which tells us whether C–F are worth running. |
| **C. Tighter time budget** (success within T, e.g. 1.25× C0's time, or 200 s instead of 480 s) | Efficiency: who completes fastest. | **Free.** Budget-truncated success can be rescored from existing runs, because the controllers never see the budget. | **Unlikely.** C1, C3 and C4 are within a few seconds of each other; C2 is fastest. Time is selector-limited, not prediction-limited. |
| **D. Narrower mazes** (corridors near the 0.48-m clearance requirement, more tight turns) | Whether accurate clearance and motion prediction matter when margins are thin: more decisions are decided by the forecast-clearance gate. | Generator change and new set registration (half a day to a day), plus about 1 day of runs. | **Possibly, but confounded.** Thin margins also trigger the harness's geometric deadlocks (the 0.45-m disk), which hit every controller and need recovery or a gate fix first. |
| **E. Larger mazes** (longer routes, more exploration) | Whether small per-decision inefficiencies compound over long routes; mostly exploration and memory. | Generator change, plus runs about 1.5–2× longer (about 2 days per pass). | **Unlikely.** It mainly stresses shared mapping and routing, not prediction. |
| **F. Dynamics perturbation** (lower floor friction, payload or actuator lag), so commands no longer give nominal motion | **Whether vision-based prediction carries information beyond command history**, the programme's question. C1's kinematics would be wrong; C3/C4 can see the true motion. | Physics parameters in Genesis (about 1 day, including checking the controllers stay stable), plus about 1 day on 20–60 mazes. C3/C4 may need refits on perturbed data, or are tested zero-shot. | **Most likely, of the cheap options, to separate C1 from C3/C4.** It does not by itself separate C3 from C4. |
| **G. Sensor noise** (RGB/depth noise, blur, pose drift, latency) | Robustness of visual prediction and of the shared perception stack to degraded sensing. | Noise injection in the sensor path (1–2 days) plus about 1 day of runs. | **Mixed.** Depth and pose noise hit the shared mapping for everyone; RGB noise hits only C3/C4, and would likely favour C1 (no vision). |
| **H. Moving obstacles** | Whether learned scene prediction (C3's latent predictor of future features) carries decision-relevant information beyond ego-kinematics: avoiding agents needs forecasts of the scene, not just of the robot. | **Largest.** Moving actors in the simulator; dynamic-obstacle mapping, since the static occupancy map breaks; new safety and evaluation definitions; decoding scene forecasts (C3/C4 currently forecast only ego-motion). Weeks. Needs **separate approval** under the programme goal. | **Yes, the strongest test of the JEPA hypothesis**, and also the most engineering. |
| **I. Foundation-model baseline** (C4 on an encoder trained from scratch, not V-JEPA) | Whether the pretrained V-JEPA representation itself adds value. C3 and C4 currently share it. | Training (GPU days), plus one 60-maze pass. | Only meaningful on a benchmark that discriminates (after B, F or H). |

**Also needed for any rigorous claim:**
- Pre-declared equivalence margins: with near-ceiling success on 60 mazes, the success-difference interval is about ±5 points.
- Several seeds of the decoder and of C4.
- The coverage-rule hold fix (agreed for after the preliminary results), so that frozen-harness quirks do not add noise.

**Suggested order:**
1. **A, as the default setting.**
2. **C now**, since it is free from existing runs.
3. **B**, which answers whether any static variant can discriminate.
4. **F**, the cheapest direct test of "beyond kinematics".
5. Then **D** or **G** if B shows sensitivity.
6. **H** as the main programme step, with separate approval.
