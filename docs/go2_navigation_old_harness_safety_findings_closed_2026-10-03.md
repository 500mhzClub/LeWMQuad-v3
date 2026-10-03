# Old-harness findings: forecast sensitivity, safety budget and calibrated margins (thread closed), 3 October 2026

**PRELIMINARY. Old-harness (V4) findings.** Development mode, prelim_test_v1 mazes 30–49, recovery off, coverage-rule fix. **This thread is closed** (Andrew, 3 October): the programme's goal, that the JEPA controller can navigate, is established, and safety analysis is a nice-to-have. The results stand as measured. Nothing here will be re-run or extended.

> **Trap caveat.** All of this was measured on the V4 harness, which has the shared reserve trap: once the robot's centre is inside the 0.48-m clearance requirement of a remembered wall, no translation passes the check, and turning drifts the body in. Most liveness losses below are that trap, or the route-versus-check clash it combines with under a margin. These numbers describe that harness, not navigation in general. The next harness version removes the trap ([plan](go2_navigation_harness_reserve_exit_plan_2026-10-02.md)); none of these experiments is repeated on it.

## Forecast sensitivity (C1, 380 missions)

C1 drove with its forecasts deliberately degraded ([details](go2_navigation_forecast_sensitivity_2026-10-02.md)).
- **Random error:** the cliff is between 20 and 40 mm of median 700-ms error (19/20 at 20 mm, 7/20 at 40 mm, 0/20 at 160 mm).
- **Consistent bias** of 0.5–1.5× on all moves was tolerated (19–20/20); 0.25× gave 15/20.
- **Bad forecasts froze the robot; none crashed it.** The cliff was mostly reserve-trap stalls: 45 of 50 stalls were the trap.
- **C1, C3 and C4 err by 5–10 mm while driving,** far inside the tolerance. So on nominal dynamics these mazes cannot separate them.

## Safety budget

From the [safety budget](go2_navigation_forecast_sensitivity_safety_budget_2026-10-02.md):
- **No contacts and no clearance violations** in any of the 832 missions: the preliminary run, sensitivity and margins.
- **The budget held.** The 0.45-m disc is 2.5 cm wider than the body's 0.425-m reach, so clearance should stay at least 2.5 cm minus the forecast, pose and map errors. No checked close approach fell below that bound.
- **The last-moment depth stop was not the safety mechanism.** Its 18 catches under degraded forecasts would all have stayed at least 10.7 cm clear without it.
- **Two blind spots:**
  - rear legs during turns are outside both forward cameras (the physical Go2's lidar would cover them);
  - never-observed cells count as free.

## Calibrated margins

From the [margin results](go2_navigation_calibrated_margin_results_2026-10-02.md):
- **The calibration transferred.** Without a margin, forecast error exceeded the p95 bound on 4.4–4.8% of near-wall decisions and the p99 bound on 0.7–0.8%, against 5% and 1%.
- **The margins cost far more than they bought.** p95 and p99 margins cut success from 20/20 to 10–11/20 for C1 and 12–15/20 for C4. Minimum clearance moved by about 1 cm, and the unmargined runs already had no contacts.
- **Why margin missions failed:**
  - the reserve trap, widened by the margin: 13 of 30 stalls, plus 2 more in its turning variant;
  - a route-versus-check clash: 14. Routing plans at nominal clearance, so the margin-reduced route-target lookahead stopped short, often on the robot's own cell. This never happened without a margin (0 of 13,102 decisions).
- **Never-observed cells blocking moves** (C1, walls at nominal): 9/20. Of its 13 stalls, 8 were the trap and 3 were bound by never-seen cells.

## Not done (dropped 3 October)

- Margin repeats on the new harness (C1, C4 and C3).
- Margin-aware routing.
- The C1 never-seen-cells re-run.
- The sensitivity re-measure.
- The C3 margin and never-seen-cells runs (cancelled earlier).

## Records

**Scripts:**
- `scripts/report_go2_forecast_sensitivity_development.py`
- `scripts/diagnose_go2_forecast_sensitivity_failures_development.py`
- `scripts/analyse_go2_forecast_sensitivity_error_budget_development.py`
- `scripts/analyse_go2_reserve_trap_development.py`
- `scripts/analyse_go2_stage2_stalls_development.py`
- `scripts/diagnose_go2_margin_route_check_clash_development.py`
- `scripts/calibrate_go2_forecast_margins_development.py`

**Cohorts** (all under the capability root):
- `sens_*`
- `margin_p95`, `margin_p99`, `margin_c4_nominal_b`
- `stage2_c1_lookaround_p95`
