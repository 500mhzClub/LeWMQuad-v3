# Completed confirmed-floor causal prefix V1

Session 61119 completed successfully in 1554.3127937079407 seconds.
Output: `go2_confirmed_floor_maze_prefix_v1_attempt_001`.
Result SHA-256: `dfc4ed16525eb3c5474d4a436b92664f700d7bd91c9ff607571493b0adbe94bc`.
Launch SHA-256: `e58b99512f217b409bfd165bf6e6065e01e6199551c5afffc62d5daf2a58a00c`.
Decision stream SHA-256: `12c9e4fa949394322acc9d3b14aee4e878bfadfe93c47d1359d80b46a474e30b`.
The 1,470-source closure and completed native/readout/model input bindings were
verified before and after replay. Model state remains
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.

The first requested-command difference occurs at frame 1482 after 1,483 public
observations. The original request is left_turn [0,0,0.45]; the new request is
left_arc [0.16,0,0.45]. Neither controller becomes terminal at this observation.
All earlier commands and other decision fields match after only the declared
controller, selection and added confirmation receipt differences are accounted
for. Original public observations, raw maps, mission, residual inputs, forecasts,
nominal paths, primary checks and complete original auxiliary surface receipts
reconstruct exactly. The diagnostic predecessor selector's entire output matches
at every selection. No observation following the changed command was consumed.

At intervention, only left_turn was clear under the original contact checks;
all six are clear under the revised partition. The five changed candidate checks
are blocked originally by FL_foot:0 auxiliary sample bounds. The selected left
arc's original blocking voxel is [148,-3,-10], with 262 samples and latest
contributing frame 1420. Original points and classifications remain stored.
No primary or non-foot contact is exempted. All six candidates pass their
unchanged eight-segment nominal path checks. Current nominal clearance is
0.5153342379580643 m. The intermediate observed waypoint is [3.725,-0.325].

The current primary plane at frame 1482 is unavailable because the 172 seed
patches fail the minimum two-axis extent. Therefore this frame adds zero floor
classifications. The intervention uses persistent confirmations made when each
earlier auxiliary return was acquired with sufficient primary evidence; it does
not extrapolate a newly accepted plane at frame 1482. The additional partition
retains all 28,473,600 returns: 26,221,902 classified floor and 2,251,698 other.

The selected left arc has causal score -0.0006261188329790453 m, greater than
the original selected left turn's -0.004440874255638868 m under the same scoring
inputs. These are policy scores, not measured progress or calibrated safety.
No alternative action outcome or completed navigation is inferred from replay.

The input strict visibility failure at frame 909 is unchanged. Because the first
intervention occurs later, a fresh exact native prefix is expected to reproduce
that failure under the unchanged gate. The prospective native successor tests
physical navigation behavior only; qualification remains false while the strict
failure persists. Zero new independent layouts or verified arrivals/returns
were established by this replay.
