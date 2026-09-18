# Reactive versus learned maze0 readout

The paired readout completed in session59988, exit0. Result SHA-256:
`db726052b74b4afe4220f86334c855a4f371e808aeeb06626520b17349bfe783`.
Launch SHA-256:
`5f754ff78a722fbfb7efe26fdc827ab9b88bd791d38eacb462a09f82ec660079`.
Root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_reactive_floor_transport_maze_readout_v1_attempt_001`.
The result binds1698sources. Input/source/artifact checks completed before and
after the analysis; original outcomes are unchanged. No native scene or model
training ran in this readout.

| Measure | Learned floor transport | Reactive floor transport |
|---|---:|---:|
| Verified outbound arrival windows | 1 | 0 |
| Verified round trips | 0 | 0 |
| Native XY path length | 10.304554m | 1.913349m |
| Minimum native outbound-goal distance | 0.021496m | 2.808443m |
| Simulated duration after initial observation | 301.3s | 28.3s |
| Strict physical visibility | Fail | Pass |
| Hard measurement failed frames | 1924,1925,1930 | None |
| Median observation/control time | 1182.856ms | 681.978ms |
| Median iteration including receipt | 1237.034ms | 718.103ms |

The learned controller reached its settled outbound arrival at frame1868 and
later exhausted the shared mission budget without returning home. Reactive
stopped at273with no action satisfying current geometry. Its first infeasible
selection occurred at115; it recovered before the terminal sequence. It selected
97forward,94right-turn and54left-turn actions, with26infeasible selections.
Its final native initial-frameXY was[1.1367318614971147,-0.7537140377218317],
2.8167497506607484m from the outbound goal. Maximum observedXY pose error was
0.0019315144435131258m. Native crossings were the two open edges from[-1,0]
through[0,0]to[0,-1]; no arrival, return traversal or physical retrace occurred.

Admission verifies22matched launch fields covering the scene/mission, robot,
budget, runtime, renderer, acquisition, source inputs and resource settings,
plus exact shared observed state at the physically reproduced first intervention.
Both methods retain persistent observed memory. Reactive has no high-level
learned model and uses current geometry; predictive feasibility gates differ.
Consequently this is one reused-layout method comparison, not an isolated
prediction-ranking, JEPA-objective or memory ablation. It does not establish
generalization, statistical reliability or a causal training/planning advantage.

Every measured observation/control iteration exceeds100ms for both methods.
The timings cover different trajectories, map histories and episode lengths;
they do not isolate model-inference cost. Simulation pauses physics during
computation. No real-time or hardware qualification follows.

Native inputs:

- Reactive `4d377b9ec202c96099615ca5e3679037d5cb81d30c319afdd7febb9b9f0c9837`.
- Learned `1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374`.
- Learned readout `a46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c`.

The fixed independent reactive cohort was submitted as session18182,
PID2475541, using these completed reactive and independent learned native
results. At inspection it was live in input verification (89.41CPU seconds,
1,310,191,616bytesRSS), without a launch result yet. No cohort outcome is
claimed here. The next memory comparison remains separately prepared.
