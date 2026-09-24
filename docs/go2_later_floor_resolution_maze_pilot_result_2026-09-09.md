# Ninth maze pilot: first goal-region reach, failed arrival and return

Native59231 and completed execution readout16797 are CLOSED exit0. The robot
traversed8.946266774399417m over188.0 simulated seconds after the initial
observation. Minimum native outbound goal distance0.021496480659927094m;
terminal distance0.039020034366980254m. It completed the six-edge loop-erased
outbound route, but the claimed arrival1866 failed the full one-second quiet
window. Visual tracking failed1870, before any return edge crossing.

Native result3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755,
launchc3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4,
wall4510.6477477950975s; rootgo2_later_floor_resolution_maze_pilot_v1_attempt_001.
Full raw audit26dad9cb08152eed49d3e518a5653af264b5dbe51091a64aca2031e30b4bb9af;
prefix comparison444ad02042b58ea96a6029243d335c78ab218b7b81c57297ef6fc98dee833dc0.
Raw sensor reconstruction, full model/controller decision replay, actual
command audit and unchanged model state all pass. All960 physical/public/
prospective decisions match through the first contact-policy intervention959.
The physical-prefix SHA-256 is
d9311f66e3d861870f9bf021979aa75d931b089e967d5af031cd704a17eab5d3.

Strict primary visibility fails only frame909, which precedes the intervention.
All1881 auxiliary visibility checks pass, with no auxiliary robot-occluded
frames; the hard-measurement-failure list is empty. None of these results
overrides the strict failure. There were no physical/acquisition stops.
All1880 commands completed, with1881 paired observations,94750 native2ms
samples and ten zero-command terminal-drain ticks.

The arrival window stays within0.03761944072156676m but reaches
0.1524744894151264m/s against the unchanged0.05m/s speed threshold. This is
braking at the beginning of the claimed window, not the speed when the RETURN
transition was issued. The independently preserved dynamics and matching
diagnoses are in go2_ninth_maze_transition_diagnosis_result_2026-09-09.md.

## Completed execution readout

Rootgo2_later_floor_resolution_maze_readout_v1_attempt_001,
result7635c951522895fc03059a491be4f1539a11f85205e406780bb529ef43b70e5c,
launch858f8f3c91e05482e33fca56e6182c8a458fd1b2c615ad5e30e23a8a97e0e31a,
1543 bound sources. It binds the completed native result above and revalidates
all required artifacts/sources. No new simulation, training, controller
selection or relabeling of the original outcome occurred.

There are86 completed100ms selected intervals whose original contact check
would block but whose later-measured floor evidence clears the selected
nominal-foot intersection. No such interval is censored. This records actual
executed interventions, not a counterfactual predecessor trajectory or proof
that all downstream policy differences are attributable to those86 intervals.
There are1765 completed waypoint-action readouts,1714 local reranking changes,
raw mean endpoint prediction error0.007662026367478671m and causal-scoring
mean endpoint error0.0063581490654792475m on those executed intervals. No
unexecuted alternative outcome was inferred. Two nominal reentry intervals
and one translating view-reentry interval were executed; observed nominal
reentry595 returned to the ordinary selection gate597.

Across1870 admitted poses, mean/max raw3D error7.505845/15.435210mm and
registered3D error4.236899/8.581741mm. Mean/max registeredXY error
4.235033/8.577553mm. Mean raw/registered rotation error0.005148482/
0.004201689rad; maximum0.011266834/0.010555701rad. These are accuracy
measurements on one executed trajectory, not calibrated uncertainty bounds.

Median actual iteration including receipt writing1184.193229ms; all1881
recorded receipt-inclusive iterations exceed100ms. Physics pauses during
computation. Real-time and hardware qualification remain false.

## Next experiment boundary

The separately named settled-boundary controller addresses observed dwell
timing and is currently undergoing complete raw-controller prefix comparison.
It retains the original observer and all later-floor/model/planner behavior.
The failed subpixel feature candidate is not adopted. A new native attempt
must establish its own physical/sensor/command/prefix evidence and cannot
repair this run's failed arrival, return or visibility result retroactively.
There remain zero verified arrivals/round trips, no independent-layout success
cohort, no matched baselines/ablations, no real-time operation and no bounded
hardware validation. The full goal remains active and unachieved.
