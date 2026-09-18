# Measured-plane reactive comparator: prospective sensor prefix

This development replay prepares a fully nonpredictive physical comparator for
the measured-plane learned controller on maze 02. It makes no navigation claim.
It is a whole-method comparison: current measured geometry and reactive route
following replace learned forecasts, predictive feasibility, scoring and learned
residual correction. It does not isolate predictive ranking alone.

Use the original closed extended-budget maze-02 sensor evidence and the complete
verified 123-observation measured-plane controller prefix. Admit the original raw
artifact roster before and after execution. Reproduce every complete learned
decision with the same corrected no-RGB direct model. The reactive controller
uses the same measured-plane estimator, floor registration, observation map,
mission, observation clock, public mission and 4,000-tick budget. It instantiates
no world model or learned residual. All modules of the baseline model are
guarded against forward calls during each reactive observation, including calls
whose exceptions the controller catches.

Compare complete visual, floor, map, mission, goal-distance and auxiliary-floor
receipts at every consumed frame, without normalization. Authenticate the full
public packet and reject any input mutation. Retain both complete decisions.
Stop at the first changed command, either terminal, or the fixed original
prefix end at frame 122. Never consume an observation after a changed command.
Check actual original command endpoints and reconstruct the complete saved
comparison and report. Preserve any failed attempt without automatic retry.

Source preflight admits the completed forecast-source replay and source closure,
checks CPU/RAM/GPU/disk resources, and creates no output. Actual execution also
requires the ended preceding CPU replay and complete original raw inputs. One
single-thread CPU replay may overlap the existing single native scene. The
`replay_go2_` name does not impersonate a native runner. The live learned native
and nominal waiter sources remain frozen and unmodified.

Output is `go2_measured_plane_reactive_prefix_v1_attempt_001` under the existing
development artifact root. No training, independent maze result, realtime result,
hardware qualification, changed-command physical outcome, or goal completion is
established by this replay. A later separately defined native run must execute
the actual reactive commands and pass the complete sensor and command audit.
