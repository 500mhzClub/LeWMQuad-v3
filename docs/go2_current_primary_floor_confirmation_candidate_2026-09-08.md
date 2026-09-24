# Current-primary floor confirmation: unlaunched policy candidate

The completed view-recovery result and readout identify an auxiliary contact
failure involving samples that agree with contemporary primary floor geometry
but exceed the retained initial plane's 10 mm band. This candidate changes
auxiliary per-return classification only; it does not correct poses, infer free
space, delete raw returns or alter the original controller yet.

`lewm/current_primary_floor_plane_development.py` fits a current plane from at
least 100 primary samples already accepted by the original four-quad and
nine-pixel fixed-plane floor classifier. Require two-dimensional support with
second covariance eigenvalue at least 0.05^2 m², alignment at least .97 with the
retained up axis, and maximum seed residual at most .003 m. Insufficient or
incoherent primary evidence leaves the original classification unchanged.

For each auxiliary sample, keep the original floor classification and optionally
confirm an additional floor patch only when all four adjacent mesh quads pass
the original ground tests, all nine pixels are valid and within .01 m of the
current measured plane, and all nine remain within .03 m of the retained floor
height. The latter is a bounded local-plane disagreement gate, not a widened
plane residual band. The .01 m plane residual and .003 m geometry scales come
from the original floor checks; .03 m is the existing lower obstacle-height
cutoff. These are development hypotheses, not calibrated uncertainty bounds.
Original seed bounds remain reported. Plane extrapolation always requires the
auxiliary patch's own measured geometry; it cannot establish unobserved floor.

An initial extra restriction to the primary seed XY bounding rectangle plus
one 5 cm cell passed synthetic checks but confirmed none of the seven cited
samples. The seed rectangle starts around map X=4.15 m while those samples are
near X=3.70 m. That candidate was not installed or executed. The final candidate
uses the additional auxiliary measurement to confirm local geometry rather than
claiming primary spatial coverage. This scope change must be explicit in replay.

Three focused tests pass: coherent sloped-floor confirmation preserves original
classes; missing neighbors, raised patches and walls cannot gain confirmation;
missing primary seeds fall back exactly. The raised-patch test compares added
classifications against the unchanged original classifier, because original
classifications must remain preserved even where that original rule accepts a
sample. Tests do not establish physical support or navigation success.

`scripts/current_primary_floor_confirmation_probe_development.py` replays the
two cited public frames and verifies sample identities. The final candidate
confirms all five cited samples at frame 1420 and both at 1424. It additionally
confirms 13,148 and 14,018 auxiliary patches respectively; original floor counts
are 3,335 and 2,576. Primary seed counts are 1,396 and 1,311. This is a substantial
classification change requiring full-history, first-command-intervention replay.
It does not prove all historical voxel samples are confirmed or that the final
constraint is resolved. No alternative command was executed or inferred.

Next integrate into a separately named mapper/controller with original returns,
partitions and receipts retained as witnesses; preserve all non-foot, unknown,
nominal-path, mission and native guard checks. Check historical classification
and complete causal observation/command scope before a prospective native run.
Address the separate frame-909 visibility discrepancy. The running reactive
connector attempt is frozen and must not be modified to include this candidate.
