# Measured floor transport candidate prepared

Implemented separate sources:

- lewm/measured_floor_transport_development.py: typed measured SE(3) composition
  from the most recent admitted floor anchor and current visual pose; retains
  the unavailable current fit and checks every candidate against the transported
  reference, with no point trimming, current-plane fabrication or error bound.
- lewm/measured_floor_transport_registration_development.py: exact original
  registration when available; missing count/extent alone may use transport.
  The initial reference must be fully admitted. Transport never promotes an
  anchor; reacquisition uses the original fit/correction. Failures latch.
- lewm/measured_floor_transport_controller_development.py: mapping, residuals,
  mission and contact geometry share explicit pose dispatch. The three copied
  consumer methods differ only in their accessor (AST comparison passed).
  Settling rules remain unchanged; its motion-source wording explicitly allows
  visual positions in the floor reference without an admitted current plane.
- lewm/measured_floor_transport_prefix_development.py and
  scripts/replay_go2_measured_floor_transport_prefix_v1.py: full fresh controller
  replay0..1904, complete1904 prior-decision/actual-command equality outside
  strictly validated labels, identical current raw visual evidence at1904,
  exact saved1903 floor anchor, current transported pose and active RETURN.
  No later recorded observations or new-action outcomes may be consumed.

Tests57641 CLOSED31passed10.56s. Coverage includes noncommuting SE(3) rotation/
translation against independent homogeneous-matrix composition; both camera
residual populations and their moments; tampered/stale pose/anchor/clock/hash;
unavailable-plane and current conflict rejection; initial admission, unchanged
full-plane path, nonpromotion across missingness and original reacquisition;
all online consumers and untouched old map entries; complete synthetic public
pipeline decisions; and incomplete raw/native-prefix admission rejection.
Synthetic motion witnesses do not establish fresh raw image fitting accuracy.

Initial test runs exposed two fixture assumptions (missing synthetic calibration
metadata and exact decimal equality after floor fitting) and one missing helper
import in the new registration file. Those were corrected before the31-test
pass. No launched source or recorded experiment was changed.

CLI2482 CLOSED pass. Source33055 CLOSED pass:1626 prepared source paths,
including all1618 frozen diagnosis/predecessor paths. Independent binding check
also confirmed1616 native,1618 diagnosis and1624 optimization launch sources
unchanged. No floor-transport replay root/preflight/native scene was launched.
Hardware33055:73,650,253,824bytes RAM available,104,757,866,496artifact-free,
3.7%CPUbusy; existing native raw-audit worker remains active. One independent
read-only geometric check86294 completed: at1904 from admitted1903 anchor,
the proposed position is[3.877104248806246,-1.3015972181875413,0.01973057004560251].
Correction norm0.014757592402m and angle0.004207392595rad pass the declared
development magnitudes. All6514auxiliary candidates have maximum transported-
reference residual4.629786670e-5m/RMS1.803456171e-5m, below3mm. Primary candidate
count is zero, with no residual/agreement claimed.65closed input/source and
diagnosis artifact bindings checked before/after. This uses saved current raw
witnesses and measured candidate points, no controller replay/nativepose/future
observations; it does not prove the complete controller intervention yet.

Eleventh native67399 is now complete, result
44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c.
Preflight30809 is running with this actual result SHA and --preflight-only.
All65 diagnosis input bindings must match that final native artifact map.
The replay has not established successful
continuation, a changed-action outcome or physical return. Protocol:
docs/go2_measured_floor_transport_prefix_v1_2026-09-09.md.
