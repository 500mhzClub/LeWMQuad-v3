# Support friction V1: zero-step source/native preflight

Prepare two matched fresh conditions (nominal1.0 and lower0.15). Use common
physics seed2026090641 and appearance seed2026090643, existing fitting-scene
geometry/spawn and unchanged gait. This pairing is one controlled intervention,
not two independent layouts. Each condition has a new declared scene identity.

Genesis combines contact coefficients using the maximum of the two geometries.
Set BOTH actual robot and ground coefficients before physics via supported
entity setters. Read solver coefficients and per-environment friction ratios,
as well as cached geometry metadata. Require all27robot+1ground coefficients
equal the declared value (1e-7 numeric readback tolerance) and all ratios exactly1.
All robot geometries change, not only feet; this also changes self-contact
friction if self-contact occurs. Walls remain unchanged. Do not call it a
floor-only intervention or infer slip from a material value alone.

This launch builds each new live-sensor session, installs contact/transducer
identities, checks unchanged checkpoint actuator gains and reads native friction.
It must execute ZERO physics steps, RGB renders or model observations. Persist
explicit visual-mesh and native-identity witnesses; no recursive artifact scan.
No native runtime monkeypatch, source export, global setter replacement or
checkpoint training. Preserve failure and do not rerun this attempt.

The prepared collector schedule has15settle ticks plus225fixed100ms commands:
10quiet,120forward(.12,0,0),10brake,30left(0,0,.25),10brake,
30right(0,0,-.25),10brake,5zero-tail. A full future collection is24s12000samples,
226RGB-D observations and1126support predictions from1.5s. The frozen centre/
rolling hypotheses only observe; they cannot select commands or release gates.
The source preflight does NOT execute this schedule. A separate collector and
raw audit must be frozen before that physical challenge starts.

The new session acquires all incident foot loads at500Hz, including terminal
stop rows. Native geometry/orientation are sensor-generation inputs only; the
consumer uses current body/gyro/load samples. Sensor-prediction faults latch
without reset. Native body/contact/speed/domain stop supervision remains separate.
The forward RGB-D renderer is retained for compatibility with prior motion
measurements, with its hidden-robot/aperture limitations explicitly unchanged;
no proposed downward view or visual qualification is introduced here.

The scientific objective still requires local sensor-only execution, memory,
novel-maze JEPA and genuine multistep/matched-baseline comparisons, independent
layouts/seeds, timing and bounded hardware. Neither preflight nor challenge
collection will count as that endpoint.
