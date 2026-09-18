# Readout for the target-reset-checked settling pilot

Run scripts/read_go2_settled_boundary_maze_pilot_v2.py only after native V2
collection, full raw audit and prefix comparison complete. Supply the actual
completed native result SHA-256. The exclusive destination is
go2_settled_boundary_maze_readout_v2_attempt_001.

This is the unchanged V1 prepared readout, with only native input, output,
protocol, script and terminal status identities updated for V2. It preserves
all executed route, intervention, pose accuracy, timing, failed/censored
intervals and native qualification accounting. It performs no controller,
model or outcome selection and cannot convert a failed run into a success.
The summarizer and numerical routines are identical. Revalidate exact input,
source and result bindings, check hardware, and retain8GiB RAM/128MiB output
above40GiB reserve admission. No native execution, independent-layout claim,
matched-comparison result or hardware qualification is implied by a readout.
