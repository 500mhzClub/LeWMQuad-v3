# Read-only command-encoding correction to the completed context-pilot audit

The twelve-episode collector completed normally. Its first auditor terminated
before completing any episode because it compared float64 native requested
commands to float32-rounded expectations. For the0.12 warm-up command this creates
a2.68220901e-9 difference. The recorder explicitly stores requested commands as
float64; applied actuator commands undergo the separate frozen float32/slew path.
The synthetic fixture used float32 requested commands and missed this distinction.

Preserve collector launch/result and the failed audit, with their exact SHA256
identities declared in the correction script. Do not modify or rerun the frozen
collector/auditor. This distinct read-only correction requires float64 requested
trace encoding and compares every command interval to the exact float64 command
tape. There is no new tolerance, cast of the recorded request, altered command,
changed trajectory, changed target definition, threshold relaxation or new physics.
Keep the existing separate applied-command slew comparison unchanged.

Reuse all original setup, raw sensor/contact, stop, prefix and target checks.
Write only distinct context_encoding_correction_v1 audit/evaluation/window/target/
prefix artifacts alongside the completed collection. Bind the original failures
and all collector artifacts before and after; freeze the correction sources,
this protocol and focused tests. No training, runtime promotion or navigation
claim. Zero contacts and missing/unequal prefixes must still be reported as such.

Tests must reproduce the original false rejection on a correctly encoded trace,
reject even one native request altered to its float32-rounded value, check all
six action-duration cells, and pass complete raw audit of at least one actual
recorded episode before this new correction audit is launched.
