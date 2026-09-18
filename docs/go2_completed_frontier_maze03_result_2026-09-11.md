# Completed reached-frontier maze03 diagnostic

The reached-frontier intervention activated and changed physical execution, but
did not complete navigation. Four observed frontier cells were retired at frames
134, 165, 193 and 198. The first intervention changed the requested right turn to
a left turn, which was executed. The original 135-observation comparison records
identical public observations, physics and requested commands before that change.

The run ended at frame 290 because no action satisfied both the sampled-surface
and nominal-clearance constraints. All six actions passed the sampled-surface
filter, but all six failed the nominal disk check from frames 280 through 290.
The current observed position itself first failed that disk check at frame 281.
At frame 279 the selected right turn still passed with 0.450444 m predicted
minimum clearance against the 0.45 m radius; at frame 280 even the best candidate
had only 0.448296 m. This locates the failure near an almost exhausted clearance
margin, after successful frontier transitions.

At terminal frame 290 the current nominal clearance was 0.447430 m. The six
recorded first-segment clearances ranged from 0.441898 to 0.444921 m. Each already
worsened current clearance, so the existing non-worsening reentry requirement
would reject these recorded trajectories even before considering their later
segments. Shortening the prediction check to the first segment alone therefore
does not resolve this terminal receipt. This is a constraint diagnosis from
recorded predictions, not a replay of a successor controller or evidence about
the actual outcome of any unexecuted action.

There were 301 observations, 300 commands and 15,750 physics samples, including
ten zero-command terminal drain observations. There were no maze-edge crossings,
arrivals, round trips or recorded native contacts. Strict physical visibility
passed. Median observation-and-control time was 948.281 ms; all 301 measurements
exceeded the 100 ms command interval. Navigation and real-time qualification
remain false.

Verification used `scripts/diagnose_go2_completed_frontier_maze03_v1.py` and
completed with exit code zero. The original completion verifier was rerun, all
1,846 native artifacts were rehashed, all 301 decision rows were read in order,
and the physical-contact readout was reconstructed. The 1,930-path diagnostic
source closure was verified. The original raw-sensor/model audit, training
ancestry and physical-prefix comparison were authenticated but not rerun.

The machine-readable diagnostic is
`docs/go2_completed_frontier_maze03_diagnosis_2026-09-11.json`, SHA-256
`dc8b8ce1678e20ab763e078dbd82b49c33aa22533157c34ad22af611614e668a`.
The native result SHA-256 is
`262fbd020dd2407e26de2a7801b53462655bf47afb9075bd9df7dd7af52352db`.
This experiment used its original assigned model
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`,
not the newer six-case batch JEPA model.

The next frontier investigation should examine approach clearance and measured
versus predicted motion before frame 280, retaining the original radius and
sampled-surface constraints. Zero observed contacts do not establish that a
smaller clearance radius is appropriate. The original experiment and currently
queued diagnostics remain frozen; this result selects no replacement policy.
