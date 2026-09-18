# Paced worker experiment: scheduling alone does not recover 10 Hz

The connected runtime consumed 61 actual paired camera/gyro packets from the
completed independent layout-0 JEPA journey at a paced 10 Hz. It ran pose,
floor registration, 2.5 Hz routing-map updates and 2.5 Hz action-conditioned
planning independently, while servicing expiring shadow requests at 50 Hz.
The model's two-interval zero prefix is enforced by nonoverlapping command
windows. The actual command deadline includes upstream processing and delivery.
Fresh observed nominal obstacle checks can veto a ready command.

Three implementations completed: threads in one process (session 60608), a
separate mapping process (75897), and separate pose and mapping processes
(41150). All exited zero, reproduced every raw and floor-registered pose
receipt apart from proposal-work counts, and retained unchanged model weights.
Each processed 61 tracking and registration jobs, 16 map jobs and 15 planning
jobs. Process initialization finished before the camera stream began; process
transfer time is included in parent-stage measurements.

| Runtime | Maximum tracking completion age | Maximum planning completion age | On-time / late plans | Nonzero shadow requests |
| --- | ---: | ---: | ---: | ---: |
| Threads | 930 ms | 1,143 ms | 1 / 14 | 0 |
| Separate mapping process | 836 ms | 1,051 ms | 1 / 14 | 0 |
| Separate pose and mapping processes | 773 ms | 984 ms | 1 / 14 | 0 |

The one on-time plan in each run encountered stale perception at dispatch.
All request loops remained responsive with zero skipped 20 ms request ticks;
maximum request computation stayed below 0.027 ms. This isolates the problem
from a blocked actuator callback. The pose pipeline cannot sustain this camera
rate on these inputs with the current computation, even with process isolation.

These are short shared-host experiments. Camera files were preloaded: real
acquisition and rendering costs are absent. Requests were shadow outputs and
did not cause the recorded motion. There is no new native navigation success,
real-time qualification or whole-body/ground-support clearance evidence.

Exact roots, hashes, service times and results are in
`docs/go2_paced_multirate_comparison_2026-09-13.json`. The native direct-model
comparison continues with its original controller. The next perception
experiment should reduce the feature workload and assess pose accuracy and
failure rate, rather than extending queue infrastructure or relaxing deadlines
to conceal the measured backlog.
