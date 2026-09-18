# Serial collection and audit consume comparable worker time

Four completed, audited development cases spend approximately half their worker
wall time in recorded collection iterations. The remaining half includes the
full raw controller/sensor/physics audit, input verification, scene/model setup,
artifact hashing and evidence persistence. It is not an isolated measurement of
audit duration and does not establish the speedup of an overlapping runtime.

| Case | Worker total | Recorded collection iterations | Other worker time | Other fraction |
| --- | ---: | ---: | ---: | ---: |
| Full JEPA | 5,773.8 s | 2,933.9 s | 2,839.9 s | 49.2% |
| Full supervised | 16,411.7 s | 8,246.6 s | 8,165.1 s | 49.8% |
| Full direct | 15,812.8 s | 7,938.5 s | 7,874.3 s | 49.8% |
| No-RGB JEPA | 2,087.5 s | 1,016.1 s | 1,071.5 s | 51.3% |

The assessment authenticated the four completed worker records and their four
collection timing streams. It checked complete ordered timing rows, summed every
recorded iteration, and subtracted that sum from the bound worker wall time.
All four original zero-round-trip results are retained. It also reverified the
2,009-source prepared independent-population runtime closure and recorded current
hardware availability. Worker peak RSS in these completed cases ranges from
approximately 3.91 to 8.44 GiB; that is not a bound on future concurrent peaks.

[Timing and hardware assessment](go2_serial_native_worker_parallelism_assessment_2026-09-10.json)
has SHA-256
`63023519436def081da46b5b89cbf8ca83fb510e7e735e8f878270800372d77b`.

The source review identifies three concrete constraints on useful overlap:

- `case_worker` collects, binds artifacts and performs the complete raw audit
  in the same process. The parent waits for that process to end before the next
  scene. Changing this requires a separately reviewed runtime and lifecycle
  receipts; it cannot be installed into the running batch.
- The current native-idle guard treats every multiprocessing-spawn worker as a
  potential simulator owner. A future coordinator would need an exact,
  authenticated collection/audit role distinction, not a generic worker
  exclusion or an inference from low GPU utilization.
- Within each independent layout, subsequent arms require the first arm's
  completed reference evidence. This imposes a real dependency barrier. Any
  overlap must preserve the frozen case order, reference validation and all
  scientific failures, with a bounded number of active jobs and no retries.

One collection process and one separate CPU audit process are therefore a
reasonable future design to assess before a large population. First prove that
the audit execution path starts no simulator scene, bind immutable collected
artifacts and both closed logs, enforce distinct process identities and resource
admission, and preserve an already active collection if an earlier audit fails.
Measure the staged implementation rather than claiming an approximately 2×
gain from this time decomposition. Timing comparisons must record competing
workload; the current runs are not isolated hardware benchmarks.

No staged runtime or new native job was started by this assessment. The current
batch, full replay audit and frontier/hold/contact/tracking queue retain their
original ownership. Faster controller queries remain useful in both collection
and replay audit, and reliable navigation still takes precedence over spending
the independent-layout population.
