"""Read-only actual-input admission probe; no controller replay or profiler launch."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import psutil

from scripts import body_projection_external_profile_admission_development as admission
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json

OUTPUT = ROOT/'docs/go2_body_projection_external_profile_actual_admission_2026-09-11.json'
EXECUTION = ROOT/'docs/go2_body_projection_external_profile_actual_admission_execution_2026-09-11.json'
FAILURE = ROOT/'docs/go2_body_projection_external_profile_actual_admission_failure_2026-09-11.json'
SOURCES = (
    'scripts/check_go2_body_projection_external_profile_admission_v1.py',
    'scripts/body_projection_external_profile_admission_development.py',
    'lewm/tests/test_body_projection_external_profile_admission_development.py',
    'docs/go2_body_projection_owned_profiler_admission_2026-09-11.md',
)


def main():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, EXECUTION, FAILURE)):
        raise ValueError('exclusive actual-admission probe required')
    environment = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in environment.items()):
        raise ValueError('original deterministic environment required')
    sources = {name:digest(ROOT/name) for name in SOURCES}
    process = psutil.Process()
    write_json(EXECUTION, dict(utc=datetime.now(timezone.utc).isoformat(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        explicit_source_sha256=sources, environment=environment,
        reference_verification_sha256=admission.VERIFICATION_SHA,
        controller_execution=False, profiler_execution=False))
    print('BODY_PROJECTION_EXTERNAL_ACTUAL_ADMISSION_STARTED', digest(EXECUTION), flush=True)
    start = time.perf_counter()
    try:
        witness, result, launch, rows = admission.admit_completed()
        verify(sources)
        write_json(OUTPUT, dict(status='BODY_PROJECTION_EXTERNAL_ACTUAL_ADMISSION_VERIFIED',
            utc=datetime.now(timezone.utc).isoformat(), execution_sha256=digest(EXECUTION),
            explicit_source_sha256=sources, original_source_count=len(witness['source_sha256']),
            reference_verification_sha256=admission.VERIFICATION_SHA,
            original_completion_reconstructed=True, original_rows=len(rows),
            sensing_scope=witness['sensing_scope'], wall_s=time.perf_counter()-start,
            actual_raw_and_model_bindings_rehashed=True, controller_execution=False,
            profiler_execution=False, navigation_qualified=False, real_time_qualified=False,
            full_training_ancestry_reexecuted=False, goal_achieved=False))
        print('BODY_PROJECTION_EXTERNAL_ACTUAL_ADMISSION_VERIFIED', digest(OUTPUT), flush=True)
    except BaseException as error:
        write_json(FAILURE, dict(status='TERMINAL_BODY_PROJECTION_EXTERNAL_ACTUAL_ADMISSION_FAILURE',
            reason=repr(error), execution_sha256=digest(EXECUTION), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
