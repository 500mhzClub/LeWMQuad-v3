"""P1 runtime trace for the research-artefact migration (6 October 2026; docs/go2_research_artifact_migration_plan_2026-10-06.md).

Runs a pinned launcher (unchanged) under coverage, so every Python process the cohort starts (mission entries, spawned
pose/mapping/registration workers, closeout readers) records which repository lines it executed. The result is the
porting list for the clean repository; the static closure over-approximates it.

How: coverage is installed outside the pinned environment (`<capability root>/p1_runtime_trace/site`, with a
sitecustomize that calls coverage.process_startup()). The cohort runner forces PYTHONPATH for its subprocesses through the
shared ENVIRONMENT dict; this wrapper prepends the coverage directory to that dict in this process only, sets
COVERAGE_PROCESS_START and a per-group COVERAGE_FILE, then calls the launcher's main() with the remaining arguments.
Nothing pinned changes, and missions are ordinary development runs (named p1t_*), labelled as trace runs.

Usage: trace_go2_p1_runtime_development.py GROUP LAUNCHER_MODULE -- LAUNCHER ARGS...
"""
import importlib
import os
from pathlib import Path
import sys

TRACE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
             'go2_navigation_capability_v1_attempt_001/p1_runtime_trace')


def main():
    group, launcher = sys.argv[1], sys.argv[2]
    assert sys.argv[3] == '--'
    data = TRACE/'data'/group
    data.mkdir(parents=True, exist_ok=True)
    os.environ['COVERAGE_PROCESS_START'] = str(TRACE/'coveragerc')
    os.environ['COVERAGE_FILE'] = str(data/'cov')
    from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT
    ENVIRONMENT['PYTHONPATH'] = f"{TRACE/'site'}:{ENVIRONMENT['PYTHONPATH']}"  # this process only; shared dict object
    module = importlib.import_module(launcher)
    sys.argv = [module.__file__]+sys.argv[4:]
    module.main()


if __name__ == '__main__':
    main()
