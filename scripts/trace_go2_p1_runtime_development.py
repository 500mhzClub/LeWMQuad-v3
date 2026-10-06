"""P1 runtime trace for the research-artefact migration (6 October 2026; docs/go2_research_artifact_migration_plan_2026-10-06.md).

Runs a pinned launcher (unchanged) so that every mission process it starts records, under coverage, which repository
lines it executed (including spawned pose/mapping/registration workers, through coverage's multiprocessing support). The
result is the porting list for the clean repository; the static closure over-approximates it.

The frozen harness verifies its environment before each episode: exact Python distributions, and an exact PYTHONPATH and
render environment. So nothing in the environment changes:
- coverage lives outside the pinned environment (`<capability root>/p1_runtime_trace/site`), with its dist-info metadata
  moved aside so the distribution list is unchanged;
- this wrapper rewrites, in its own process only, each subprocess command of the form [python, X.py, args...] into
  [python, boot, X.py, args...]; `boot` (this module, run as a script) appends the coverage directory to sys.path, starts
  coverage, restores sys.argv and sys.path[0] as `python X.py` would have them, and runs X.py as __main__.
Missions are ordinary development runs (cohorts named p1t2_*), labelled as trace runs.

Usage: trace_go2_p1_runtime_development.py GROUP LAUNCHER_MODULE -- LAUNCHER ARGS...
"""
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys

TRACE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
             'go2_navigation_capability_v1_attempt_001/p1_runtime_trace')


def boot():
    """Run as `python THIS_FILE --boot SCRIPT ARGS...`: start coverage, then run SCRIPT as `python SCRIPT ARGS` would."""
    script = sys.argv[2]
    sys.path.append(str(TRACE/'site'))
    import coverage
    cov = coverage.Coverage(config_file=str(TRACE/'coveragerc'), data_file=os.environ['P1_COVERAGE_FILE'])
    cov.start()
    import atexit
    atexit.register(lambda: (cov.stop(), cov.save()))
    sys.argv = [script]+sys.argv[3:]
    sys.path[0] = str(Path(script).resolve().parent)
    runpy.run_path(script, run_name='__main__')


def main():
    group, launcher = sys.argv[1], sys.argv[2]
    assert sys.argv[3] == '--'
    data = TRACE/'data'/group
    data.mkdir(parents=True, exist_ok=True)
    os.environ['P1_COVERAGE_FILE'] = str(data/'cov')
    original = subprocess.run

    def run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 1 and command[0] == sys.executable and str(command[1]).endswith('.py'):
            command = [command[0], str(Path(__file__).resolve()), '--boot', *command[1:]]
        return original(command, *args, **kwargs)
    subprocess.run = run  # this process only
    module = importlib.import_module(launcher)
    sys.argv = [module.__file__]+sys.argv[4:]
    module.main()


if __name__ == '__main__':
    boot() if sys.argv[1] == '--boot' else main()
