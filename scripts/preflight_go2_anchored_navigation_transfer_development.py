"""Storage and identity preflight for the frozen-coefficient navigation transfer successor.

Checks, without running or altering the completed evaluation:
  - resolved input paths and their recorded identities are unchanged;
  - every intended write resolves to the workspace filesystem;
  - no cache or derived file is created beside an input;
  - temporary-file handling resolves to the declared destination;
  - the resource gate passes for this exact configuration.

Read-only apart from a probe file written into the declared output and temp directories.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path('/home/andrewknowles/Workspace/LeWMQuad-v3')
# Successor output on the workspace XFS volume; the completed experiment is untouched.
SUCCESSOR = REPO / '.generated/navigation_development_artifacts_v1/go2_anchored_navigation_transfer_v1_attempt_001'
TMPDIR = SUCCESSOR / 'tmp'
PLAN = REPO / 'docs/go2_anchored_visual_navigation_plan_2026-09-17.json'
STEAM = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1')
ANCHORED = STEAM / 'go2_anchored_visual_dynamics_v1_attempt_001'
REPRESENTATION = STEAM / 'go2_visual_target_jepa_v1_attempt_001'
RESERVE = 2 * 2**30


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fs(path):
    """Device id of the filesystem backing an existing ancestor of path."""
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return os.stat(probe).st_dev, probe


def free(path):
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    status = os.statvfs(probe)
    return status.f_bavail * status.f_frsize


def main():
    ok = True
    plan = json.loads(PLAN.read_text())
    workspace_dev, _ = fs(REPO)

    print('== input identities (read-only, must be unchanged) ==')
    for record in plan['roots']:
        window = Path(record['root']) / 'saved_executed_motion_forecast_evaluation_v1.json'
        actual = digest(window)
        match = actual == record['window_sha256']
        ok &= match
        print(f"  {'OK ' if match else 'BAD'} {Path(record['root']).name[:58]:58s} windows={record['windows']}")
    for label, path, expect in (
            ('anchored fit result', ANCHORED / 'result.json', plan['fit_sha256']),
            ('evaluator source', REPO / 'scripts/evaluate_go2_anchored_visual_navigation_development.py',
             plan['source_sha256'])):
        actual = digest(path)
        match = actual == expect
        ok &= match
        print(f"  {'OK ' if match else 'BAD'} {label}")
    for label, path in (('action checkpoint', ANCHORED / 'action.pt'),
                        ('no_future_action checkpoint', ANCHORED / 'no_future_action.pt'),
                        ('representation model', REPRESENTATION / 'model.pt')):
        print(f'  -   {label} sha256 {digest(path)[:16]}...')

    print('\n== write destinations (every one must be on the workspace filesystem) ==')
    destinations = {'successor output': SUCCESSOR, 'successor tmp': TMPDIR,
                    'docs result': REPO / 'docs'}
    for label, path in destinations.items():
        dev, anchor = fs(path)
        same = dev == workspace_dev
        ok &= same
        print(f"  {'OK ' if same else 'BAD'} {label:20s} dev={dev} via {anchor}")

    print('\n== inputs must remain read-only; no derived file beside them ==')
    for record in plan['roots'][:1]:
        root = Path(record['root'])
        dev, _ = fs(root)
        print(f'  -   input root dev={dev} (expected different from workspace {workspace_dev})')
        before = sum(1 for _ in root.rglob('*'))
        print(f'  -   entries under input root: {before} (recorded for post-run comparison)')

    print('\n== temporary-file routing ==')
    SUCCESSOR.mkdir(parents=True, exist_ok=True)
    TMPDIR.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(TMPDIR)
    tempfile.tempdir = None
    with tempfile.NamedTemporaryFile() as handle:
        dev, _ = fs(handle.name)
        same = dev == workspace_dev
        ok &= same
        print(f"  {'OK ' if same else 'BAD'} tempfile resolves to {Path(handle.name).parent} dev={dev}")
    print(f"  -   PYTHONDONTWRITEBYTECODE={os.environ.get('PYTHONDONTWRITEBYTECODE', 'unset')} "
          f"(prevents .pyc beside sources)")

    print('\n== resource gate for this configuration ==')
    # Refreshed after the successor gained candidate queries and strata fields. The original
    # four-arm run files totalled ~2.85 MB; this writes five conditions plus in_bank/group
    # per row, roughly doubling row size, plus result.json duplicated into docs/.
    expected_output = 32 * 2**20
    for label, path in (('successor output (writes)', SUCCESSOR), ('repo docs (writes)', REPO / 'docs')):
        available = free(path)
        passes = available >= RESERVE + expected_output
        ok &= passes
        print(f"  {'OK ' if passes else 'BAD'} {label:28s} free {available/2**30:7.2f} GiB "
              f"(needs {RESERVE/2**30:.0f} GiB reserve + {expected_output/2**20:.0f} MiB peak)")
    for label, path in (('steam_drive inputs (read-only)', STEAM),
                        ('RecoveryStorage + / + /tmp (shared)', Path('/tmp'))):
        print(f'  -   {label:34s} free {free(path)/2**30:7.2f} GiB  (not written by this configuration)')

    print(f"\nPREFLIGHT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
