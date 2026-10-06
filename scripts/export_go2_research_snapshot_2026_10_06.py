"""Frozen reproduction snapshot (artefact A) of the navigation-capability programme (consolidation, 6 October 2026).

Migration plan: docs/go2_research_artifact_migration_plan_2026-10-06.md, section 3, artefact A: "the exact closure at a frozen
commit, exported by an explicit, SHA-256-checked file manifest ... with fresh history and a provenance manifest". It is the
oracle the clean repository (Go2-JEPA-Navigation) is gated against, and stays local.

**What is exported:** an explicit list, never a tree.
- The static closure (scripts/trace_go2_research_artifact_closure_2026_10_06.py, all groups): Python and shell files, and
  the non-code files they read (docs bindings, configs, assets, models).
- Explicit extras: the locomotion policy checkpoint folder's files, the platform configs, the tracked CC0 textures, the
  programme's results documents, `.ignore`, and the migration plan and P1 trace.

**How:** for each path, by name only first: any sealed-pattern path (`sealed_test.json`, a `sealed/` or `sealed_*`
directory) is refused before anything is read. Then the file's blob at the frozen commit (`git show COMMIT:path`) must
have the same SHA-256 as the working-tree file, the copy is written file by file, and its SHA-256 is checked again.
No archive, worktree, checkout, recursive copy or wildcard copy is used. The destination must not exist.

**Provenance:** SNAPSHOT_MANIFEST.json lists every file (path, sha256, bytes) with the source repository, branch and commit.
The destination is initialised as a new git repository with one commit (fresh history: the source history contains legacy
sealed blobs).

Usage: export_go2_research_snapshot_2026_10_06.py --closure CLOSURE.json --dest DIR [--dry-run]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
SEALED = re.compile(r'(^|/)(sealed_test\.json$|sealed/|sealed_[^/]*/)')
EXTRA = [
    '.ignore', 'config/go2_platform_manifest.yaml',
    'models/tier_a_go2_locomotion/20260516_contract_ppo/README.md', 'models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl',
    'models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt',
    'docs/CURRENT_RESEARCH_BRIEF.md', 'docs/go2_navigation_capability_handoff_2026-09-28.md',
    'docs/go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md',
    'docs/go2_navigation_preliminary_results_2026-10-02.md', 'docs/go2_navigation_preliminary_results_tables_2026-10-02.md',
    'docs/go2_navigation_preliminary_budget_rescore_2026-10-02.md', 'docs/go2_navigation_forecast_sensitivity_2026-10-02.md',
    'docs/go2_navigation_forecast_sensitivity_tables_2026-10-02.md', 'docs/go2_navigation_forecast_sensitivity_2026-10-02.png',
    'docs/go2_navigation_calibrated_margin_results_2026-10-02.md', 'docs/go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md',
    'docs/go2_navigation_reserve_exit_rerun_results_2026-10-03.md', 'docs/go2_navigation_harness_reserve_exit_plan_2026-10-02.md',
    'docs/go2_navigation_harness_v4_known_limitations_2026-09-29.md', 'docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md',
    'docs/go2_navigation_dynamics_stage1_characterisation_2026-10-03.md', 'docs/go2_navigation_dynamics_stage1_results_2026-10-04.md',
    'docs/go2_navigation_dynamics_stage1_v2_results_2026-10-05.md', 'docs/go2_navigation_stage2_strip_stall_diagnosis_2026-10-05.md',
    'docs/go2_navigation_dynamics_patch_marker_start_frame_2026-10-03.png', 'docs/go2_storage_review_2026-10-04.md',
    'docs/go2_development_artifact_retention_2026-09-14.md', 'docs/go2_research_artifact_migration_plan_2026-10-06.md',
    'docs/go2_research_artifact_p1_runtime_trace_2026-10-06.json',
    'scripts/trace_go2_research_artifact_closure_2026_10_06.py', 'scripts/discover_go2_lineage_scripts_2026_10_06.py',
    'scripts/export_go2_research_snapshot_2026_10_06.py', 'scripts/trace_go2_p1_runtime_development.py',
    'docs/go2_research_artifact_closure_2026-10-06.json', 'docs/go2_research_artifact_lineage_scripts_2026-10-06.json',
]


def git(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, check=True).stdout
    return out if binary else out.decode()


def main(closure, dest, dry_run):
    report = json.loads(Path(closure).read_text())
    commit, branch = git('rev-parse', 'HEAD').strip(), git('rev-parse', '--abbrev-ref', 'HEAD').strip()
    tracked = set(git('ls-files').splitlines())
    textures = sorted(p for p in tracked if p.startswith('assets/textures/') and not SEALED.search(p))
    paths = sorted(set(report['python_files']) | set(report['data_files']) | set(EXTRA) | set(textures))
    refused = [p for p in paths if SEALED.search(p)]
    paths = [p for p in paths if not SEALED.search(p)]
    missing = [p for p in paths if p not in tracked]
    if missing:
        raise SystemExit(f'not tracked: {missing[:10]}')
    if git('status', '--porcelain', '--', *paths).strip():
        raise SystemExit('exported paths differ from HEAD; commit first')
    dest = Path(dest)
    if not dry_run and dest.exists():
        raise SystemExit(f'destination exists: {dest}')
    entries = []
    for p in paths:
        blob = git('show', f'{commit}:{p}', binary=True)
        sha = hashlib.sha256(blob).hexdigest()
        if hashlib.sha256((REPO/p).read_bytes()).hexdigest() != sha:
            raise SystemExit(f'working file differs from commit: {p}')
        entries.append(dict(path=p, sha256=sha, bytes=len(blob)))
        if not dry_run:
            target = dest/p
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            os.chmod(target, (REPO/p).stat().st_mode & 0o777)
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise SystemExit(f'copy check failed: {p}')
    manifest = dict(schema='go2_research_snapshot_manifest.v1', source_repository='git@github.com:500mhzClub/LeWMQuad-v3.git',
                    source_branch=branch, source_commit=commit, files=len(entries), bytes=sum(e['bytes'] for e in entries),
                    sealed_paths_refused_by_name=len(refused), closure_groups=sorted(report['groups']), entries=entries)
    summary = {k: v for k, v in manifest.items() if k != 'entries'}
    if dry_run:
        print(json.dumps(summary, indent=1))
        return
    (dest/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
    (dest/'README_SNAPSHOT.md').write_text(
        '# LeWMQuad-v3 frozen reproduction snapshot (6 October 2026)\n\n'
        f'Exact files of the navigation-capability programme at {branch} {commit}, exported by explicit, SHA-256-checked '
        'manifest (`SNAPSHOT_MANIFEST.json`). It is the oracle the clean repository Go2-JEPA-Navigation is gated against.\n\n'
        'See `docs/go2_research_artifact_migration_plan_2026-10-06.md`. Sealed material is excluded by name and never read. '
        'Absolute artefact paths (RecoveryStorage, workspace and data drives) are those of the original machine.\n')
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=dest, check=True)
    subprocess.run(['git', 'add', '-A'], cwd=dest, check=True)
    subprocess.run(['git', 'commit', '-q', '-m',
                    f'Frozen reproduction snapshot of LeWMQuad-v3 {branch} {commit[:12]} ({len(entries)} files, manifest-exported)'],
                   cwd=dest, check=True)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--closure', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    main(a.closure, a.dest, a.dry_run)
