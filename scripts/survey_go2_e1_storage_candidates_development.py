"""Storage candidates for E1 (Andrew, 30 Sep 2026): read-only survey, no file is changed.

E1 must start with at least 15 GiB above the 12-GiB RecoveryStorage reserve, after its own
projected footprint. The survey lists the largest artifact directories on the RecoveryStorage
filesystem and marks each one "protected" if the current programme references it.

References are searched in:
- the frozen V4 harness's bound source files;
- the capability, C3-v2 and E1 documents;
- the current training lineage: C3-v2/C4-v2 matched data, C4-v1 preparation, the C3 readout
  and predictor training records, and the transfer set;
- the scripts E1 would run for seeds (readout, C4 and predictor training) and their imports.

Also protected:
- the decision-headroom V4.2 audit outputs (Andrew: do not touch);
- the capability programme's own artifact root.
A candidate is anything else, listed for Andrew's approval only.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

REPO = owner.REPO
BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
ARTIFACTS = BASE.parent
LINEAGE_FILES = [
    BASE/'c3v2_data_v1/frame_paths.json', BASE/'c3v2_data_v1/train_samples.json', BASE/'c4_preparation/frame_paths.json',
    ARTIFACTS/'go2_maze_view_readout_v1_attempt_003/frame_paths.json', ARTIFACTS/'go2_maze_view_readout_v1_attempt_003/plan.json',
    REPO/'.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/frame_paths.json',
    REPO/'.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/plan.json',
    ARTIFACTS/'go2_maze_view_transfer_v1_attempt_001/transfer_targets.json',
]
E1_SCRIPTS = ['scripts/train_go2_c3v2_readout_development.py', 'scripts/train_go2_c4v2_development.py',
              'scripts/train_go2_horizon_dense_predictor_development.py', 'scripts/evaluate_go2_c3v2_acceptance_development.py',
              'scripts/run_go2_c3v2_check_development.py']
PROTECTED_PREFIXES = ('go2_decision_headroom', 'go2_headroom_')  # V4.2 audit and its lineage: do not touch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imports(script, seen):
    """Repository modules a script imports, transitively."""
    path = REPO/script
    if script in seen or not path.exists():
        return
    seen.add(script)
    for node in ast.walk(ast.parse(path.read_text())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module+'.'+a.name for a in node.names]+[node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            for candidate in (name.replace('.', '/')+'.py',):
                if candidate.split('/')[0] in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'):
                    imports(candidate, seen)


def corpus():
    freeze = json.loads(owner.FREEZE.read_text())
    texts = [(REPO/name).read_text(errors='ignore') for name in list(freeze['original_source_bindings'])+list(freeze['implementation_bindings'])
             if (REPO/name).exists()]
    texts += [p.read_text(errors='ignore') for p in (REPO/'docs').glob('go2_navigation_*') if p.is_file() and p.stat().st_mtime > 0
              and any(k in p.name for k in ('capability', 'c3v2', 'e1_', 'harness_v4'))]
    texts += [p.read_text(errors='ignore') for p in LINEAGE_FILES if p.exists()]
    seen = set()
    for script in E1_SCRIPTS:
        imports(script, seen)
    texts += [(REPO/s).read_text(errors='ignore') for s in seen]
    return '\n'.join(texts), len(seen)


def main():
    output.install(BASE)
    text, e1_modules = corpus()
    sizes = subprocess.run(['du', '-s', '--block-size=1M', *[str(p) for p in sorted(ARTIFACTS.iterdir()) if p.is_dir()]],
                           capture_output=True, text=True).stdout.split('\n')
    rows = []
    for line in sizes:
        if not line.strip():
            continue
        mib, path = line.split('\t')
        name = Path(path).name
        referenced = name in text
        protected = referenced or name.startswith(PROTECTED_PREFIXES) or Path(path) == BASE
        rows.append(dict(directory=name, mib=int(mib), referenced_by_current_programme=referenced,
                         protected=protected, reason=('capability artifact root' if Path(path) == BASE else
                                                      'V4.2 audit / decision-headroom lineage' if name.startswith(PROTECTED_PREFIXES) else
                                                      'referenced by current programme' if referenced else 'not referenced')))
    rows.sort(key=lambda r: -r['mib'])
    free = shutil.disk_usage(BASE).free
    projection = json.loads((BASE/'e1_projection/projection_v2_10c3_with_analysis_fits.json').read_text())
    need = 12*1024**3+15*1024**3+projection['storage']['projected_bytes']
    result = dict(schema='e1_storage_candidates.v1', free_bytes=free, e1_projected_bytes=projection['storage']['projected_bytes'],
                  required_free_bytes_at_e1_start=need, shortfall_bytes=max(0, need-free),
                  candidates=[r for r in rows if not r['protected']][:40], protected_largest=[r for r in rows if r['protected']][:20],
                  e1_dependency_modules=e1_modules, survey_sha256=sha(__file__))
    root = BASE/'e1_storage'
    root.mkdir(exist_ok=True)
    owner.save(root/'candidates.json', result)
    print(json.dumps(dict(free_gib=free/2**30, need_gib=need/2**30, shortfall_gib=result['shortfall_bytes']/2**30), indent=1))
    for r in result['candidates'][:30]:
        print(f"{r['mib']/1024:6.1f} GiB  {r['directory']}")
    print('protected (largest):')
    for r in result['protected_largest'][:12]:
        print(f"{r['mib']/1024:6.1f} GiB  {r['directory']}  [{r['reason']}]")


if __name__ == '__main__':
    main()
