#!/usr/bin/env python3
"""Full audit with an explicit binding for its three inherited audit helpers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'.generated/go2_successive_choice_maze_development_v1_attempt_001'
DEPENDENCIES={
    'scripts/audit_go2_successive_choice_maze_development_v1.py':'79253302c8d59fd066ae451ab2e78dc3bd3d1ab45ac6cdc4cd98aacdbc9026f4',
    'lewm/tests/test_successive_choice_raw_audit.py':'96c8d5bc664496b5c872d1c31cb29fac6ba543ef40fbc3d80a248a3d1cc830e3',
    'scripts/audit_go2_contact_attributed_execution_development_v1.py':'fd465ca68abd8a8021554cf431d2c8f84e5b137dc64fcbabde4506dde5f8b1a5',
    'scripts/audit_go2_local_control_factorial_development_v1.py':'e804e86ff34978f2b162a7ab9acc03650eb3aeaf39a2f890d854466e89ea8311',
    'scripts/audit_go2_causal_rgb_body_capture_development_v1.py':'db244087bbd2238d7892e41e7716c50d0c19d93168040b2aa0c26c69eba091ed',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bindings):
    for name,expected in bindings.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in p.parts):
            raise ValueError('nonprotected explicit relative source required')
        path=ROOT/p
        if path.resolve()!=path or sha(path)!=expected: raise ValueError('source binding changed: '+name)


def main():
    if len(sys.argv)!=1: raise ValueError('fixed full audit only')
    if sha(OUTPUT/'launch.json')!='e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5':
        raise ValueError('physical launch identity changed')
    launch=json.loads((OUTPUT/'launch.json').read_text())
    bindings=launch['source_sha256']|DEPENDENCIES|{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    witness=OUTPUT/'full_audit_source_dependency_witness.json'
    if witness.exists() or (OUTPUT/'raw_artifact_audit.json').exists(): raise ValueError('full audit or witness already exists')
    verify(bindings)
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_go2_successive_choice_maze_development_v1.py')],cwd=ROOT,check=True)
    verify(bindings)
    audit=json.loads((OUTPUT/'raw_artifact_audit.json').read_text())
    if audit['status']!='PASS' or not audit['full_study'] or audit['audited_trials']!=144: raise ValueError('full audit not passed')
    with witness.open('x') as stream:
        json.dump({'status':'PASS','source_sha256':bindings,'full_audit_sha256':sha(OUTPUT/'raw_artifact_audit.json'),
            'scope':'source identities checked before and after full audit; supplemental helper binding, not retrospective source freeze for interim36'},stream,indent=2)
        stream.write('\n')
    print(json.dumps({'full_audit':'PASS','source_witness_sha256':sha(witness),'source_paths':len(bindings)}),flush=True)


if __name__=='__main__': main()
