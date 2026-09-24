"""Inspect protected process descriptors as root, then retire approved cache as its owner."""
import argparse
import json
import os
from pathlib import Path
import psutil
import shutil
from scripts.geometry_cache_retirement_development import retire, validate
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, ROOT
from scripts.startup_raw_sensor_audit_development import read_json

PROPOSAL = ROOT/'docs/go2_gsd_cache_retirement_proposal_2026-09-08.json'
CACHE = Path('/home/andrewknowles/.cache/genesis/gsd')
OUTPUT = BASE/'go2_reviewed_geometry_cache_retirement_v1_attempt_001'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal-sha256', required=True)
    parser.add_argument('--authorization-file', required=True)
    parser.add_argument('--authorization-sha256', required=True)
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    if os.getuid() != 0 or os.geteuid() != 0:
        raise ValueError('administrator access required only for process inspection')
    target_uid = target_gid = 1000
    if digest(PROPOSAL) != args.proposal_sha256: raise ValueError('reviewed proposal identity required')
    proposal = json.loads(PROPOSAL.read_text()); authorization_path = Path(args.authorization_file)
    if (authorization_path.parent != ROOT/'docs' or authorization_path.is_symlink()
            or authorization_path.resolve() != authorization_path
            or not authorization_path.name.startswith('go2_gsd_cache_retirement_authorization_')
            or digest(authorization_path) != args.authorization_sha256):
        raise ValueError('exact separately recorded user authorization required')
    authorization = json.loads(authorization_path.read_text())
    assert authorization['explicit_user_approval'] is True
    assert authorization['proposal_sha256'] == args.proposal_sha256 and authorization['user_message']
    assert proposal['cache_root'] == str(CACHE) and proposal['status'] == 'CACHE_RETIREMENT_PROPOSAL_ONLY'
    for path, sha in proposal['implementation_sha256'].items():
        if digest(ROOT/path) != sha: raise ValueError('reviewed retirement implementation changed')
    # The authorized plan waits for this original native/audit job to finish.
    for pid in proposal['must_be_terminal_pids']:
        if psutil.pid_exists(pid): raise ValueError('original experiment process still present')
    blockers = []
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid == os.getpid(): continue
        try:
            if process.uids().real != target_uid: continue
            if any(f.path.startswith(str(CACHE)+'/') for f in process.open_files()):
                blockers.append(dict(pid=process.pid, reason='open GSD file'))
            if process.info['name'].startswith('python'):
                scripts = [Path(x).name for x in process.cmdline()[1:] if x.endswith('.py')]
                if any(n.startswith(('run_go2_', 'inspect_go2_')) for n in scripts):
                    blockers.append(dict(pid=process.pid, reason='possible active native task'))
        except psutil.NoSuchProcess: pass
        except psutil.AccessDenied: raise ValueError('cannot inspect current-user cache users')
    if blockers: raise ValueError('cache retirement requires quiescent native tasks: '+str(blockers))
    # Permanently drop administrator privileges before metadata validation or
    # creating any output. The frozen helper still requires ordinary UID-1000
    # ownership, single links and exact reviewed metadata for every cache leaf.
    os.setgroups([])
    os.setgid(target_gid)
    os.setuid(target_uid)
    if os.getresuid() != (target_uid, target_uid, target_uid) or os.getresgid() != (target_gid, target_gid, target_gid):
        raise ValueError('permanent return to cache owner required before mutation')
    inspection = Path(proposal['inspection_root']); validate_root(inspection)
    verify_artifacts(inspection, proposal['inspection_artifact_sha256'])
    entries = proposal['candidate_metadata']; keep = proposal['retained_current_geometry_keys']
    validate(CACHE, entries, keep)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive retirement record required')
    create_output(OUTPUT)
    write_json(OUTPUT/'authorization.json', authorization)
    write_json(OUTPUT/'proposal_identity.json', dict(proposal_sha256=args.proposal_sha256,
        authorization_sha256=args.authorization_sha256, candidate_count=len(entries),
        retained_current_geometry_keys=keep, free_bytes_before=shutil.disk_usage(CACHE).free))
    completed = []
    try:
        with (OUTPUT/'removed_names.jsonl').open('x') as log:
            def record(name):
                log.write(json.dumps(dict(name=name))+'\n'); log.flush(); completed.append(name)
            retire(CACHE, entries, keep, record)
        assert len(completed) == len(entries)
        write_json(OUTPUT/'result.json', dict(status='APPROVED_GEOMETRY_CACHE_RETIREMENT_COMPLETE',
            proposal_sha256=args.proposal_sha256, removed_count=len(completed),
            retained_current_geometry_keys=keep, experiment_artifacts_removed=False,
            historical_geometry_cache_entries_removed=True,
            nominal_allocated_bytes_retired=sum(x['allocated_bytes'] for x in entries.values()),
            free_bytes_after=shutil.disk_usage(CACHE).free))
        print('APPROVED_CACHE_RETIREMENT_COMPLETE', digest(OUTPUT/'result.json'), len(completed), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='CACHE_RETIREMENT_PARTIAL_FAILURE',
            removed_count=len(completed), reason=repr(error)))
        raise


if __name__ == '__main__': main()
