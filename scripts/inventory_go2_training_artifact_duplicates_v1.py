"""Byte-verify duplicate immutable training artifacts; never mutate an input."""
import hashlib
import json
import stat
import time
from collections import defaultdict
from pathlib import Path
from scripts.navigation_artifact_root_development import BASE, artifact_path, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify as verify_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_training_artifact_duplicate_inventory_v1_attempt_001'
INPUTS = {
    'go2_geometry_progress_family_v1_attempt_001':'376b2eeacfd5e741ba6b7a0e1b5e04399f782d16943d66ffdff92eada93ef0fb',
    'go2_moving_action_switch_family_v1_attempt_001':'e89ea05cb4590164bcdc48e45136f022cdc941a7b63ea13bedcd4cfc1e6dffa1',
    'go2_observation_horizon_fits_v1_attempt_001':'45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418',
}
SOURCES = ('scripts/inventory_go2_training_artifact_duplicates_v1.py',
    'lewm/tests/test_training_artifact_duplicate_inventory_development.py',
    'docs/go2_training_artifact_duplicate_inventory_v1_2026-09-09.md')
MAX_LEAVES = 60000
ALLOWANCE = 128*1024**2


def metadata(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('ordinary nonsymlink candidate required')
    return {name:getattr(value, 'st_'+name) for name in
        ('dev','ino','mode','uid','gid','size','nlink','mtime_ns','ctime_ns','blocks')}


def duplicate_groups(records):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row['root'],row['path'])
        if key in seen: raise ValueError('unique bound artifact path required')
        seen.add(key); m = row['metadata']
        if m['nlink'] != 1: continue
        groups[(row['sha256'],m['size'],m['dev'],m['mode'],m['uid'],m['gid'])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        if len(rows) < 2: continue
        if len({(r['metadata']['dev'],r['metadata']['ino']) for r in rows}) != len(rows):
            raise ValueError('single-link metadata contradicts repeated inode')
        # Keep the largest allocation, making the predicted saving conservative.
        rows = sorted(rows,key=lambda r:(-r['metadata']['blocks'],r['root'],r['path']))
        saving = sum(r['metadata']['blocks']*512 for r in rows[1:])
        if saving:
            result.append(dict(sha256=key[0],size_bytes=key[1],canonical=rows[0],
                duplicate_paths=rows[1:],potential_allocated_saving_bytes=saving))
    return result


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive inventory output required')
    sources = {p:digest(Path(p)) for p in SOURCES}; verify_sources(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 20*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+ALLOWANCE:
        raise ValueError('inventory4 GiB plus replay16 GiB and artifact reserve required')
    inherited = {}; records = []; summaries = []
    for name, sha in INPUTS.items():
        root = BASE/name; verify_artifacts(root,{'result.json':sha}); result = read_json(root,'result.json')
        verify_sources(result['source_sha256'])
        inherited[name] = result['source_sha256']
        if len(records)+len(result['artifact_sha256']) > MAX_LEAVES: raise ValueError('bounded manifest population required')
        for leaf, expected in result['artifact_sha256'].items():
            path = artifact_path(root,leaf)
            records.append(dict(root=name,path=leaf,sha256=expected,metadata=metadata(path)))
        summaries.append(dict(root=name,result_sha256=sha,bindings=len(result['artifact_sha256']),status=result['status']))
    groups = duplicate_groups(records)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_result_sha256=INPUTS,
        hardware=resources,scope='declared artifact paths in three exact completed training roots only',
        records=len(records),candidate_groups=len(groups),output_allowance_bytes=ALLOWANCE,
        input_mutations_permitted=False,sealed_access=False,model_loaded=False,native_execution=False))
    print('TRAINING_DUPLICATE_INVENTORY_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter(); checked=hashed_bytes=0
    try:
        for group in groups:
            for row in [group['canonical'],*group['duplicate_paths']]:
                path=artifact_path(BASE/row['root'],row['path'])
                if metadata(path)!=row['metadata']: raise ValueError('candidate metadata changed before hashing')
                if digest(path)!=group['sha256']: raise ValueError('candidate content differs from completed artifact binding')
                if metadata(path)!=row['metadata']: raise ValueError('candidate changed during hashing')
                checked+=1;hashed_bytes+=row['metadata']['size']
            if checked%512 < len(group['duplicate_paths'])+1:
                print('TRAINING_DUPLICATE_PATHS_VERIFIED',checked,flush=True)
        for group in groups:
            for row in [group['canonical'],*group['duplicate_paths']]:
                if metadata(artifact_path(BASE/row['root'],row['path']))!=row['metadata']:
                    raise ValueError('candidate population changed after hashing')
        for name,sha in INPUTS.items():
            verify_artifacts(BASE/name,{'result.json':sha});verify_sources(inherited[name])
        verify_sources(sources)
        proposal=dict(input_result_sha256=INPUTS,source_sha256=sources,groups=groups,
            proposed_operation='replace duplicate storage with same-volume hard links only after separate approval',
            all_input_paths_retained=True,all_input_contents_retained=True,
            input_mutations_performed=False,regeneration_claimed=False,approval_granted=False,
            requires_no_active_artifact_consumers_at_execution=True,
            requires_fresh_content_and_metadata_recheck=True,
            storage_sharing_requires_artifacts_remain_immutable=True,
            verified_paths=checked,hashed_bytes=hashed_bytes,
            potential_allocated_saving_bytes=sum(g['potential_allocated_saving_bytes'] for g in groups))
        if len(json.dumps(proposal).encode()) > ALLOWANCE//2: raise ValueError('bounded proposal output required')
        write_json(OUTPUT/'proposal.json',proposal)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json','proposal.json')}
        verify_artifacts(OUTPUT,bindings)
        report=dict(status='TRAINING_ARTIFACT_DUPLICATE_INVENTORY_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,input_roots=summaries,groups=len(groups),
            verified_paths=checked,hashed_bytes=hashed_bytes,
            duplicate_paths=sum(len(g['duplicate_paths']) for g in groups),
            potential_allocated_saving_bytes=proposal['potential_allocated_saving_bytes'],
            input_mutations_performed=False,approval_granted=False,wall_s=time.perf_counter()-started)
        write_json(OUTPUT/'result.json',report)
        print('TRAINING_DUPLICATE_INVENTORY_COMPLETE',digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k not in ('source_sha256','artifact_sha256','input_roots')},flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_TRAINING_DUPLICATE_INVENTORY_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
