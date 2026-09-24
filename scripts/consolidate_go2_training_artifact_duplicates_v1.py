"""Explicitly approved, journaled hard-link consolidation of frozen duplicates."""
import argparse
import json
import os
from pathlib import Path
import psutil
from scripts.inventory_go2_training_artifact_duplicates_v1 import metadata, INPUTS
from scripts.navigation_artifact_root_development import BASE, artifact_path, verify_artifacts, create_output
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify as verify_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INVENTORY = BASE/'go2_training_artifact_duplicate_inventory_v1_attempt_001'
INVENTORY_SHA = '732b0a95a98d598843e1d56f85612093b3d78e3e1ed145f4e27d24f2ceb6d577'
OUTPUT = BASE/'go2_training_artifact_hardlink_consolidation_v1_attempt_001'
PROTOCOL = 'docs/go2_training_artifact_hardlink_consolidation_v1_2026-09-09.md'
SOURCES = ('scripts/consolidate_go2_training_artifact_duplicates_v1.py',
    'lewm/tests/test_training_artifact_hardlink_consolidation_development.py', PROTOCOL)


def require_quiet_consumers():
    roots = tuple(str(BASE/name)+'/' for name in INPUTS)
    blockers=[];inaccessible=[];checked=0
    for process in psutil.process_iter():
        if process.pid==os.getpid(): continue
        argv=None
        try:
            if process.uids().real!=os.getuid(): continue
            argv=process.cmdline();cmd=' '.join(argv)
            checked+=1
            if (argv and 'genesis_rocm_0_4_6_v1/bin/python' in argv[0]):
                blockers.append(dict(pid=process.pid,reason='other development Python worker'))
            files=[str(item.path) for item in process.open_files()]
            files += [str(item.path) for item in process.memory_maps(grouped=False)]
            if any(path.startswith(roots) for path in files):
                blockers.append(dict(pid=process.pid,reason='open training artifact'))
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            if argv is None:
                raise ValueError('could not identify process ownership and command before consolidation') from error
            inaccessible.append(dict(pid=process.pid,executable=argv[0] if argv else None))
    if blockers: raise ValueError('active consumers must finish: '+json.dumps(blockers))
    return dict(checked_owned_processes=checked,inaccessible_owned_descriptors=inaccessible,
        known_development_workers_absent=True,accessible_training_descriptors_absent=True,
        global_process_quiescence_proven=False)


def resolve(row):
    if row['root'] not in INPUTS: raise ValueError('exact reviewed training root required')
    return artifact_path(BASE/row['root'],row['path'])


def check_row(row, *, expected_sha, expected_metadata=None, resolver=resolve):
    path=resolver(row); expected=row['metadata'] if expected_metadata is None else expected_metadata
    if metadata(path)!=expected: raise ValueError('reviewed file metadata changed')
    if digest(path)!=expected_sha: raise ValueError('reviewed file bytes changed')
    if metadata(path)!=expected: raise ValueError('file changed during digest')
    return path


def consolidate_group(group, journal, *, sequence, resolver=resolve):
    canonical=group['canonical']; sha=group['sha256']
    source=check_row(canonical,expected_sha=sha,resolver=resolver)
    expected_source=metadata(source)
    for index,row in enumerate(group['duplicate_paths']):
        source=check_row(canonical,expected_sha=sha,expected_metadata=expected_source,resolver=resolver)
        target=check_row(row,expected_sha=sha,resolver=resolver)
        target_before=metadata(target)
        if (source==target or source.stat().st_dev!=target.stat().st_dev
                or target_before['nlink']!=1
                or any(target_before[k]!=expected_source[k] for k in ('mode','uid','gid','size'))):
            raise ValueError('distinct identical single-link duplicate on same volume required')
        temporary=target.with_name('.lewm-consolidate-v1-'+str(sequence)+'-'+str(index))
        if temporary.exists() or temporary.is_symlink(): raise ValueError('exclusive temporary link required')
        journal(dict(stage='intent',group=sequence,index=index,canonical=canonical['path'],
            canonical_root=canonical['root'],target=row['path'],target_root=row['root'],
            sha256=sha,target_before=target_before,source_before=expected_source,
            temporary_name=temporary.name))
        os.link(source,temporary,follow_symlinks=False)
        # A failure retains the temporary link and journal for exact recovery.
        linked=metadata(temporary)
        if (linked['dev']!=expected_source['dev'] or linked['ino']!=expected_source['ino']
                or linked['nlink']!=expected_source['nlink']+1
                or any(linked[k]!=expected_source[k] for k in ('mode','uid','gid','size','mtime_ns','blocks'))
                or digest(temporary)!=sha or metadata(temporary)!=linked
                or metadata(source)!=linked or metadata(target)!=target_before):
            raise ValueError('source or target changed before atomic replacement')
        os.replace(temporary,target)
        expected_source=metadata(source)
        if (metadata(target)!=expected_source or expected_source['ino']!=linked['ino']
                or expected_source['nlink']!=linked['nlink'] or digest(target)!=sha):
            raise ValueError('hard-link replacement failed identity or byte verification')
        parent_fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
        journal(dict(stage='complete',group=sequence,index=index,target=row['path'],
            target_root=row['root'],sha256=sha,source_after=expected_source,
            original_path_retained=True,original_contents_retained=True))


def admit():
    verify_artifacts(INVENTORY,{'result.json':INVENTORY_SHA})
    result=read_json(INVENTORY,'result.json');verify_sources(result['source_sha256'])
    verify_artifacts(INVENTORY,result['artifact_sha256'])
    proposal=read_json(INVENTORY,'proposal.json')
    if (result['status']!='TRAINING_ARTIFACT_DUPLICATE_INVENTORY_V1_COMPLETE'
            or proposal['input_result_sha256']!=INPUTS or proposal['input_mutations_performed'] is not False
            or len(proposal['groups'])!=5523 or result['duplicate_paths']!=27210
            or proposal['potential_allocated_saving_bytes']!=7219273728):
        raise ValueError('exact completed byte-verified proposal required')
    for name,sha in INPUTS.items():verify_artifacts(BASE/name,{'result.json':sha})
    before=require_quiet_consumers()
    count=0
    for group in proposal['groups']:
        for row in [group['canonical'],*group['duplicate_paths']]:
            check_row(row,expected_sha=group['sha256']);count+=1
    if count!=32733: raise ValueError('complete reviewed candidate population required')
    after=require_quiet_consumers()
    return result,proposal,dict(before=before,after=after)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--approval-json')
    args=parser.parse_args()
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive consolidation; no automatic retry')
    sources={p:digest(Path(p)) for p in SOURCES};verify_sources(sources)
    if not args.preflight_only:
        if not args.approval_json:raise ValueError('explicit reviewed user authorization required')
        approval_path=Path(args.approval_json)
        if (approval_path.parent!=Path('docs') or approval_path.name!='go2_training_artifact_hardlink_authorization_2026-09-09.json'
                or approval_path.is_symlink()):raise ValueError('exact ordinary authorization path required')
        approval=json.loads(approval_path.read_text())
        if (approval.get('approved') is not True or approval.get('inventory_result_sha256')!=INVENTORY_SHA
                or approval.get('source_sha256')!=sources or approval.get('output_root')!=str(OUTPUT)
                or not isinstance(approval.get('user_instruction'),str) or not approval['user_instruction'].strip()):
            raise ValueError('explicit instruction bound to exact reviewed operation required')
    result,proposal,process_checks=admit();resources=hardware()
    if resources['artifact_free_bytes']<40*1024**3+128*1024**2:
        raise ValueError('retain 40 GiB reserve and bounded transaction evidence allowance')
    if args.preflight_only:
        print('TRAINING_CONSOLIDATION_PREFLIGHT_PASS',dict(groups=len(proposal['groups']),
            duplicate_paths=result['duplicate_paths'],potential_allocated_saving_bytes=7219273728,
            proposal_sha256=result['artifact_sha256']['proposal.json'],input_mutations_performed=False,
            hardware=resources,process_checks=process_checks),flush=True)
        return
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,inventory_result_sha256=INVENTORY_SHA,
        proposal_sha256=result['artifact_sha256']['proposal.json'],approval=approval,
        approval_file_sha256=digest(approval_path),hardware=resources,process_checks=process_checks))
    try:
        with (OUTPUT/'transactions.jsonl').open('x') as stream:
            def journal(row):
                stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
            for i,group in enumerate(proposal['groups']):
                consolidate_group(group,journal,sequence=i)
                if i%256==0:print('TRAINING_CONSOLIDATION_GROUP',i,flush=True)
        for name,sha in INPUTS.items():
            root=BASE/name;verify_artifacts(root,{'result.json':sha})
            original=read_json(root,'result.json');verify_artifacts(root,original['artifact_sha256'])
            verify_sources(original['source_sha256'])
        verify_sources(sources)
        after=hardware()
        write_json(OUTPUT/'result.json',dict(status='TRAINING_ARTIFACT_HARDLINK_CONSOLIDATION_V1_COMPLETE',
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','transactions.jsonl')},
            groups=len(proposal['groups']),replaced_duplicate_paths=27210,all_original_artifact_hashes_verified=True,
            all_original_paths_retained=True,all_original_contents_retained=True,hardware_after=after,
            observed_free_space_change_bytes=after['artifact_free_bytes']-resources['artifact_free_bytes']))
        print('TRAINING_CONSOLIDATION_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_TRAINING_CONSOLIDATION_FAILURE',reason=repr(error),
            automatic_retry_permitted=False,partial_journal_and_temporary_links_retained=True))
        raise


if __name__=='__main__':main()
