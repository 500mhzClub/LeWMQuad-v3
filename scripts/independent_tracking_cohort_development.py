"""Bounded eight-trial receipts and sensor-only replay phase; no native launcher.

The future frozen launcher owns source/runtime/resource admission. This module
does not start Genesis or training. Native arrays are never parsed here; their
artifact bytes are authenticated before/after sensor replay.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
import shutil
import time
import numpy as np

from lewm.independent_tracking_challenge_development import (
    TRIALS, MAX_TICKS, MAX_FRAMES, MAX_PHYSICS_SAMPLES, specification)
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.temporal_anchor_continuity_development import TemporalAnchorVisualLedMotion
from lewm.joint_rgbd_rigid_pose_development import proper
from scripts.independent_tracking_artifacts_development import STATIC, frame_names, serial, EPISODE_BYTES, RESERVE_BYTES
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.replay_go2_temporal_anchor_continuity_v1 import ContinuityAudit
from scripts.independent_tracking_native_contact_guard_development import validate_report as validate_native_contact_report

RAW_BYTES = len(TRIALS) * EPISODE_BYTES
TOTAL_BYTES = RAW_BYTES + 4 * 1024**3  # Base-only replay/evaluation/metadata headroom.
MAX_METADATA = 32 * 1024**2
MAX_ROW = 128 * 1024
ARMS = ('original', 'temporal_anchor')
MODELS = dict(original=MultiReferenceVisualLedMotion, temporal_anchor=TemporalAnchorVisualLedMotion)


def require(value, message):
    if not value: raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=serial)+'\n').encode()


def read(output, name):
    p = artifact_path(output, name)
    require(p.stat().st_size <= MAX_METADATA, 'bounded cohort metadata required')
    return json.loads(p.read_text())


def validate_arm_row(row, frame, now):
    require(set(row)=={'pose','selection','failure','continuity','observer_wall_ms'}, 'exact paired arm fields required')
    ms=row['observer_wall_ms']
    require(type(ms) in (int,float) and np.isfinite(ms) and ms>=0, 'finite observer-only timing required')
    p=row['pose']
    if p is not None:
        require(p['frame']==frame and p['measured_ns']==now and p['mode']=='gyro'
                and p['position_error_bound'] is None and p['global_history_reset'] is False,
                'same-frame gyro-conditioned pose without invented bounds/reset required')
        xyz=np.asarray(p['position_initial_body_m'],float)
        require(xyz.shape==(3,) and np.isfinite(xyz).all(), 'finite measured pose required')
        proper(p['rotation_initial_body_from_current_body'])


def expected_episode_names(result):
    frames = result['rgbd_frames']
    require(type(frames) is int and 0 <= frames <= MAX_FRAMES, 'bounded actual frame count required')
    excluded = {'failure.json', 'persistence.json', 'static_objects.json',
                'startup_native_robot_geometry.json', 'setup_checks.json'}
    names = set(STATIC) - excluded
    if result['setup_checked']:
        names |= {'static_objects.json', 'startup_native_robot_geometry.json', 'setup_checks.json'}
    names |= {name for i in range(frames) for name in frame_names(i)}
    return names


def verify_episode(output, trial, result, receipt):
    require(trial in TRIALS and result['trial'] == trial, 'exact challenge trial identity required')
    require(result['status'] == 'TRACKING_TAPE_REQUIRES_RAW_AUDIT' and result['initialized'] is True
            and result['infrastructure_failure'] is None and result['secondary_failures'] == [],
            'infrastructure-failed or uninitialized trial cannot enter complete collection')
    for key, limit in (('command_ticks',MAX_TICKS), ('completed_ticks',MAX_TICKS),
                       ('decisions',MAX_FRAMES), ('physics_samples',MAX_PHYSICS_SAMPLES)):
        require(type(result[key]) is int and 0 <= result[key] <= limit, 'bounded trial count required: '+key)
    for key in ('setup_checked','setup_admitted','schedule_complete'):
        require(type(result[key]) is bool, 'explicit trial boolean required')
    require(result['completed_ticks'] <= result['command_ticks'] <= result['completed_ticks']+1,
            'at most one interrupted command required')
    require(result['decisions'] == result['command_ticks']+int(result['schedule_complete'])
            and result['rgbd_frames'] in (result['decisions'],result['decisions']+1),
            'exact capture/decision/dispatch population required')
    for key in ('navigation_qualified','real_time_qualified','observer_executed','native_evaluation_executed',
                'sensor_reconstruction_verified','tracker_required_for_commands','native_state_used_for_commands'):
        require(result[key] is False, 'collection cannot grant qualification or observer/native command input')
    require(receipt['failed_internal'] is False and type(receipt['failed_external']) is bool,
            'failed internal persistence cannot enter a complete collection')
    if receipt['failed_external']:
        require(result['physical_stop'] == 'FRESH_MISSION_INITIAL_SETUP_REJECTED'
                and result['setup_checked'] and not result['setup_admitted'],
                'only complete setup-rejection evidence permits a stopped external operation')
    for key in ('physical_stop','acquisition_stop'):
        require(result[key] is None or type(result[key]) is str and bool(result[key]), 'explicit nullable stop reason')
    if result['schedule_complete']:
        require(result['setup_admitted'] and result['completed_ticks'] == MAX_TICKS
                and result['physical_stop'] is None and result['acquisition_stop'] is None,
                'complete tape contradicts setup, stops or commands')
    else:
        require(result['physical_stop'] is not None or result['acquisition_stop'] is not None,
                'incomplete tape needs a retained stop reason')
    names = expected_episode_names(result)
    require(set(receipt['artifact_sha256']) == set(receipt['artifact_sizes']) == names,
            'exact complete episode artifact roster required; no orphan or omitted frames')
    bindings = {trial+'/'+name:sha for name,sha in receipt['artifact_sha256'].items()}
    verify_artifacts(output, bindings)
    for name,size in receipt['artifact_sizes'].items():
        require(type(size) is int and size >= 0 and artifact_path(output,trial+'/'+name).stat().st_size == size,
                'actual artifact size differs from receipt')
    require(type(receipt['artifact_bytes']) is int and receipt['artifact_bytes'] == sum(receipt['artifact_sizes'].values())
            and receipt['artifact_bytes'] <= EPISODE_BYTES, 'episode byte accounting differs')
    require(read(output,trial+'/specification.json') == specification(trial), 'exact recorded scene specification required')
    require(read(output,trial+'/result.json') == result, 'receipt/result mismatch')
    validate_native_contact_report(result['native_contact_integrity'],result['physics_samples'])
    # This parser opens sensor metadata only, never camera/native pose audits.
    manifest = read(output,trial+'/policy_observations.json')
    require(len(manifest['frames']) == result['rgbd_frames'], 'manifest/capture count mismatch')
    for i,frame in enumerate(manifest['frames']):
        require(frame == dict(rgb_file=f'rgb_{i:04d}.png', image_ns=1_500_000_000+i*100_000_000,
                             decision_ns=1_500_000_000+i*100_000_000), 'complete fixed sensor clock/prefix required')
    return bindings


class CohortStore:
    def __init__(self, output):
        self.output = validate_root(output)
        self.allowed = {'launch.json','collection_complete.json','sensor_phase_complete.json','failure.json','result.json'}
        self.allowed |= {t+s for t in TRIALS for s in ('_receipt.json','_estimates.jsonl','_evaluation.jsonl','_audit.json')}
        require(not any((self.output/n).exists() or (self.output/n).is_symlink() for n in self.allowed),
                'fresh cohort metadata only; no retry/resume')
        self.hashes = {}; self.sizes = {}; self.episodes = {}; self.failed = False

    @property
    def used(self):
        return sum(self.sizes.values())+sum(r['receipt']['artifact_bytes'] for r in self.episodes.values())

    def path(self, name):
        require(type(name) is str and name in self.allowed, 'explicit cohort metadata path required')
        p = self.output/name
        require(p.resolve() == p and self.output.resolve() == self.output, 'no symlink cohort output')
        return p

    def check(self, size):
        require(type(size) is int and size >= 0 and self.used+size <= TOTAL_BYTES, 'cohort byte budget exceeded')
        require(shutil.disk_usage(self.output).free >= RESERVE_BYTES+size, 'cohort free-space reserve exhausted')

    @contextmanager
    def stream(self, name):
        p=self.path(name)
        require(name not in self.hashes and not p.exists(), 'exclusive new cohort artifact required')
        self.check(0)
        try:
            with p.open('xb') as stream:
                yield stream
                stream.flush();os.fsync(stream.fileno())
        except BaseException:
            self.failed=True
            raise
        finally:
            if p.is_file():
                self.path(name)
                with p.open('rb') as stream: self.hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
                self.sizes[name]=p.stat().st_size
        self.check(0)

    def append(self, stream, row):
        payload=encode(row)
        require(len(payload) <= MAX_ROW, 'bounded complete paired row required')
        # Open-stream bytes are not yet in the saved-file roster.
        self.check(stream.tell()+len(payload))
        if stream.write(payload) != len(payload): raise OSError('short cohort row write')

    def save(self, name, value):
        payload=encode(value)
        require(len(payload) <= MAX_METADATA, 'bounded serialized cohort metadata required')
        self.check(len(payload))
        with self.stream(name) as stream:
            if stream.write(payload) != len(payload): raise OSError('short cohort metadata write')
        return self.hashes[name]

    def admit_episode(self, trial, result, receipt):
        require(not self.failed and len(self.episodes)<len(TRIALS) and trial == TRIALS[len(self.episodes)],
                'ordered complete new trial receipt required')
        verify_episode(self.output,trial,result,receipt)
        require(sum(r['receipt']['artifact_bytes'] for r in self.episodes.values())+receipt['artifact_bytes'] <= RAW_BYTES,
                'raw cohort budget exceeded')
        self.check(receipt['artifact_bytes'])
        self.episodes[trial]=dict(result=deepcopy(result),receipt=deepcopy(receipt))
        self.save(trial+'_receipt.json',self.episodes[trial])

    def complete_collection(self):
        require(not self.failed and tuple(self.episodes)==TRIALS, 'all eight ordered terminal trial receipts required')
        for trial,row in self.episodes.items(): verify_episode(self.output,trial,row['result'],row['receipt'])
        return self.save('collection_complete.json',dict(status='ALL_EIGHT_TRACKING_TAPES_RECORDED_NOT_YET_RAW_AUDITED',
            trials=list(TRIALS), receipts={t:self.hashes[t+'_receipt.json'] for t in TRIALS},
            native_coordinates_parsed=False, sensor_reconstruction_verified=False, navigation_qualified=False))


def verify_collection(output, collection_sha256):
    verify_artifacts(output,{'collection_complete.json':collection_sha256})
    marker=read(output,'collection_complete.json')
    require(marker['status']=='ALL_EIGHT_TRACKING_TAPES_RECORDED_NOT_YET_RAW_AUDITED'
            and marker['trials']==list(TRIALS) and set(marker['receipts'])==set(TRIALS)
            and marker['native_coordinates_parsed'] is False and marker['sensor_reconstruction_verified'] is False
            and marker['navigation_qualified'] is False, 'complete sensor-only collection marker required')
    episodes={}
    for trial in TRIALS:
        name=trial+'_receipt.json';verify_artifacts(output,{name:marker['receipts'][trial]})
        row=read(output,name);verify_episode(output,trial,row['result'],row['receipt']);episodes[trial]=row
    require(sum(r['receipt']['artifact_bytes'] for r in episodes.values())<=RAW_BYTES, 'complete raw-byte budget required')
    return marker,episodes


def _replay_population(store, collection_sha256):
    _,episodes=verify_collection(store.output,collection_sha256)
    require(not store.failed and tuple(store.episodes)==TRIALS and store.episodes==episodes,
            'same admitted complete cohort required')
    reports={}
    for trial in TRIALS:
        count=episodes[trial]['result']['rgbd_frames']
        reader=IntentReturnRGBDReplay(store.output/trial) if count else None
        require(reader is None or len(reader.frames)==count, 'complete declared sensor replay population required')
        models={a:MODELS[a]() for a in ARMS};continuity=ContinuityAudit()
        counts={a:dict(frames=0,available=0,first_failure=None) for a in ARMS}
        categories=dict(both=0,original_only=0,temporal_anchor_only=0,neither=0)
        name=trial+'_estimates.jsonl'
        with store.stream(name) as target:
            for frame in range(count):
                p,d,f,now=reader.packet(frame)
                require(now==1_500_000_000+frame*100_000_000, 'fixed complete sensor clock required')
                arms={}
                for arm,model in models.items():
                    # Give each observer private copies of identical causal input.
                    pp,dd,ff=deepcopy((p,d,f))
                    start=time.perf_counter_ns();r=model.observe(pp,dd,ff,now_ns=now)
                    ms=(time.perf_counter_ns()-start)/1e6
                    row=json.loads(encode(dict(pose=r['current_pose'],selection=r['reference_selection'],
                        failure=r['terminal_failure'],continuity=r.get('continuity_evidence'),observer_wall_ms=ms)))
                    validate_arm_row(row,frame,now)
                    c=counts[arm];c['frames']+=1
                    if row['pose'] is not None:
                        require(c['first_failure'] is None, 'no observer reset after missing current pose')
                        require(row['pose']['frame']==frame and row['pose']['measured_ns']==now, 'current frame pose required')
                        c['available']+=1
                    elif c['first_failure'] is None:c['first_failure']=frame
                    arms[arm]=row
                a,b=(arms[x]['pose'] is not None for x in ARMS)
                category='both' if a and b else 'original_only' if a else 'temporal_anchor_only' if b else 'neither'
                categories[category]+=1
                row=dict(frame=frame,measured_ns=now,arms=arms,availability=category,
                         native_pose_input=False,navigation_qualified=False)
                continuity.observe(row);store.append(target,row)
        reports[trial]=dict(frames=count,arms=counts,availability=categories,continuity=continuity.summary(),
            estimates_file=name,estimates_sha256=store.hashes[name],post_failure_frames_retained=True,
            native_coordinates_parsed=False,independent_observations_verified=False,navigation_qualified=False)
        print('INDEPENDENT_TRACKING_SENSOR_REPLAY',trial,count,{a:c['available'] for a,c in counts.items()},flush=True)
    verify_collection(store.output,collection_sha256)
    verify_artifacts(store.output,{r['estimates_file']:r['estimates_sha256'] for r in reports.values()})
    phase=dict(status='ALL_EIGHT_PAIRED_TRACKING_SENSOR_STREAMS_COMPLETE',collection_sha256=collection_sha256,
        trials=list(TRIALS),reports=reports,native_coordinates_parsed=False,navigation_qualified=False,
        independent_observations_verified=False)
    return store.save('sensor_phase_complete.json',phase),phase


def record_phase_failure(store,stage,error):
    store.failed=True
    if not store.path('failure.json').exists():
        try:
            store.save('failure.json',dict(status='TERMINAL_TRACKING_COHORT_PHASE_FAILURE',stage=stage,
                reason=repr(error),output_sha256=dict(store.hashes),retry_performed=False,
                navigation_qualified=False,goal_achieved=False))
        except Exception:
            # Preserve the original error and any partial failure-file bytes.
            # A storage failure is not permission to overwrite or resume.
            pass


def replay_population(store,collection_sha256):
    require(not store.failed,'failed cohort cannot restart sensor replay')
    try:return _replay_population(store,collection_sha256)
    except BaseException as error:
        record_phase_failure(store,'sensor_replay_or_admission',error)
        raise


def admit_native_evaluation(output, collection_sha256, sensor_phase_sha256):
    """Authenticate the full sensor phase BEFORE a caller can parse native arrays."""
    _,episodes=verify_collection(output,collection_sha256)
    verify_artifacts(output,{'sensor_phase_complete.json':sensor_phase_sha256})
    phase=read(output,'sensor_phase_complete.json')
    require(phase['status']=='ALL_EIGHT_PAIRED_TRACKING_SENSOR_STREAMS_COMPLETE'
            and phase['collection_sha256']==collection_sha256 and phase['trials']==list(TRIALS)
            and set(phase['reports'])==set(TRIALS) and phase['native_coordinates_parsed'] is False
            and phase['navigation_qualified'] is False and phase['independent_observations_verified'] is False,
            'complete bound paired sensor phase required before native evaluation')
    for trial in TRIALS:
        report=phase['reports'][trial]
        require(report['estimates_file']==trial+'_estimates.jsonl'
                and report['frames']==episodes[trial]['result']['rgbd_frames'], 'complete report/frame identity required')
        verify_artifacts(output,{report['estimates_file']:report['estimates_sha256']})
        counts={a:dict(frames=0,available=0,first_failure=None) for a in ARMS}
        categories=dict(both=0,original_only=0,temporal_anchor_only=0,neither=0)
        continuity=ContinuityAudit();n=0
        with artifact_path(output,report['estimates_file']).open('rb') as stream:
            for line in stream:
                require(len(line)<=MAX_ROW, 'bounded paired sensor row required')
                row=json.loads(line)
                require(row['frame']==n and row['measured_ns']==1_500_000_000+n*100_000_000
                        and row['native_pose_input'] is False and row['navigation_qualified'] is False
                        and set(row['arms'])==set(ARMS), 'ordered sensor-only paired row required')
                for arm in ARMS:
                    validate_arm_row(row['arms'][arm],n,row['measured_ns'])
                    c=counts[arm];c['frames']+=1;pose=row['arms'][arm]['pose']
                    if pose is not None:
                        require(c['first_failure'] is None and pose['frame']==n and pose['measured_ns']==row['measured_ns'],
                                'same current frame and no reset required')
                        c['available']+=1
                    elif c['first_failure'] is None:c['first_failure']=n
                a,b=(row['arms'][arm]['pose'] is not None for arm in ARMS)
                category='both' if a and b else 'original_only' if a else 'temporal_anchor_only' if b else 'neither'
                require(row['availability']==category, 'paired availability differs')
                categories[category]+=1;continuity.observe(row);n+=1
        require(n==report['frames'] and counts==report['arms'] and categories==report['availability']
                and continuity.summary()==report['continuity'], 'sensor phase summaries differ from saved complete rows')
    return episodes,phase
