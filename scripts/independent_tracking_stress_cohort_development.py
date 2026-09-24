"""Complete fixed stress replay and post-phase scoring; no native launcher.

All eight base streams and all88 trial/scenario streams are authenticated and
reconstructed before the first native-coordinate auditor can run. Reference
injections remain mechanism tests, not physical occlusion or independent trials.
"""
from copy import deepcopy
import json
import shutil

from lewm.independent_tracking_stress_development import (
    SCENARIOS, ARMS, ONSET_FRAME, INTERVAL_NS, PairedStressObserver,
    identity as stress_identity, packet_digest, transform)
from lewm.independent_tracking_challenge_development import TRIALS, MAX_FRAMES
from scripts import independent_tracking_cohort_development as base
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts

TOTAL_BYTES = base.RAW_BYTES + 12 * 1024**3
PHASE = 'stress_sensor_phase_complete.json'
ARM_FIELDS = {'pose','selection','failure','continuity','observer_wall_ms'}
ROW_FIELDS = {'frame','measured_ns','scenario','definition_sha256','source_packet_sha256',
              'intervened_packet_sha256','intervention','arms','native_pose_input',
              'navigation_qualified','physical_occlusion_simulated','uncertainty_calibrated','availability'}


def stream_name(trial, scenario, evaluation=False):
    base.require(trial in TRIALS and scenario in SCENARIOS, 'fixed trial/scenario required')
    return trial+'__'+scenario+('_stress_evaluation.jsonl' if evaluation else '_stress_estimates.jsonl')


def resource_contract():
    # Base has21 metadata files; the stress phase adds one. All other outputs
    # are bounded row streams. Partial failed streams are retained, never retried.
    streams = len(TRIALS)*2*(1+len(SCENARIOS))
    metadata = 6+2*len(TRIALS)
    bound = base.RAW_BYTES + streams*MAX_FRAMES*base.MAX_ROW + metadata*base.MAX_METADATA
    base.require(bound <= TOTAL_BYTES, 'schema-derived raw/stream/metadata envelope exceeds total budget')
    return dict(raw_bytes=base.RAW_BYTES, total_bytes=TOTAL_BYTES,
        reserve_bytes=base.RESERVE_BYTES, row_bytes=base.MAX_ROW, metadata_file_bytes=base.MAX_METADATA,
        maximum_frames_per_stream=MAX_FRAMES, streams=streams, metadata_files=metadata,
        worst_case_bound_bytes=bound, os_quota_enforced=False, memory_bound_proved=False)


class StressCohortStore(base.CohortStore):
    def __init__(self, output):
        super().__init__(output)
        extra = {PHASE} | {stream_name(t,s,e) for t in TRIALS for s in SCENARIOS for e in (False,True)}
        base.require(not any((self.output/n).exists() or (self.output/n).is_symlink() for n in extra),
                     'fresh stress metadata only; no retry/resume')
        self.allowed |= extra
        contract = resource_contract()
        base.require(len(self.allowed) == contract['streams']+contract['metadata_files'],
                     'resource contract must cover the exact output roster')

    def check(self, size):
        base.require(type(size) is int and size >= 0 and self.used+size <= TOTAL_BYTES,
                     'combined stress cohort byte budget exceeded')
        base.require(shutil.disk_usage(self.output).free >= base.RESERVE_BYTES+size,
                     'combined stress free-space reserve exhausted')


def rows(output, name):
    """Bound each read before allocation and reject hidden/trailing populations."""
    with artifact_path(output,name).open('rb') as stream:
        count = 0
        while line := stream.readline(base.MAX_ROW+1):
            base.require(len(line) <= base.MAX_ROW and count < MAX_FRAMES, 'bounded complete stream required')
            yield json.loads(line)
            count += 1


def category(row):
    a,b = (row['arms'][k]['pose'] is not None for k in ARMS)
    return 'both' if a and b else 'original_only' if a else 'temporal_anchor_only' if b else 'neither'


class StressAudit:
    """Reconstruct outcome/exposure denominators; no success from unexercised faults."""
    def __init__(self, scenario):
        base.require(scenario in SCENARIOS, 'fixed stress scenario required')
        self.scenario = scenario; self.frames = 0; self.continuity = base.ContinuityAudit()
        self.counts = {a:dict(frames=0,available=0,first_failure=None) for a in ARMS}
        self.failures = {a:None for a in ARMS}
        self.availability = dict(both=0,original_only=0,temporal_anchor_only=0,neither=0)
        self.exposure = {a:dict(update_attempts=0,updates_with_changed_packet=0,
            retained_reference_denials=0,increment_corruptions=0,frames_with_reference_injection=0,
            onset_update_attempted=False) for a in ARMS}
        self.changed = 0

    def observe(self, row, packet, nominal=None):
        frame=self.frames; now=1_500_000_000+frame*INTERVAL_NS
        base.require(set(row)==ROW_FIELDS and type(row['frame']) is int and type(row['measured_ns']) is int
            and frame < MAX_FRAMES and row['frame']==frame and row['measured_ns']==now
            and row['scenario']==self.scenario and row['definition_sha256']==stress_identity()
            and set(row['arms'])==set(ARMS), 'exact sequential fixed stress row required')
        for key in ('native_pose_input','navigation_qualified','physical_occlusion_simulated','uncertainty_calibrated'):
            base.require(row[key] is False, 'stress cannot grant physical or calibrated qualification')
        modified,intervention=transform(packet,self.scenario,frame,1_500_000_000)
        base.require(row['source_packet_sha256']==packet_digest(packet)
            and row['intervened_packet_sha256']==packet_digest(modified)
            and row['intervention']==intervention, 'recorded intervention differs from bound source sensor bytes')
        changed = row['source_packet_sha256'] != row['intervened_packet_sha256']
        self.changed += int(changed)
        for arm in ARMS:
            r=row['arms'][arm];c=self.counts[arm];e=self.exposure[arm]
            base.require(set(r)==ARM_FIELDS|{'observer_update_attempted','reference_injections'}, 'exact stress arm fields')
            base.validate_arm_row({k:r[k] for k in ARM_FIELDS},frame,now)
            attempted=c['first_failure'] is None
            base.require(type(r['observer_update_attempted']) is bool and r['observer_update_attempted']==attempted,
                         'update attempt after terminal failure or hidden skipped update')
            e['update_attempts']+=int(attempted);e['updates_with_changed_packet']+=int(attempted and changed)
            if frame==ONSET_FRAME:e['onset_update_attempted']=attempted
            if r['pose'] is not None:
                base.require(attempted and r['failure'] is None, 'no reset or pose alongside failure')
                c['available']+=1
            else:
                base.require(r['failure'] is not None, 'missing pose requires retained terminal reason')
                if attempted:c['first_failure']=frame;self.failures[arm]=deepcopy(r['failure'])
                else:base.require(r['failure']==self.failures[arm], 'terminal reason rewritten')
            c['frames']+=1
            events=r['reference_injections']
            base.require(type(events) is list and len(events)<=8 and (attempted or not events),
                         'bounded actual reference calls required')
            e['frames_with_reference_injection']+=int(bool(events))
            references=[]
            for event in events:
                base.require(set(event)=={'kind','reference_frame'} and type(event['reference_frame']) is int
                    and 0<=event['reference_frame']<frame, 'actual prior reference identity required')
                if event['kind']=='retained_reference_denied':
                    base.require(self.scenario.startswith('anchor_absence_')
                        and ONSET_FRAME<=frame<ONSET_FRAME+int(self.scenario.rsplit('_',1)[1]),
                        'reference denial outside fixed intervention')
                    e['retained_reference_denials']+=1
                else:
                    base.require(event['kind']=='qualified_increment_position_corrupted'
                        and self.scenario=='anchor_increment_conflict' and arm=='temporal_anchor'
                        and frame==ONSET_FRAME and event['reference_frame']==frame-1
                        and r['pose'] is None and r['continuity']['status']=='ANCHOR_INCREMENT_CONFLICT',
                        'qualified injected contradiction must be terminal')
                    e['increment_corruptions']+=1
                references.append(event['reference_frame'])
            base.require(len(references)==len(set(references)), 'duplicate reference injection event')
            if self.scenario in ('rgb_unavailable_1','depth_unavailable_1','gyro_unavailable_1') and frame==ONSET_FRAME:
                base.require(r['pose'] is None, 'missing current measurement cannot supply a current pose')
            if nominal is not None:
                base.require(all(r[k]==nominal['arms'][arm][k] for k in ARM_FIELDS-{'observer_wall_ms'}),
                             'nominal stress wrapper changed original base observer outputs')
        expected=category(row)
        base.require(row['availability']==expected, 'stress paired availability mismatch')
        self.availability[expected]+=1;self.continuity.observe(row);self.frames+=1

    def summary(self):
        return dict(frames=self.frames,arms=deepcopy(self.counts),availability=deepcopy(self.availability),
            continuity=self.continuity.summary(),exposure=deepcopy(self.exposure),
            changed_packet_frames=self.changed,onset_recorded=self.frames>ONSET_FRAME,
            complete_requested_tape=self.frames==MAX_FRAMES,
            independent_observations=False,noise_distribution_calibrated=False,navigation_qualified=False)


def _same_store(store,episodes):
    base.require(isinstance(store,StressCohortStore) and not store.failed
        and tuple(store.episodes)==TRIALS and store.episodes==episodes, 'same complete fresh stress cohort required')


def _report(store,trial,scenario,reader,count,nominal_report):
    model=PairedStressObserver(scenario);audit=StressAudit(scenario)
    nominal=iter(rows(store.output,nominal_report['estimates_file'])) if scenario=='nominal' else None
    name=stream_name(trial,scenario)
    with store.stream(name) as stream:
        for frame in range(count):
            packet=reader.packet(frame)
            row=json.loads(base.encode(model.observe(packet)));row['availability']=category(row)
            audit.observe(row,packet,next(nominal) if nominal is not None else None)
            store.append(stream,row)
    if nominal is not None:base.require(next(nominal,None) is None, 'extra nominal base rows')
    return audit.summary() | dict(estimates_file=name,estimates_sha256=store.hashes[name])


def _replay(store,collection_sha256,base_phase_sha256):
    # Despite its historical name, this admission only parses sensor metadata.
    episodes,phase=base.admit_native_evaluation(store.output,collection_sha256,base_phase_sha256)
    _same_store(store,episodes);reports={}
    for trial in TRIALS:
        count=episodes[trial]['result']['rgbd_frames']
        reader=base.IntentReturnRGBDReplay(store.output/trial) if count else None
        base.require(reader is None or len(reader.frames)==count,'complete stress sensor population required')
        reports[trial]={s:_report(store,trial,s,reader,count,phase['reports'][trial]) for s in SCENARIOS}
        print('INDEPENDENT_TRACKING_STRESS_REPLAY',trial,count,len(SCENARIOS),flush=True)
    base.admit_native_evaluation(store.output,collection_sha256,base_phase_sha256)
    verify_artifacts(store.output,{r['estimates_file']:r['estimates_sha256']
        for trial in reports.values() for r in trial.values()})
    result=dict(status='ALL_EIGHT_BASE_AND_88_STRESS_SENSOR_STREAMS_COMPLETE',
        collection_sha256=collection_sha256,base_phase_sha256=base_phase_sha256,
        definition_sha256=stress_identity(),resource_contract=resource_contract(),
        trials=list(TRIALS),scenarios=list(SCENARIOS),reports=reports,
        native_coordinates_parsed=False,independent_observations_verified=False,navigation_qualified=False)
    return store.save(PHASE,result),result


def replay_stress_population(store,collection_sha256,base_phase_sha256):
    base.require(not store.failed,'failed cohort cannot restart stress replay')
    try:return _replay(store,collection_sha256,base_phase_sha256)
    except BaseException as error:
        base.record_phase_failure(store,'stress_replay_or_admission',error)
        raise


def admit_complete_sensor_phase(output,collection_sha256,base_phase_sha256,stress_phase_sha256):
    """No native arrays: reconstruct all transforms, counts and continuity first."""
    episodes,phase=base.admit_native_evaluation(output,collection_sha256,base_phase_sha256)
    verify_artifacts(output,{PHASE:stress_phase_sha256});stress=base.read(output,PHASE)
    base.require(stress['status']=='ALL_EIGHT_BASE_AND_88_STRESS_SENSOR_STREAMS_COMPLETE'
        and stress['collection_sha256']==collection_sha256 and stress['base_phase_sha256']==base_phase_sha256
        and stress['definition_sha256']==stress_identity() and stress['resource_contract']==resource_contract()
        and stress['trials']==list(TRIALS) and stress['scenarios']==list(SCENARIOS)
        and set(stress['reports'])==set(TRIALS), 'complete exact bound stress population required')
    for key in ('native_coordinates_parsed','independent_observations_verified','navigation_qualified'):
        base.require(stress[key] is False, 'sensor-only unqualified stress marker required')
    for trial in TRIALS:
        count=episodes[trial]['result']['rgbd_frames'];reports=stress['reports'][trial]
        base.require(set(reports)==set(SCENARIOS), 'all eleven fixed scenarios required')
        reader=base.IntentReturnRGBDReplay(output/trial) if count else None
        base.require(reader is None or len(reader.frames)==count, 'complete source sensor population')
        for scenario in SCENARIOS:
            report=reports[scenario];name=stream_name(trial,scenario)
            base.require(report['estimates_file']==name and report['frames']==count, 'exact stress report identity/count')
            verify_artifacts(output,{name:report['estimates_sha256']});audit=StressAudit(scenario)
            nominal=iter(rows(output,phase['reports'][trial]['estimates_file'])) if scenario=='nominal' else None
            for frame,row in enumerate(rows(output,name)):
                base.require(frame<count, 'extra stress frames')
                audit.observe(row,reader.packet(frame),next(nominal) if nominal is not None else None)
            if nominal is not None:base.require(next(nominal,None) is None, 'missing nominal stress frames')
            base.require(report==audit.summary()|dict(estimates_file=name,estimates_sha256=report['estimates_sha256']),
                         'stress summaries differ from complete reconstructed sensor rows')
    # Catch source/phase edits during reconstruction before native admission.
    base.admit_native_evaluation(output,collection_sha256,base_phase_sha256)
    verify_artifacts(output,{PHASE:stress_phase_sha256} | {r['estimates_file']:r['estimates_sha256']
        for trial in stress['reports'].values() for r in trial.values()})
    return episodes,phase,stress


class _ScoreDestination:
    """Narrow adapter for the unchanged base pose scorer, not a second writer."""
    def __init__(self,store,trial,scenario):
        self.store=store;self.output=store.output;self.trial=trial;self.scenario=scenario
    def stream(self,name):
        base.require(name==self.trial+'_evaluation.jsonl', 'exact scoring destination required')
        return self.store.stream(stream_name(self.trial,self.scenario,True))
    def append(self,stream,row):return self.store.append(stream,row)


def _evaluate(store,collection_sha256,base_phase_sha256,stress_phase_sha256,protocol_sha256):
    episodes,phase,stress=admit_complete_sensor_phase(
        store.output,collection_sha256,base_phase_sha256,stress_phase_sha256)
    _same_store(store,episodes)
    base.require(type(protocol_sha256) is str and len(protocol_sha256)==64
        and all(c in '0123456789abcdef' for c in protocol_sha256), 'exact setup protocol SHA required')
    # This import/callback boundary occurs only after every sensor stream passes.
    from scripts import independent_tracking_evaluation_development as scoring
    scores={};stress_scores={};audits={}
    for trial in TRIALS:
        raw,audit=scoring._raw_audit(store.output,trial,episodes[trial]['result'],protocol_sha256)
        score=scoring._score_trial(store,trial,phase['reports'][trial],raw)
        treated={s:scoring._score_trial(_ScoreDestination(store,trial,s),trial,stress['reports'][trial][s],raw)
                 for s in SCENARIOS}
        audits[trial]=audit;scores[trial]=score;stress_scores[trial]=treated
        store.save(trial+'_audit.json',dict(raw_audit=audit,base_pose_score=score,stress_pose_scores=treated))
    admit_complete_sensor_phase(store.output,collection_sha256,base_phase_sha256,stress_phase_sha256)
    result=dict(status='EIGHT_TRIAL_BASE_AND_FIXED_STRESS_RAW_AUDIT_AND_SCORING_COMPLETE',
        collection_sha256=collection_sha256,base_phase_sha256=base_phase_sha256,
        stress_phase_sha256=stress_phase_sha256,protocol_sha256=protocol_sha256,
        trials=list(TRIALS),scenarios=list(SCENARIOS),resource_contract=resource_contract(),
        scores=scores,stress_scores=stress_scores,output_sha256=dict(store.hashes),
        all_intended_motion_covered=all(a['coverage']['intended_motion_covered'] for a in audits.values()),
        all_candidate_base_pose_allocations_met=all(
            s['empirical_local_pose_allocation_met']['temporal_anchor'] for s in scores.values()),
        strict_depth_visibility_pass=all(a['sensors']['depth_checks'] and all(
            x['within1mm'] and x['physical_visibility']['passes_sampled_physical_visibility']
            for x in a['sensors']['depth_checks']) for a in audits.values()),
        stress_arms_evaluated=True,stress_mechanism_validation_is_hardware_qualification=False,
        predecessor_prefix_comparison_performed=False,independent_observations_verified=False,
        full_challenge_pass=False,navigation_qualified=False,real_time_qualified=False,goal_achieved=False)
    store.save('result.json',result)
    return result


def evaluate_complete_population(store,collection_sha256,base_phase_sha256,stress_phase_sha256,protocol_sha256):
    base.require(not store.failed,'failed cohort cannot restart complete evaluation')
    try:return _evaluate(store,collection_sha256,base_phase_sha256,stress_phase_sha256,protocol_sha256)
    except BaseException as error:
        base.record_phase_failure(store,'complete_stress_native_audit_or_admission',error)
        raise
