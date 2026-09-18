"""Describe actual requests, sensor arrivals and remaining host deadline misses."""
import argparse
from collections import Counter
import json
import numpy as np
from scripts.compare_continuous_navigation_arms_development import path, read


def distribution(values):
    a=np.asarray(values,dtype=float)/1e6
    if not len(a):return None
    return dict(samples=len(a),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),maximum_ms=float(a.max()))


def summarize(root):
    request=read(root,'requests.json'); acquisition=read(root,'acquisitions.json')
    service=read(root,'physical_command_service_receipts.json')
    all_receipts=read(root,'asynchronous_camera_receipts.json')
    receipt=[r for r in all_receipts if 'controller_submitted_wall_ns' in r]
    assert len(request)==len(service)
    missing_request_timestamp=[]
    for index,(a,b) in enumerate(zip(request,service)):
        assert a['simulator_ns']==b['simulator_ns'] and a['requested_command']==b['serviced_request']
        if 'now_ns' in a:
            assert a['now_ns']==b['host_request_ns']
        else:
            # The terminal pipeline-failure gate returns only a zero command.
            # The separate ledger still records the exact incoming host time.
            assert index==len(request)-1 and a['reason']=='PIPELINE_FAILURE'
            assert not any(a['requested_command']) and not any(b['proposed_command'])
            missing_request_timestamp.append(index)
    request=[a|dict(now_ns=b['host_request_ns']) for a,b in zip(request,service)]
    assert len(acquisition)==len(receipt)
    assert all(a['frame']==b['frame'] and a['measured_ns']==b['measured_ns']
        and b['started_wall_ns']<=b['snapshot_submitted_wall_ns']<=b['renderer_completed_wall_ns']
        <=b['received_wall_ns']<=b['controller_submitted_wall_ns'] for a,b in zip(acquisition,receipt))
    by={a['measured_ns']:a for a in acquisition}
    late=[r for r in request if r['reason']=='FIRST_DISPATCH_TOO_LATE'
        or r.get('underlying_reason')=='FIRST_DISPATCH_TOO_LATE']
    selected=[p for p in read(root,'planning.json') if 'selection' in p]
    gc_intervals=[]; started={}
    if (root/'gc_timing_events.json').exists():
        for event in read(root,'gc_timing_events.json'):
            if event['phase']=='start':started[event['generation']]=event['wall_ns']
            elif event['generation'] in started:
                start=started.pop(event['generation'])
                gc_intervals.append(dict(generation=event['generation'],started_wall_ns=start,
                    completed_wall_ns=event['wall_ns'],elapsed_ms=(event['wall_ns']-start)/1e6))
    long_intervals=[]
    for row in request:
        if 'physical_service_completed_wall_ns' not in row:continue
        begin,end=row['request_started_wall_ns'],row['physical_service_completed_wall_ns']
        if end-begin<20_000_000:continue
        long_intervals.append(dict(simulator_ns=row['simulator_ns'],total_ms=(end-begin)/1e6,
            request_ms=(row['request_finished_wall_ns']-begin)/1e6,
            physics_service_ms=(end-row['physical_service_started_wall_ns'])/1e6,
            overlapping_gc=[g for g in gc_intervals if g['started_wall_ns']<end and g['completed_wall_ns']>begin]))
    return dict(root_name=root.name,request_count=len(request),completed_camera_acquisitions=len(acquisition),
        camera_results_retained_after_shutdown=len(all_receipts)-len(receipt),
        exact_physical_service_request_history=True,actual_acquisition_times_ordered=True,
        host_request_timestamp_source='physical_service_ledger',
        request_rows_missing_duplicate_host_timestamp=missing_request_timestamp,
        nonzero_requested_intervals=sum(any(r['requested_command']) for r in request),
        nonzero_applied_intervals=sum(any(r['applied_command']) for r in request),
        request_reasons=dict(Counter(r['reason'] for r in request)),
        initial_dispatch_late=len(late),late_initial_dispatches_on_camera_boundaries=sum(r['simulator_ns'] in by for r in late),
        initial_dispatch_lateness=distribution([r['now_ns']-r['simulator_ns'] for r in late]),
        snapshot_submission=distribution([a['snapshot_submission_wall_ns'] for a in acquisition]),
        snapshot_before_late_initial_dispatch=distribution([by[r['simulator_ns']]['snapshot_submission_wall_ns']
            for r in late if r['simulator_ns'] in by]),
        pre_snapshot_lag_for_late_dispatch=distribution([by[r['simulator_ns']]['acquisition_started_ns']-r['simulator_ns']
            for r in late if r['simulator_ns'] in by]),
        rendering=distribution([a['rendering_wall_ns'] for a in acquisition]),
        acquisition_to_owner_receipt=distribution([a['measured_acquisition_wall_ns'] for a in acquisition]),
        host_minus_simulator_lag=distribution([r['simulator_lag_ns'] for r in request]),
        request_through_physical_completion=distribution([r['completed_wall_ns']-r['now_ns'] for r in request]),
        selected_plans=len(selected),on_time_plans=sum(p['on_time'] for p in selected),
        independent_physical_evaluation=read(root,'continuous_native_arrival_evaluation.json'),
        result=read(root,'result.json') if (root/'result.json').exists() else None,
        failure=read(root,'failure.json') if (root/'failure.json').exists() else None,
        garbage_collection_intervals=gc_intervals,host_intervals_over_20ms=long_intervals,
        real_time_qualified=False,hardware_validated=False,
        actual_actuator_application_timestamp_measured=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    root=path(parser.parse_args().root_name); result=summarize(root)
    with (root/'async_host_deadline_diagnostic_v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_physical_evaluation','result',
        'garbage_collection_intervals','host_intervals_over_20ms')}))


if __name__=='__main__':main()
