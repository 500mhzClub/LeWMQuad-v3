"""Reconstruct closed successful lifecycle resource receipts against collection."""
import json
import math

from scripts import extended_return_budget_resource_guard_development as limits


def check_phase(root,episode,phase,collection):
    limits.validate_root(root)
    stream_name,result_name=limits.names(episode,phase)
    for name in (stream_name,result_name):
        path=root/name
        if path.is_symlink() or not path.is_file() or path.resolve()!=path:
            raise ValueError('actual nonsymlink lifecycle evidence required')
    count=collection['decisions']
    if type(count) is not int or not 0<=count<=8014:
        raise ValueError('actual bounded collection decision count required')
    rows=[]
    with (root/stream_name).open() as stream:
        while True:
            line=stream.readline(4097)
            if not line:break
            if len(line)>4096 or not line.endswith('\n') or len(rows)>=4*8014+4:
                raise ValueError('bounded complete lifecycle resource rows required')
            rows.append(json.loads(line))
    with (root/result_name).open() as stream:body=stream.read(32769)
    if len(body)>32768:raise ValueError('bounded resource result required')
    receipt=json.loads(body)
    if len(rows)<2:raise ValueError('initial and completed resource observations required')
    if phase=='collection':
        limits.admission(rows[0])
        cycle=('before_packet','after_packet','before_controller','after_controller')
        expected=[('begin',None)]+[(stage,frame) for frame in range(count) for stage in cycle]
        trailing=len(rows)-len(expected)-1
        if trailing:
            if (not 1<=trailing<=3 or count>=8014
                    or not (collection.get('acquisition_stop') or collection.get('physical_stop'))):
                raise ValueError('only an actual stopped acquisition may have a partial final resource cycle')
            expected += [(stage,count) for stage in cycle[:trailing]]
    else:
        expected=[('begin',None),('before_sensor_audit',None),('after_sensor_audit',None)]
        expected += [(stage,frame) for frame in range(count) for stage in ('before_controller','after_controller')]
    expected.append(('completed',None))
    if [(r['stage'],r['frame']) for r in rows]!=expected:
        raise ValueError('complete ordered lifecycle samples must match the actual collection population')
    initial=rows[0]['artifact_free_bytes'];prior=-1.
    for index,row in enumerate(rows):
        if (type(row['sample']) is not int or row['sample']!=index or row['phase']!=phase
                or row['frame'] is not None and type(row['frame']) is not int
                or any(type(row[k]) is not int or row[k]<0
                    for k in ('rss_bytes','memory_available_bytes','artifact_free_bytes','disk_consumed_since_phase_start_bytes'))
                or type(row['monotonic_s']) not in (int,float) or not math.isfinite(row['monotonic_s'])
                or row['monotonic_s']<prior or row['reasons']!=[]):
            raise ValueError('exact ordered successful resource sample required')
        prior=row['monotonic_s']
        remaining=(limits.AUDIT_ALLOWANCE_BYTES+(limits.PERSISTENCE_ALLOWANCE_BYTES if row['stage']!='completed' else 0)
            if phase=='collection' else 0)
        if (row['memory_available_bytes']<limits.MIN_AVAILABLE_RAM_BYTES
                or row['rss_bytes']>limits.MAX_WORKER_RSS_BYTES
                or row['artifact_free_bytes']<limits.RETAINED_DISK_RESERVE_BYTES+remaining
                or row['disk_consumed_since_phase_start_bytes']!=max(0,initial-row['artifact_free_bytes'])):
            raise ValueError('actual sampled headroom and phase disk allowance required')
    expected_receipt=dict(phase=phase,samples=len(rows),maximum_sampled_rss_bytes=max(r['rss_bytes'] for r in rows),
        minimum_sampled_available_ram_bytes=min(r['memory_available_bytes'] for r in rows),
        minimum_sampled_artifact_free_bytes=min(r['artifact_free_bytes'] for r in rows),
        first_breach=None,phase_completed=True,sampled_limits_passed=True,error=None,
        operating_system_memory_limit_enforced=False,between_sample_peak_bounded=False,
        cleanup_guaranteed=False,navigation_qualified=False)
    if json.dumps(receipt,sort_keys=True,allow_nan=False)!=json.dumps(expected_receipt,sort_keys=True,allow_nan=False):
        raise ValueError('complete lifecycle result must reconstruct from actual samples')
    return expected_receipt


def check(root,episode,collection):
    return dict(collection=check_phase(root,episode,'collection',collection),
        audit=check_phase(root,episode,'audit',collection),both_phases_complete=True,
        resource_artifact_authentication_required=True,between_sample_peak_bounded=False,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False)
