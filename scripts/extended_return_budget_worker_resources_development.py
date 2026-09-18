"""Whole-worker sampled envelope through prefix verification and terminal writing."""
import json
import math
import resource

from scripts import extended_return_budget_resource_guard_development as limits

STAGES=('start','inputs_verified','collection_persisted','collection_artifacts_verified',
    'audit_written','prefix_accounted','worker_validated','inputs_reverified','artifacts_verified','terminal_written')
TOTAL_ALLOWANCE_BYTES=limits.COLLECTION_ALLOWANCE_BYTES+limits.PERSISTENCE_ALLOWANCE_BYTES+limits.AUDIT_ALLOWANCE_BYTES


def names(episode):
    limits.names(episode,'collection')
    return episode+'_worker_envelope.jsonl',episode+'_worker_envelope_result.json'


def snapshot(root):
    return limits.snapshot(root)|dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def validate_sample(row,index,initial_free,previous_clock):
    if (type(row.get('sample')) is not int or row['sample']!=index or row.get('stage')!=STAGES[index]
            or any(type(row.get(k)) is not int or row[k]<0 for k in
                ('rss_bytes','peak_rss_bytes','memory_available_bytes','artifact_free_bytes'))
            or type(row.get('monotonic_s')) not in (int,float) or not math.isfinite(row['monotonic_s'])
            or row['monotonic_s']<previous_clock):
        raise ValueError('exact ordered whole-worker resource sample required')
    if index==0:limits.admission(row)
    if (max(row['rss_bytes'],row['peak_rss_bytes'])>limits.MAX_WORKER_RSS_BYTES
            or row['memory_available_bytes']<limits.MIN_AVAILABLE_RAM_BYTES
            or row['artifact_free_bytes']<limits.RETAINED_DISK_RESERVE_BYTES
            or initial_free-row['artifact_free_bytes']>TOTAL_ALLOWANCE_BYTES):
        raise limits.ResourceLimitError('whole-worker RAM, peak RSS or cumulative disk envelope exceeded')


def summary(rows,error=None):
    return dict(samples=len(rows),stages=[r['stage'] for r in rows],
        worker_completed=bool(error is None and len(rows)==len(STAGES)),error=None if error is None else repr(error),
        maximum_sampled_rss_bytes=max((r['rss_bytes'] for r in rows),default=0),
        maximum_reported_peak_rss_bytes=max((r['peak_rss_bytes'] for r in rows),default=0),
        minimum_sampled_available_ram_bytes=min((r['memory_available_bytes'] for r in rows),default=None),
        minimum_sampled_artifact_free_bytes=min((r['artifact_free_bytes'] for r in rows),default=None),
        cumulative_disk_allowance_bytes=TOTAL_ALLOWANCE_BYTES,
        terminal_record_write_observed=bool(rows and rows[-1]['stage']=='terminal_written'),
        operating_system_memory_limit_enforced=False,between_sample_availability_bounded=False,
        parent_result_serialization_included=False,navigation_qualified=False)


class WorkerEnvelope:
    def __init__(self,root,episode):
        limits.validate_root(root);self.root=root;self.stream_name,self.result_name=names(episode)
        if (root/self.result_name).exists() or (root/self.result_name).is_symlink():
            raise ValueError('exclusive whole-worker resource receipt required')
        self.stream=(root/self.stream_name).open('x');self.rows=[];self.failure=None;self.closed=False

    def check(self,stage):
        if self.closed or self.failure is not None or len(self.rows)>=len(STAGES):
            raise limits.ResourceLimitError('closed or latched whole-worker resource envelope')
        row=dict(sample=len(self.rows),stage=stage,**snapshot(self.root))
        error=None
        try:
            validate_sample(row,len(self.rows),self.rows[0]['artifact_free_bytes'] if self.rows else row['artifact_free_bytes'],
                self.rows[-1]['monotonic_s'] if self.rows else -1.)
        except Exception as failure:error=failure;self.failure=failure
        self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush();self.rows.append(row)
        if error is not None:raise error
        return row

    def finish(self,error=None):
        if self.closed:raise ValueError('whole-worker envelope already finalized')
        self.closed=True;self.stream.close()
        result=summary(self.rows,self.failure or error)
        limits.write_json(self.root/self.result_name,result)
        return result


def check(root,episode):
    limits.validate_root(root);stream_name,result_name=names(episode)
    for name in (stream_name,result_name):
        p=root/name
        if p.is_symlink() or not p.is_file() or p.resolve()!=p:
            raise ValueError('actual nonsymlink whole-worker evidence required')
    rows=[]
    with (root/stream_name).open() as stream:
        while True:
            line=stream.readline(4097)
            if not line:break
            if len(line)>4096 or not line.endswith('\n') or len(rows)>=len(STAGES):
                raise ValueError('exact bounded whole-worker resource population required')
            row=json.loads(line)
            validate_sample(row,len(rows),rows[0]['artifact_free_bytes'] if rows else row['artifact_free_bytes'],
                rows[-1]['monotonic_s'] if rows else -1.)
            rows.append(row)
    if len(rows)!=len(STAGES):raise ValueError('all native worker stages and terminal write required')
    with (root/result_name).open() as stream:body=stream.read(32769)
    if len(body)>32768:raise ValueError('bounded whole-worker resource receipt required')
    result=json.loads(body);expected=summary(rows)
    if json.dumps(result,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('whole-worker resource receipt must reconstruct exactly')
    return expected
