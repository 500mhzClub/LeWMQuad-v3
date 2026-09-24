"""Sampled lifecycle headroom for the prospective longer native trial."""
import json
import math
import re
import time

import psutil

from scripts.navigation_artifact_root_development import validate_root
from scripts.run_go2_successive_choice_maze_development_v1 import write_json

GIB = 1024**3
INITIAL_RAM_BYTES = 64*GIB
MIN_AVAILABLE_RAM_BYTES = 16*GIB
MAX_WORKER_RSS_BYTES = 48*GIB
COLLECTION_ALLOWANCE_BYTES = 28*GIB
PERSISTENCE_ALLOWANCE_BYTES = 8*GIB
AUDIT_ALLOWANCE_BYTES = 8*GIB
RETAINED_DISK_RESERVE_BYTES = 40*GIB
INITIAL_DISK_BYTES = (COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_ALLOWANCE_BYTES
    +AUDIT_ALLOWANCE_BYTES+RETAINED_DISK_RESERVE_BYTES)


class ResourceLimitError(RuntimeError):
    """Abort further execution; caller still performs original best-effort cleanup."""


def admission(hardware):
    for key in ('memory_available_bytes','artifact_free_bytes'):
        if type(hardware.get(key)) is not int or hardware[key] < 0:
            raise ValueError('measured integer RAM and disk availability required')
    if (hardware['memory_available_bytes'] < INITIAL_RAM_BYTES
            or hardware['artifact_free_bytes'] < INITIAL_DISK_BYTES):
        raise ResourceLimitError('64 GiB available RAM and 84 GiB artifact space required')
    return dict(minimum_initial_available_ram_bytes=INITIAL_RAM_BYTES,
        minimum_initial_artifact_free_bytes=INITIAL_DISK_BYTES,
        minimum_runtime_available_ram_bytes=MIN_AVAILABLE_RAM_BYTES,
        maximum_sampled_worker_rss_bytes=MAX_WORKER_RSS_BYTES,
        retained_disk_reserve_bytes=RETAINED_DISK_RESERVE_BYTES,
        collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_allowance_bytes=PERSISTENCE_ALLOWANCE_BYTES,
        audit_allowance_bytes=AUDIT_ALLOWANCE_BYTES,native_scene_workers=1,
        operating_system_memory_limit_enforced=False,between_sample_peak_bounded=False)


def names(episode,phase):
    if (type(episode) is not str or re.fullmatch('[a-z0-9_]+',episode) is None
            or episode == 'sealed' or episode.startswith('sealed_') or phase not in ('collection','audit')):
        raise ValueError('ordinary exact episode and lifecycle phase required')
    return (episode+'_'+phase+'_resources.jsonl',episode+'_'+phase+'_resource_result.json')


def snapshot(root):
    return dict(monotonic_s=time.monotonic(),rss_bytes=psutil.Process().memory_info().rss,
        memory_available_bytes=psutil.virtual_memory().available,artifact_free_bytes=psutil.disk_usage(root).free)


class ResourceGuard:
    """Record every check; latch a breach even if availability subsequently recovers.

The phase wrapper preserves original persistence in its finally blocks. A
latched guard forbids another controller call, but cannot guarantee that a
failing process will have enough memory to finish cleanup.
"""
    def __init__(self,root,episode,phase):
        validate_root(root)
        self.root=root;self.phase=phase
        self.stream_name,self.result_name=names(episode,phase)
        if (root/self.result_name).exists() or (root/self.result_name).is_symlink():
            raise ValueError('exclusive resource receipt required')
        self.stream=(root/self.stream_name).open('x')
        self.initial_free=None;self.count=0;self.first_breach=None;self.maximum_rss=0
        self.minimum_ram=None;self.minimum_disk=None;self.last_clock=-1.;self.finished=False;self.last_stage=None

    def check(self,stage,frame=None):
        if self.finished:raise ValueError('closed resource guard cannot admit more work')
        if self.last_stage=='completed' or self.count>=4*8014+4:
            raise ValueError('no work after phase completion or bounded check population')
        if self.first_breach is not None:
            raise ResourceLimitError('latched resource stop: '+','.join(self.first_breach['reasons']))
        if ((self.count==0) != (stage=='begin')
                or stage not in ('begin','before_sensor_audit','after_sensor_audit','before_packet','after_packet',
                'before_controller','after_controller','completed')
                or frame is not None and (type(frame) is not int or not 0 <= frame < 8014)):
            raise ValueError('declared bounded resource-check stage and frame required')
        sample=snapshot(self.root)
        if (any(type(sample.get(k)) is not int or sample[k] < 0
                for k in ('rss_bytes','memory_available_bytes','artifact_free_bytes'))
                or type(sample.get('monotonic_s')) not in (int,float)
                or not math.isfinite(sample['monotonic_s']) or sample['monotonic_s'] < self.last_clock):
            raise ValueError('finite ordered actual resource sample required')
        self.last_clock=sample['monotonic_s']
        if self.initial_free is None:self.initial_free=sample['artifact_free_bytes']
        self.maximum_rss=max(self.maximum_rss,sample['rss_bytes'])
        self.minimum_ram=sample['memory_available_bytes'] if self.minimum_ram is None else min(self.minimum_ram,sample['memory_available_bytes'])
        self.minimum_disk=sample['artifact_free_bytes'] if self.minimum_disk is None else min(self.minimum_disk,sample['artifact_free_bytes'])
        # During collection keep separate space for persistence and the audit;
        # after collection persistence has finished, keep the audit allowance.
        remaining=(AUDIT_ALLOWANCE_BYTES+(PERSISTENCE_ALLOWANCE_BYTES if stage!='completed' else 0)
            if self.phase=='collection' else 0)
        reasons=[]
        if sample['memory_available_bytes'] < MIN_AVAILABLE_RAM_BYTES:reasons.append('AVAILABLE_RAM_FLOOR')
        if sample['rss_bytes'] > MAX_WORKER_RSS_BYTES:reasons.append('WORKER_RSS_CEILING')
        if sample['artifact_free_bytes'] < RETAINED_DISK_RESERVE_BYTES+remaining:reasons.append('DISK_RESERVE')
        # Filesystem free-space changes include every concurrent writer. They
        # cannot measure this phase's consumption; enforce actual reserve only.
        row=dict(sample=self.count,phase=self.phase,stage=stage,frame=frame,**sample,
            disk_consumed_since_phase_start_bytes=max(0,self.initial_free-sample['artifact_free_bytes']),
            reasons=reasons)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush();self.count+=1;self.last_stage=stage
        if reasons:
            self.first_breach=row
            raise ResourceLimitError('resource stop: '+','.join(reasons))
        return row

    def finish(self,error=None):
        if self.finished:raise ValueError('resource result already finalized')
        self.finished=True;self.stream.close()
        report=dict(phase=self.phase,samples=self.count,maximum_sampled_rss_bytes=self.maximum_rss,
            minimum_sampled_available_ram_bytes=self.minimum_ram,minimum_sampled_artifact_free_bytes=self.minimum_disk,
            first_breach=self.first_breach,phase_completed=bool(error is None and self.first_breach is None and self.last_stage=='completed'),
            sampled_limits_passed=bool(self.count and self.first_breach is None),
            error=None if error is None else repr(error),operating_system_memory_limit_enforced=False,
            between_sample_peak_bounded=False,cleanup_guaranteed=False,navigation_qualified=False)
        write_json(self.root/self.result_name,report)
        return report
