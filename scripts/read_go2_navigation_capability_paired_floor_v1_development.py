"""Unchanged native physical criteria, current budget and corrected cohort label."""
import argparse
from lewm.eligible_floor_registration_development import bind
from scripts import read_go2_navigation_capability_episode_development as reader
from scripts import run_go2_navigation_capability_paired_floor_v1_development as owner


def report(root):
    def save(path,value):
        if path.name=='episode_evaluation.json':
            value=dict(value,label=('Capability qualification' if value.get('role')=='validation' else 'Paired-floor development harness evidence; not validation'),
                       harness_version='v1_paired_floor',reader_owner_sha256=owner.sha(__file__))
        owner.save(path,value)
    return bind(reader.report,Budget=owner.Budget,PROTOCOL=owner.PROTOCOL,save=save)(root)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    r=report(p.parse_args().root)
    print({k:r.get(k) for k in ('controller','episode_id','round_trip_success','disallowed_contact_samples','failure_and_stall_taxonomy')})
