"""Post-hoc independence check: fresh execution need not mean new dynamics."""
import numpy as np
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_support_friction_collection_v1 import OUTPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def main():
    if (OUTPUT/'prefix_identity_audit_launch.json').exists():raise ValueError('exclusive prefix identity audit')
    if digest(OUTPUT/'raw_support_audit.json')!='8144a8bdd84dbd5534c7c166a043fd077101c371d44b97036e6337c3a9501673':raise ValueError('fixed raw audit identity required')
    old=read_json(OUTPUT,'raw_support_audit_launch.json');verify_bindings(old['source_sha256']|old['input_sha256'])
    sources=discover_sources(('scripts/audit_go2_support_friction_prefix_identity_v1.py',),old['source_sha256'])
    inputs=old['input_sha256']|{str((OUTPUT/'raw_support_audit.json').relative_to(ROOT)):digest(OUTPUT/'raw_support_audit.json')}
    a=OUTPUT/'nominal/physics_trace.npz';b=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit/physics_trace.npz'
    assert all(str(p.relative_to(ROOT)) in inputs for p in (a,b))
    write_json(OUTPUT/'prefix_identity_audit_launch.json',dict(source_sha256=sources,input_sha256=inputs,
        scope='post-hoc physics-prefix identity, not new model scoring'))
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        equal={k:bool(np.array_equal(x[k][:7250],y[k][:7250])) for k in x.files}
    verify_bindings(sources|inputs)
    write_json(OUTPUT/'prefix_identity_audit.json',dict(status='PREFIX_IDENTITY_AUDIT_COMPLETE',samples=7250,
        through_time_s=14.5,fields_equal=equal,all_fields_exact=all(equal.values()),
        independent_nominal_dynamics_prefix=False,new_seed_does_not_prove_physical_randomization=True))


if __name__=='__main__':main()
