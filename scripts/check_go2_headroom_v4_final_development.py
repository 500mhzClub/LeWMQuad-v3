"""Final CPU implementation checks restricted to the same six qualified packets."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from lewm import decision_headroom_v4_development as v4
from scripts.run_go2_decision_headroom_branches_development import load_bound
from scripts.qualify_go2_source_decision_packet_development import differences


def main():
    config=json.loads(Path('docs/go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json').read_text());root=Path(config['execution_caps']['output_root'])
    pairs=[(Path(s['source_root']),s) for s in config['previous_corrected_states']]
    pairs += [(root/f'source_{i:02d}',json.loads((root/f'source_{i:02d}/snapshots.json').read_text())[0]) for i in range(4)]
    checks=[]
    for source,state in pairs:
        packet=load_bound(source/f'state_{state["frame"]:04d}'/'decision.pkl',state['decision'])
        motion=np.asarray(packet['source_correction']['applied_prediction_after_yaw_ablation']);motion=np.concatenate((motion[:,:,:2],np.arctan2(motion[:,:,2:3],motion[:,:,3:4])),axis=-1)
        selected=v4.selector(packet,motion,reactive=state['source_controller']=='reactive_feedback')
        assert not differences(selected,packet['source_selection'],atol=1e-6,rtol=1e-6)
        # Remove one existing candidate's motion to check quantity-local missingness,
        # not to create a new scientific state or produce a comparative result.
        missing=motion.copy();missing[1]=np.nan
        panel=v4.row_panel(packet,{'old_data':{'R3':None,'R4':motion,'R4s':motion[v4.PERMUTATION]}},missing,state_id='implementation-only')
        assert panel['R2']['status']=='unresolved' and panel['R2']['eligibility']['candidates'][1]['eligible'] is None
        assert all(panel['R2']['eligibility']['candidates'][j]['eligible'] is not None for j in (0,2,3,4,5))
        assert panel['R3/old_data']['status']=='unresolved' and panel['R4/old_data']['status']=='available' and panel['R5c']['status']=='available'
        checks.append(dict(decision_sha256=state['decision']['sha256'],own_source_parity=True,missing_quantity_does_not_disable_other_rows=True,unknown_true_candidate_does_not_disable_other_candidate_gates=True))
    files=['lewm/decision_headroom_v4_development.py','lewm/decision_headroom_v4_collection_development.py','lewm/decision_headroom_v4_layout_runtime_development.py','scripts/run_go2_headroom_v4_development.py','scripts/run_go2_headroom_v4_source_development.py','scripts/run_go2_headroom_v4_branches_development.py','scripts/read_go2_headroom_v4_development.py']
    for p in files:ast.parse(Path(p).read_text())
    Path('docs/go2_decision_headroom_v4_final_checks_2026-09-23.json').write_text(json.dumps(dict(status='PASS',checks=checks,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},physics_steps=0,new_states=0,comparative_results_retained=False),indent=2)+'\n')
    print('PASS: six source packets; quantity-local missingness; seven implementation files parsed')

if __name__=='__main__':main()
