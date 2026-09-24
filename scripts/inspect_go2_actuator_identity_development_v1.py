#!/usr/bin/env python3
"""Construct one fresh development scene and read actuator gains, without stepping."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession, array
from lewm.physical_execution_development import build_case


def main():
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.absolute()
    if any(p == 'sealed' or p == 'sealed_test.json' or p.startswith('sealed_') for p in output.parts):
        parser.error('protected output forbidden')
    output.mkdir(exist_ok=False)
    spec = build_case('straight', 1.)
    spec.update(scene_id='go2-actuator-identity-dev-v1', procedural_seed=2026090700)
    session = None
    with (output/'process.log').open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        try:
            session = AttributedSession(spec)
            runner, policy, robot = session.ctx.runner, session.ctx.policy, session.ctx.build.robot
            dofs = runner._leg_dof_idx.tolist()
            kp, kv = array(robot.get_dofs_kp(dofs)), array(robot.get_dofs_kv(dofs))
            report = {'status':'MEASURED','physics_steps':0,'scene_spec':spec,
                'actual_kp_rollout_order':kp.tolist(),'actual_kv_rollout_order':kv.tolist(),
                'expected_checkpoint_kp':policy.env_cfg['kp'],'expected_checkpoint_kd':policy.env_cfg['kd'],
                'dof_indices_rollout_order':dofs,'simulate_action_latency':policy.simulate_action_latency,
                'checkpoint_sha256':hashlib.sha256(Path(policy.checkpoint_path).read_bytes()).hexdigest(),
                'cfg_sha256':hashlib.sha256(Path(policy.cfg_path).read_bytes()).hexdigest(),
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                    'scripts/inspect_go2_actuator_identity_development_v1.py',
                    'scripts/run_go2_contact_attributed_execution_development_v1.py',
                    'scripts/run_physical_graph_edge_handoff_qualification_v1.py',
                    'lewm_genesis/lewm_genesis/rollout.py','lewm_genesis/lewm_genesis/scene_builder.py')},
                'scope':'fresh native actuator readback only; no physics rollout or causal performance claim'}
        finally:
            if session is not None:
                session.ctx.build.scene.destroy()
            shutdown_genesis()
    with (output/'result.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
