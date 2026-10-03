"""Real-session test of the stage-1 friction hook (lewm/dev_dynamics_friction_development.py).

Builds the capability session for dev_tune maze 0 into a temporary directory through the wrapped make_session, checks
the solver read-back (floor and 27 robot geometries at mu, walls unchanged, zero physics steps), the receipt file, and
that installing after a physics step is refused. No mission runs.
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_dynamics_friction_development
"""
import json
from pathlib import Path
import tempfile

import numpy as np

from lewm.dev_dynamics_friction_development import friction_session, install_uniform_friction, solver_friction
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


def main():
    root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    spec, _ = owner.episode_inputs(root, 0, 0)
    owner.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)/'native'
        directory.mkdir()
        session = friction_session(owner.make_session, .4)(spec, directory)
        build = session.ctx.build
        receipt = json.loads((directory/'dynamics_friction.json').read_text())
        assert receipt['mu'] == .4 and receipt['physics_steps_at_install'] == 0 and len(receipt['robot_geom_ids']) == 27
        _, robot = solver_friction(build, [build.robot])
        _, floor = solver_friction(build, [build.collision_floor])
        assert np.allclose(robot, .4) and np.allclose(floor, .4)
        walls = [e for e in build.physical_environment if e is not build.collision_floor]
        _, wall = solver_friction(build, walls)
        assert len(walls) > 0 and np.allclose(wall, wall[0]) and not np.isclose(wall[0], .4), wall[:3]
        print('ok install: robot/floor 0.4, walls', round(float(wall[0]), 3))
        session.install_contact_identity()
        session.settle_recorded()
        assert int(build.scene.t) > 0
        try:
            install_uniform_friction(build, .3)
            raise AssertionError('install after physics accepted')
        except ValueError:
            print('ok refuse after physics')
        for bad in (0., 1.5, float('nan')):
            try:
                install_uniform_friction(build, bad)
                raise AssertionError(f'mu {bad} accepted')
            except ValueError:
                pass
        print('ok range checks')


if __name__ == '__main__':
    main()
