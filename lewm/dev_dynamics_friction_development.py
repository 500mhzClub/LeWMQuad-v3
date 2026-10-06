"""Dynamics perturbation, stage 1: uniform floor friction (development; Andrew, 2-3 October 2026).

docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md. Genesis combines a contact pair's friction by the maximum
of the two geometries' coefficients (verified in lewm/support_friction_challenge_development.py, which reads the solver's
per-geometry coefficients). A floor change alone therefore does nothing while the robot stays at 1.0, so, as in the
15 September lower-friction room-return trial, mu is set on the collision floor AND all 27 robot geometries, before any
physics step, and read back from the solver. Walls keep their own coefficient (1.0 in the capability scenes, read
back in the test), so wall pairs stay at max(mu, wall) = the nominal value.

`friction_session(make_session, mu)` wraps the capability owner's make_session: the session is built exactly as before,
then the friction is installed and its receipt written to <directory>/dynamics_friction.json.
"""
import json
from pathlib import Path

import numpy as np

ROBOT_GEOMS = 27
MU_RANGE = (.05, 1.)


def solver_friction(build, entities):
    ids = [int(g.idx) for e in entities for g in e.geoms]
    values = build.robot._solver.get_geoms_friction(ids)
    values = values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)
    return ids, np.asarray(values, float).reshape(-1)


def install_uniform_friction(build, mu):
    """Set mu on the collision floor and every robot geometry before any physics; verify from the solver."""
    if not (np.isfinite(mu) and MU_RANGE[0] <= mu <= MU_RANGE[1]):
        raise ValueError(f'friction coefficient must lie in {MU_RANGE}')
    if int(build.scene.t) != 0:
        raise ValueError('friction must be installed before any physics step')
    robot_ids = [int(g.idx) for g in build.robot.geoms]
    floor_ids = [int(g.idx) for g in build.collision_floor.geoms]
    if len(robot_ids) != ROBOT_GEOMS or len(floor_ids) != 1 or set(robot_ids) & set(floor_ids):
        raise ValueError('exactly 27 robot geometries and one collision-floor geometry required')
    walls = [e for e in build.physical_environment if e is not build.collision_floor]
    _, before_walls = solver_friction(build, walls) if walls else ([], np.zeros(0))
    build.robot.set_friction(float(mu))
    build.collision_floor.set_friction(float(mu))
    _, robot = solver_friction(build, [build.robot])
    _, floor = solver_friction(build, [build.collision_floor])
    _, after_walls = solver_friction(build, walls) if walls else ([], np.zeros(0))
    np.testing.assert_allclose(robot, mu, atol=1e-7, rtol=0)
    np.testing.assert_allclose(floor, mu, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(before_walls, after_walls)
    return dict(perturbation='uniform_floor_friction', mu=float(mu), pair_combination='maximum',
                robot_geom_ids=robot_ids, floor_geom_ids=floor_ids, solver_robot_friction=robot.tolist(),
                solver_floor_friction=floor.tolist(), wall_friction_unchanged=sorted(set(np.round(after_walls, 6).tolist())),
                physics_steps_at_install=int(build.scene.t),
                note='all robot geometries changed, including non-foot; wall pairs stay at max(mu, wall)')


def friction_session(make_session, mu):
    def make(spec, directory, full_frames=False):
        session = make_session(spec, directory, full_frames=full_frames)
        receipt = install_uniform_friction(session.ctx.build, mu)
        session.dynamics_friction = receipt
        Path(directory, 'dynamics_friction.json').write_text(json.dumps(receipt, indent=1)+'\n')
        return session
    make.dynamics = dict(perturbation='uniform_floor_friction', mu=float(mu))
    return make
