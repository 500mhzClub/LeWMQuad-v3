from pathlib import Path
import ast


def test_visible_robot_builder_changes_only_visualization_contract():
    root=Path(__file__).resolve().parents[2]/'lewm_genesis/lewm_genesis'
    original=(root/'variable_height_union_rgbd_scene_development.py').read_text()
    expected=original.replace('or render_robot:','or not render_robot:').replace(
        'one CPU offscreen robot-hidden development scene required','one CPU offscreen robot-visible sensor characterization scene required').replace(
        'fixed=False,visualization=False,collision=True','fixed=False,visualization=True,collision=True').replace(
        'or robot.morph.visualization:','or not robot.morph.visualization:').replace(
        'actual collision-only Go2 geometry','actual collision-and-visual Go2 geometry')
    assert (root/'visible_robot_union_rgbd_scene_development.py').read_text().rstrip()==expected.rstrip()


def test_sensor_session_preserves_native_sampling_and_commands():
    from scripts.auxiliary_depth_visible_robot_session_development import VisibleRobotFamilySession as current
    from scripts.geometry_progress_family_session_development import GeometryProgressFamilySession as original
    assert current._build_contact_topology is original._build_contact_topology
    assert current.install_contact_identity is original.install_contact_identity
    assert current.sensor_packets is original.sensor_packets
    import inspect,textwrap
    for name in ('_sample','command_tick'):
        a=ast.parse(textwrap.dedent(inspect.getsource(getattr(current,name))))
        b=ast.parse(textwrap.dedent(inspect.getsource(getattr(original,name))))
        assert ast.dump(a)==ast.dump(b)
