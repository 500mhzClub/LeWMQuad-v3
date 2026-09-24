import ast
import inspect
import textwrap
from pathlib import Path


def test_capture_only_changes_native_background_decoding():
    root=Path(__file__).resolve().parents[2]/'scripts'
    old=(root/'auxiliary_tilted_depth_capture_development.py').read_text()
    expected=old.replace('from PIL import Image','from PIL import Image\nfrom lewm.native_link_segmentation_map_development import decode')
    expected=expected.replace('        mapping={int(k):tuple(map(int,v)) for k,v in context.seg_idxc_map.items()}\n        robot_ids=[k for k,v in mapping.items() if v[0]==int(robot.idx)]',
        '        mapping,robot_ids=decode(context.seg_idxc_map,int(robot.idx))')
    expected=expected.replace('diagnostic_segmentation_map={str(k):list(v) for k,v in mapping.items()},robot_segmentation_ids=robot_ids,',
        'diagnostic_segmentation_map=mapping,robot_segmentation_ids=robot_ids,')
    assert (root/'auxiliary_tilted_depth_capture_integrity_development.py').read_text().rstrip()==expected.rstrip()


def test_collection_and_raw_audit_bodies_are_unchanged():
    from scripts import capture_go2_auxiliary_tilted_depth_prefix_v1 as old
    from scripts import capture_go2_auxiliary_tilted_depth_prefix_integrity_v1 as new
    assert new.COMMAND_TICKS==old.COMMAND_TICKS==19 and new.TRIAL==old.TRIAL and new.CASE==old.CASE
    for name in ('collect','audit'):
        a=ast.parse(textwrap.dedent(inspect.getsource(getattr(old,name))))
        b=ast.parse(textwrap.dedent(inspect.getsource(getattr(new,name))))
        assert ast.dump(a)==ast.dump(b)
