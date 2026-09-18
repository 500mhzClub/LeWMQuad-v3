"""Enforce complete collector/auditor equivalence outside declared substitutions."""
import ast
import hashlib
from pathlib import Path
import pytest

HASHES = dict(episode='075ec30735843f2d1dc3fbe4bec4edaafa3917b04e4963befbba1625c2bdd4d0',
    audit='4e7f91aa7a2e88f64709f7711cef9acaaf51d3008829012b660c8c19b6a91531')


class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='SustainedHoldReorientationController':node.id='HoldReorientationController'
        return node
    def visit_ImportFrom(self,node):
        if node.module=='lewm.sustained_hold_reorientation_controller_development':
            assert len(node.names)==1 and node.names[0].name=='SustainedHoldReorientationController'
            node.module='lewm.hold_reorientation_controller_development'
            node.names[0].name='HoldReorientationController'
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value.startswith('SUSTAINED_HOLD_REORIENTATION_MAZE02_'):
            assert node.value in ('SUSTAINED_HOLD_REORIENTATION_MAZE02_TERMINAL_AUDIT_REQUIRED',
                'SUSTAINED_HOLD_REORIENTATION_MAZE02_COLLECTED')
            node.value=node.value.replace('SUSTAINED_HOLD_REORIENTATION_MAZE02_','HOLD_REORIENTATION_MAZE02_')
        return node
    def visit_keyword(self,node):
        if node.arg=='sustained_hold_reorientation_enabled':
            assert isinstance(node.value,ast.Constant) and node.value.value is True
            return None
        return self.generic_visit(node)
    def visit_Assert(self,node):
        if ast.unparse(node.test)=="result['sustained_hold_reorientation_enabled'] is True":return None
        return self.generic_visit(node)


def compare(kind,text):
    original=Path(f'scripts/hold_reorientation_maze02_{kind}_development.py').read_bytes()
    assert hashlib.sha256(original).hexdigest()==HASHES[kind]
    assert ast.dump(Normalize().visit(ast.parse(text)),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)


@pytest.mark.parametrize('kind',['episode','audit'])
def test_entire_source_matches_frozen_original_outside_explicit_changes(kind):
    compare(kind,Path(f'scripts/sustained_hold_reorientation_maze02_{kind}_development.py').read_text())


@pytest.mark.parametrize('kind,old,new',[
    ('episode','navigation_ticks=NAVIGATION_TICKS, persistent=True','navigation_ticks=4000, persistent=True'),
    ('episode','controller.observe(p, d, f','controller.observe(p, d, None'),
    ('episode','if drain == DRAIN_TICKS:','if drain == 0:'),
    ('audit',"assert result['tracker_required_for_commands']", "assert True"),
    ('audit','and visibility and not failed','and visibility'),
    ('audit',"assert json.loads(json.dumps(replay)) == row['decision']", "assert True")])
def test_guard_rejects_changes_to_budget_sensors_terminal_drain_and_audit(kind,old,new):
    text=Path(f'scripts/sustained_hold_reorientation_maze02_{kind}_development.py').read_text()
    assert text.count(old)==1
    with pytest.raises(AssertionError):compare(kind,text.replace(old,new))


def test_collect_and_raw_audit_use_same_new_controller_and_originals_remain_unchanged():
    from scripts import sustained_hold_reorientation_maze02_episode_development as episode
    from scripts import sustained_hold_reorientation_maze02_audit_development as audit
    from scripts import hold_reorientation_maze02_episode_development as old_episode
    from scripts import hold_reorientation_maze02_audit_development as old_audit
    from lewm.sustained_hold_reorientation_controller_development import SustainedHoldReorientationController
    from lewm.hold_reorientation_controller_development import HoldReorientationController
    assert episode.SustainedHoldReorientationController is audit.SustainedHoldReorientationController is SustainedHoldReorientationController
    assert old_episode.HoldReorientationController is old_audit.HoldReorientationController is HoldReorientationController
