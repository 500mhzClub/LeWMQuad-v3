"""Enforce complete collector/auditor equivalence outside declared substitutions."""
import ast
import hashlib
from pathlib import Path
import pytest

HASHES = dict(episode='77f8f9098160a607f5e346c0a374a921777bb15f036f3b92db7b003b950c32b3',
    audit='f8f494d507a584ed9d28450c443fcd44829dceb5697f9ceb267a76e317d3ba3b')


class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='DirectFlowCommitmentContactController':node.id='CommitmentContactAnchoredController'
        return node
    def visit_ImportFrom(self,node):
        if node.module=='lewm.direct_flow_commitment_contact_controller_development':
            assert len(node.names)==1 and node.names[0].name=='DirectFlowCommitmentContactController'
            node.module='lewm.commitment_contact_anchored_controller_development'
            node.names[0].name='CommitmentContactAnchoredController'
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value.startswith('DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_'):
            assert node.value in ('DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_TERMINAL_AUDIT_REQUIRED',
                'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_COLLECTED')
            node.value=node.value.replace('DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_','COMMITMENT_CONTACT_ANCHORED_MAZE02_')
        return node
    def visit_keyword(self,node):
        if node.arg=='direct_corner_flow_missingness_fallback_enabled':
            assert isinstance(node.value,ast.Constant) and node.value.value is True
            return None
        return self.generic_visit(node)
    def visit_Assert(self,node):
        if ast.unparse(node.test)=="result['direct_corner_flow_missingness_fallback_enabled'] is True":return None
        return self.generic_visit(node)


def compare(kind,text):
    original=Path(f'scripts/commitment_contact_anchored_maze02_{kind}_development.py').read_bytes()
    assert hashlib.sha256(original).hexdigest()==HASHES[kind]
    assert ast.dump(Normalize().visit(ast.parse(text)),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)


@pytest.mark.parametrize('kind',['episode','audit'])
def test_entire_source_matches_frozen_original_outside_explicit_changes(kind):
    compare(kind,Path(f'scripts/direct_flow_commitment_contact_maze02_{kind}_development.py').read_text())


@pytest.mark.parametrize('kind,old,new',[
    ('episode','navigation_ticks=NAVIGATION_TICKS, persistent=True','navigation_ticks=4000, persistent=True'),
    ('episode','controller.observe(p, d, f','controller.observe(p, d, None'),
    ('episode','if drain == DRAIN_TICKS:','if drain == 0:'),
    ('audit',"assert result['tracker_required_for_commands']", "assert True"),
    ('audit','and visibility and not failed','and visibility'),
    ('audit',"assert json.loads(json.dumps(replay)) == row['decision']", "assert True")])
def test_guard_rejects_changes_to_budget_sensors_terminal_drain_and_audit(kind,old,new):
    text=Path(f'scripts/direct_flow_commitment_contact_maze02_{kind}_development.py').read_text()
    assert text.count(old)==1
    with pytest.raises(AssertionError):compare(kind,text.replace(old,new))


def test_collect_and_raw_audit_use_same_new_controller_and_originals_remain_unchanged():
    from scripts import direct_flow_commitment_contact_maze02_episode_development as episode
    from scripts import direct_flow_commitment_contact_maze02_audit_development as audit
    from scripts import commitment_contact_anchored_maze02_episode_development as old_episode
    from scripts import commitment_contact_anchored_maze02_audit_development as old_audit
    from lewm.direct_flow_commitment_contact_controller_development import DirectFlowCommitmentContactController
    from lewm.commitment_contact_anchored_controller_development import CommitmentContactAnchoredController
    assert episode.DirectFlowCommitmentContactController is audit.DirectFlowCommitmentContactController is DirectFlowCommitmentContactController
    assert old_episode.CommitmentContactAnchoredController is old_audit.CommitmentContactAnchoredController is CommitmentContactAnchoredController
