"""Bind the complete original physical loop and full audit calculations."""
import ast
import hashlib
from pathlib import Path
import pytest

ORIGINAL = {
    'episode':'4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949',
    'audit':'6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3',
}


class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='DirectFlowResidualAnchoredController':node.id='ResidualAnchoredContinuationController'
        return node

    def visit_ImportFrom(self,node):
        if node.module=='lewm.direct_flow_residual_anchored_controller_development':
            assert len(node.names)==1 and node.names[0].name=='DirectFlowResidualAnchoredController'
            node.module='lewm.residual_anchored_continuation_controller_development'
            node.names[0].name='ResidualAnchoredContinuationController'
        return node

    def visit_Constant(self,node):
        mapping={
            'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_TERMINAL_AUDIT_REQUIRED':'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED',
            'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_COLLECTED':'RESIDUAL_ANCHORED_CONTINUATION_MAZE_COLLECTED',
        }
        if isinstance(node.value,str) and node.value in mapping:node.value=mapping[node.value]
        return node

    def visit_keyword(self,node):
        if node.arg=='direct_corner_flow_missingness_fallback_enabled':
            assert isinstance(node.value,ast.Constant) and node.value.value is True
            return None
        return self.generic_visit(node)

    def visit_Assert(self,node):
        if ast.unparse(node.test)=="result['direct_corner_flow_missingness_fallback_enabled'] is True":return None
        return self.generic_visit(node)


def source_pair(kind):
    old=Path(f'scripts/residual_anchored_continuation_maze_{kind}_development.py').read_bytes()
    assert hashlib.sha256(old).hexdigest()==ORIGINAL[kind]
    new=Path(f'scripts/no_rgb_jepa_direct_flow_maze02_{kind}_development.py').read_text()
    return old.decode(),new


def equal(old,new):
    return ast.dump(ast.parse(old),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(new)),include_attributes=False)


@pytest.mark.parametrize('kind',['episode','audit'])
def test_complete_modules_match_the_bound_original_outside_declared_changes(kind):
    assert equal(*source_pair(kind))


def test_collector_and_audit_use_the_exact_full_prefix_controller():
    from scripts import no_rgb_jepa_direct_flow_maze02_episode_development as collect
    from scripts import no_rgb_jepa_direct_flow_maze02_audit_development as audit
    from lewm.direct_flow_residual_anchored_controller_development import DirectFlowResidualAnchoredController
    assert collect.DirectFlowResidualAnchoredController is audit.DirectFlowResidualAnchoredController is DirectFlowResidualAnchoredController


def test_normalization_cannot_hide_an_abbreviated_physical_episode():
    old,new=source_pair('episode')
    assert 'range(MAX_OBSERVATIONS)' in new
    assert not equal(old,new.replace('range(MAX_OBSERVATIONS)','range(1)'))


def test_normalization_cannot_hide_a_relaxed_visibility_success_gate():
    old,new=source_pair('audit')
    assert ' and visibility and not failed' in new
    assert not equal(old,new.replace(' and visibility and not failed',''))


def test_false_fallback_identity_is_not_a_normalizable_metadata_change():
    old,new=source_pair('episode')
    with pytest.raises(AssertionError):equal(old,new.replace('direct_corner_flow_missingness_fallback_enabled=True',
        'direct_corner_flow_missingness_fallback_enabled=False'))
