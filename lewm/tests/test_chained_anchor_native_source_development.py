"""The entire collector and raw audit differ only in controller and explicit labels."""
import ast
import hashlib
from pathlib import Path

import pytest

ORIGINAL = dict(episode='332a7494f3806f30df71e97a35c8a58f6d0a249f475a132b6c57c1033ac6df8a',
                audit='188dc125d38873c5eadc492c9e1bbc3f1e6991cac4180973f20ee0688ce04963')


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'ChainedAnchorResidualController': node.id = 'DirectFlowResidualAnchoredController'
        return node

    def visit_ImportFrom(self, node):
        if node.module == 'lewm.chained_anchor_residual_controller_development':
            assert len(node.names) == 1 and node.names[0].name == 'ChainedAnchorResidualController'
            node.module = 'lewm.direct_flow_residual_anchored_controller_development'
            node.names[0].name = 'DirectFlowResidualAnchoredController'
        return node

    def visit_Constant(self, node):
        labels = {
            'CHAINED_ANCHOR_MAZE02_TERMINAL_AUDIT_REQUIRED': 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_TERMINAL_AUDIT_REQUIRED',
            'CHAINED_ANCHOR_MAZE02_COLLECTED': 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_COLLECTED',
        }
        if isinstance(node.value, str) and node.value in labels: node.value = labels[node.value]
        return node

    def visit_keyword(self, node):
        if node.arg == 'chained_retained_anchor_reacquisition_enabled':
            assert isinstance(node.value, ast.Constant) and node.value.value is True
            return None
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if ast.unparse(node.test) == "result['chained_retained_anchor_reacquisition_enabled'] is True": return None
        return self.generic_visit(node)


def pair(kind):
    path = Path(f'scripts/no_rgb_jepa_direct_flow_maze02_{kind}_development.py')
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == ORIGINAL[kind]
    return original.decode(), Path(f'scripts/chained_anchor_maze02_{kind}_development.py').read_text()


def equal(original, candidate):
    return ast.dump(ast.parse(original), include_attributes=False) == ast.dump(
        Normalize().visit(ast.parse(candidate)), include_attributes=False)


@pytest.mark.parametrize('kind', ['episode', 'audit'])
def test_complete_module_preserves_bound_original_outside_declared_changes(kind):
    assert equal(*pair(kind))


def test_collector_and_audit_use_the_exact_controller_under_replay():
    from scripts import chained_anchor_maze02_episode_development as collector
    from scripts import chained_anchor_maze02_audit_development as audit
    from lewm.chained_anchor_residual_controller_development import ChainedAnchorResidualController
    assert collector.ChainedAnchorResidualController is audit.ChainedAnchorResidualController is ChainedAnchorResidualController


@pytest.mark.parametrize('before,after', [('range(MAX_OBSERVATIONS)', 'range(1)'),
    ('drain == DRAIN_TICKS', 'drain == 0'), ('session.command_tick(item[\'requested_command\'])', 'session.command_tick([0.,0.,0.])')])
def test_shortened_or_replaced_physical_execution_cannot_be_normalized(before, after):
    old, new = pair('episode')
    assert before in new
    assert not equal(old, new.replace(before, after))


@pytest.mark.parametrize('before,after', [
    (' and visibility and not failed', ''),
    ('assert json.loads(json.dumps(replay)) == row[\'decision\']', 'assert True'),
    ('state_digest(model.state_dict()) == before', 'True'),
])
def test_weakened_audit_or_success_gate_cannot_be_normalized(before, after):
    old, new = pair('audit')
    assert before in new
    assert not equal(old, new.replace(before, after))


def test_false_feature_flag_cannot_be_hidden_as_metadata():
    old, new = pair('episode')
    with pytest.raises(AssertionError):
        equal(old, new.replace('chained_retained_anchor_reacquisition_enabled=True',
                              'chained_retained_anchor_reacquisition_enabled=False'))
