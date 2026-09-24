"""Whole-module AST equality outside the explicitly declared substitutions."""
import ast
import hashlib
from pathlib import Path
import pytest

HASHES = dict(episode='4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949',
    audit='6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3')


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'CommitmentContactAnchoredController': node.id = 'ResidualAnchoredContinuationController'
        return node
    def visit_ImportFrom(self, node):
        if node.module == 'lewm.commitment_contact_anchored_controller_development':
            assert len(node.names) == 1 and node.names[0].name == 'CommitmentContactAnchoredController'
            node.module = 'lewm.residual_anchored_continuation_controller_development'
            node.names[0].name = 'ResidualAnchoredContinuationController'
        return node
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith('COMMITMENT_CONTACT_ANCHORED_MAZE02_'):
            assert node.value in ('COMMITMENT_CONTACT_ANCHORED_MAZE02_TERMINAL_AUDIT_REQUIRED', 'COMMITMENT_CONTACT_ANCHORED_MAZE02_COLLECTED')
            node.value = node.value.replace('COMMITMENT_CONTACT_ANCHORED_MAZE02_', 'RESIDUAL_ANCHORED_CONTINUATION_MAZE_')
        return node
    def visit_keyword(self, node):
        if node.arg == 'ordinary_waypoint_commitment_contact_enabled':
            assert isinstance(node.value, ast.Constant) and node.value.value is True
            return None
        return self.generic_visit(node)
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "result['ordinary_waypoint_commitment_contact_enabled'] is True": return None
        return self.generic_visit(node)


@pytest.mark.parametrize('kind', ['episode', 'audit'])
def test_only_declared_controller_and_receipt_changes_from_bound_original(kind):
    old = Path(f'scripts/residual_anchored_continuation_maze_{kind}_development.py').read_bytes()
    assert hashlib.sha256(old).hexdigest() == HASHES[kind]
    new = Path(f'scripts/commitment_contact_anchored_maze02_{kind}_development.py').read_text()
    assert ast.dump(Normalize().visit(ast.parse(new)), include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)


def test_collector_and_auditor_use_same_prospective_controller():
    from scripts import commitment_contact_anchored_maze02_episode_development as episode
    from scripts import commitment_contact_anchored_maze02_audit_development as audit
    from lewm.commitment_contact_anchored_controller_development import CommitmentContactAnchoredController
    assert episode.CommitmentContactAnchoredController is audit.CommitmentContactAnchoredController is CommitmentContactAnchoredController
