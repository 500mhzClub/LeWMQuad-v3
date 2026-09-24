"""Only controller identity and intervention receipts change in native I/O."""
import ast
from pathlib import Path
import pytest


@pytest.mark.parametrize('kind,name', [('episode', 'collect'), ('episode', 'artifacts'), ('audit', 'audit')])
def test_native_calculations_and_full_raw_audit_are_preserved(kind, name):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'ReachedFrontierRecentQualifiedController': node.id = 'RecentQualifiedDirectFlowController'
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace('REACHED_FRONTIER_MAZE03', 'RECENT_QUALIFIED_DIRECT_FLOW_MAZE03')
            return node
        def visit_Call(self, node):
            node = self.generic_visit(node)
            node.keywords = [k for k in node.keywords if k.arg != 'reached_frontier_transition_enabled']
            return node
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "result['reached_frontier_transition_enabled'] is True": return None
            return self.generic_visit(node)
    old = function('scripts/recent_qualified_direct_flow_maze03_'+kind+'_development.py')
    new = function('scripts/reached_frontier_maze03_'+kind+'_development.py')
    assert ast.dump(old) == ast.dump(Normalize().visit(new))
