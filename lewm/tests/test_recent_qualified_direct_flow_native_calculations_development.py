"""Preserve collection and raw audit calculations under the isolated controller."""
import ast
from pathlib import Path
import pytest


@pytest.mark.parametrize('kind,name',[('episode','collect'),('episode','artifacts'),('audit','audit')])
def test_native_calculations_preserved(kind,name):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name==name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='RecentQualifiedDirectFlowController':node.id='RecentQualifiedAnchorController'
            return node
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                node.value=node.value.replace('RECENT_QUALIFIED_DIRECT_FLOW_MAZE03','RECENT_QUALIFIED_ANCHOR_MAZE01')
            return node
    old=function('scripts/recent_qualified_anchor_maze01_'+kind+'_development.py')
    new=function('scripts/recent_qualified_direct_flow_maze03_'+kind+'_development.py')
    assert ast.dump(old)==ast.dump(Normalize().visit(new))


def test_controller_directly_inherits_original_direct_flow_without_height_change():
    tree=ast.parse(Path('lewm/recent_qualified_direct_flow_controller_development.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    assert [ast.unparse(n) for n in cls.bases]==['DirectFlowFloorTransportController']
