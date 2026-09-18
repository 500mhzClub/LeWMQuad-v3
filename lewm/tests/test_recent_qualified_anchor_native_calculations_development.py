"""Native collection/evaluation differs only by the explicit new controller."""
import ast
from pathlib import Path
import pytest


@pytest.mark.parametrize('kind,name',[('episode','collect'),('episode','artifacts'),('audit','audit')])
def test_original_height_native_calculations_preserved(kind,name):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='RecentQualifiedAnchorController':node.id='PartialHeightDirectFlowController'
            return node
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('RECENT_QUALIFIED_ANCHOR_MAZE01','PARTIAL_FLOOR_HEIGHT_MAZE01')
            return node
        def visit_Call(self,node):
            node.keywords=[k for k in node.keywords if k.arg!='recent_qualified_anchor_enabled']
            return self.generic_visit(node)
        def visit_Assert(self,node):
            if ast.unparse(node.test)=="result['recent_qualified_anchor_enabled'] is True":return None
            return self.generic_visit(node)
    old=function('scripts/partial_floor_height_maze01_'+kind+'_development.py')
    new=function('scripts/recent_qualified_anchor_maze01_'+kind+'_development.py')
    assert ast.dump(old)==ast.dump(Normalize().visit(new))
