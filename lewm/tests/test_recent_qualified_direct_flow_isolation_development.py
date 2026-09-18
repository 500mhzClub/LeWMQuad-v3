"""Retain the direct-flow computation and all existing causal comparison gates."""
import ast
from pathlib import Path
from lewm.direct_flow_floor_transport_controller_development import DirectFlowFloorTransportController
from lewm.recent_qualified_direct_flow_controller_development import RecentQualifiedDirectFlowController
from lewm.recent_qualified_anchor_controller_development import RecentQualifiedAnchorVisualMotion


def test_reference_change_preserves_every_other_controller_owner(monkeypatch):
    names=('registration','mapper','memory','selector','mission','model')
    owners={n:object() for n in names}
    def initialize(self,*args,**kwargs):
        for name,value in owners.items():setattr(self,name,value)
        self.motion=object()
    monkeypatch.setattr(DirectFlowFloorTransportController,'__init__',initialize)
    controller=RecentQualifiedDirectFlowController()
    assert type(controller.motion) is RecentQualifiedAnchorVisualMotion
    assert all(getattr(controller,n) is value for n,value in owners.items())
    assert RecentQualifiedDirectFlowController.__bases__==(DirectFlowFloorTransportController,)
    assert RecentQualifiedDirectFlowController.observe is DirectFlowFloorTransportController.observe
    assert RecentQualifiedDirectFlowController.advance is DirectFlowFloorTransportController.advance


def test_result_changes_only_identity_and_declared_retention_flag(monkeypatch):
    original=dict(controller='direct_flow_floor_transport_controller_v1',receipt={'same':True})
    monkeypatch.setattr(DirectFlowFloorTransportController,'_result',lambda *a,**k:original)
    controller=RecentQualifiedDirectFlowController.__new__(RecentQualifiedDirectFlowController)
    result=controller._result()
    assert result==original|dict(controller='recent_qualified_direct_flow_controller_v1',
        recent_qualified_anchor_enabled=True)
    assert original==dict(controller='direct_flow_floor_transport_controller_v1',receipt={'same':True})


def test_original_comparator_gates_preserved_except_explicit_scope_guard():
    def method(path):
        tree=ast.parse(Path(path).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrefixComparison')
        return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
    class Normalize(ast.NodeTransformer):
        def visit_BoolOp(self,node):
            node.values=[v for v in node.values if not any(isinstance(x,ast.Constant)
                and x.value=='partial_floor_height_constraint_enabled' for x in ast.walk(v))]
            return self.generic_visit(node)
        def visit_Constant(self,node):
            mapping={
                'recent_qualified_direct_flow_controller_v1':'recent_qualified_anchor_controller_v1',
                'direct_flow_floor_transport_controller_v1':'partial_height_direct_flow_controller_v1',
                'ordered original direct-flow episode and retention-only successor required':
                    'ordered original height episode and explicit retention successor required'}
            if isinstance(node.value,str):node.value=mapping.get(node.value,node.value)
            return node
    old=method('lewm/recent_qualified_anchor_prefix_development.py')
    new=method('lewm/recent_qualified_direct_flow_prefix_development.py')
    assert ast.dump(Normalize().visit(old))==ast.dump(Normalize().visit(new))
