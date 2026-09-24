"""New layout constructor, unchanged native dynamics, RGB-only recordings."""
from lewm.eligible_floor_registration_development import bind
from lewm.fresh_visual_goal_layouts_development import specification, pack
from scripts.geometry_progress_family_session_development import GeometryProgressFamilyPhysicalInit, GeometryProgressFamilySession
from scripts.rgb_only_goal_session_development import RGBOnlyGoalCapture


class FreshVisualGoalPhysicalInit(GeometryProgressFamilyPhysicalInit):
    __init__ = bind(GeometryProgressFamilyPhysicalInit.__init__, specification=specification, pack=pack)


class FreshVisualGoalSession(RGBOnlyGoalCapture, GeometryProgressFamilySession, FreshVisualGoalPhysicalInit):
    pass
