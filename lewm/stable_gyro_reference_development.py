"""Retain one preferred anchor alongside seven refreshed measured references."""
import numpy as np
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.orthonormal_gyro_visual_motion_development import (
    OrthonormalGyroPose, OrthonormalGyroVisualMotion)


class StableGyroReferencePose(OrthonormalGyroPose):
    def __init__(self, *, activation_frame=0):
        super().__init__()
        self.activation_frame = activation_frame
        self.stable_reference = None
        self.stable_active = False

    def observe(self, *args, **kwargs):
        self.stable_active = self.frame + 1 >= self.activation_frame
        if self.stable_active and self.stable_reference is None and self.references:
            self.stable_reference = self.references[-1]
        return super().observe(*args, **kwargs)

    def _measure(self, current, G, now):
        if not self.stable_active or self.stable_reference is None:
            return super()._measure(current, G, now)
        chronological = self.references
        # The inherited selector tries the last reference first. Keep exactly
        # the same bounded population and all its fallback/conflict checks.
        self.references = [r for r in chronological if r is not self.stable_reference] + [self.stable_reference]
        try:
            return super()._measure(current, G, now)
        finally:
            self.references = chronological

    def _remember(self, current, R, G, p, now):
        super()._remember(current, R, G, p, now)
        if not self.stable_active:
            return
        stable = self.stable_reference
        selected = (self.last_selection or {}).get('selected_reference')
        # Re-anchor from an accepted pose when the old anchor loses support or
        # the existing motion threshold is crossed. Low-overlap promotions
        # still replenish recent references without replacing a usable anchor.
        if (stable is None or selected != stable.frame
                or np.linalg.norm(p-stable.position) >= .4
                or angle(stable.rotation.T @ R) >= .35):
            self.stable_reference = self.references[-1]
        else:
            recent = [r for r in self.references if r is not stable][-7:]
            self.references = sorted([stable] + recent, key=lambda r: r.frame)


class StableGyroReferenceMotion(OrthonormalGyroVisualMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity)
        self.model = StableGyroReferencePose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        ref = self.model.stable_reference
        return super().snapshot(now_ns=now_ns) | dict(
            stable_reference_selection_active=self.model.stable_active,
            stable_reference_frame=None if ref is None else ref.frame,
            reference_promotion_treatment_activation_frame=self.model.activation_frame,
            low_overlap_reference_refresh_preserved=True,
            maximum_retained_references=8)


class CompiledFloorStableGyroReferencePose(StableGyroReferencePose):
    from lewm.jit_floor_gyro_visual_motion_development import JitFloorGyroPose as _compiled
    _prepare_plane = _compiled._prepare_plane


class CompiledFloorStableGyroReferenceMotion(StableGyroReferenceMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        from lewm.jit_floor_candidates_development import warmup
        super().__init__(identity=identity, activation_frame=activation_frame)
        warmup()
        self.model = CompiledFloorStableGyroReferencePose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(compiled_floor_candidate_predicates=True)
