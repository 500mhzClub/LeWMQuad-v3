"""Runtime local-obstacle contract for deployment-valid navigation.

Navigation policy code should depend on :class:`LocalObstacleModel`, not on a
scene manifest or privileged planning grid. Simulation benchmarks may adapt a
privileged grid explicitly, but must report that contract as deployment-invalid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol, runtime_checkable

XY = tuple[float, float]


@dataclass(frozen=True)
class LocalObstacleContract:
    """Provenance and deployment status of one runtime obstacle model."""

    source: str
    deployment_valid: bool
    query_frame: str
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


@runtime_checkable
class LocalObstacleModel(Protocol):
    """Minimal collision-query surface consumed by local navigation control.

    ``xy`` is expressed in the controller's current working frame. A deployable
    perception implementation may maintain a rolling local map and transform
    these queries using onboard egomotion; it must not consult the scene
    manifest, oracle occupancy, or ground-truth world pose.
    """

    @property
    def contract(self) -> LocalObstacleContract:
        ...

    def is_free(self, xy: XY) -> bool:
        ...

    def nearest_free(self, xy: XY, *, max_radius_m: float | None = None) -> XY | None:
        ...


class PrivilegedGridObstacleModel:
    """Explicit deployment-invalid adapter around a simulator planning grid."""

    def __init__(self, grid) -> None:
        self._grid = grid
        self._contract = LocalObstacleContract(
            source="privileged_manifest_grid",
            deployment_valid=False,
            query_frame="simulator_world_xy",
            detail="inflated occupancy raster built from scene-manifest geometry",
        )

    @property
    def contract(self) -> LocalObstacleContract:
        return self._contract

    def is_free(self, xy: XY) -> bool:
        return bool(self._grid.is_free(xy))

    def nearest_free(self, xy: XY, *, max_radius_m: float | None = None) -> XY | None:
        return self._grid.nearest_free(xy, max_radius_m=max_radius_m)


class PerceptionObstacleModel:
    """Deployable adapter for an externally maintained perception-local map.

    This class deliberately accepts query callbacks instead of importing a
    particular depth/occupancy implementation. The provider owns sensing,
    egomotion, map updates, and query-frame transforms.
    """

    def __init__(
        self,
        is_free: Callable[[XY], bool],
        nearest_free: Callable[[XY, float | None], XY | None],
        *,
        source: str,
        query_frame: str = "controller_xy",
        detail: str = "onboard perception-derived local occupancy",
    ) -> None:
        if not source or source == "privileged_manifest_grid":
            raise ValueError("perception obstacle source must identify the onboard provider")
        self._is_free = is_free
        self._nearest_free = nearest_free
        self._contract = LocalObstacleContract(
            source=source,
            deployment_valid=True,
            query_frame=query_frame,
            detail=detail,
        )

    @property
    def contract(self) -> LocalObstacleContract:
        return self._contract

    def is_free(self, xy: XY) -> bool:
        return bool(self._is_free(xy))

    def nearest_free(self, xy: XY, *, max_radius_m: float | None = None) -> XY | None:
        return self._nearest_free(xy, max_radius_m)


def require_deployment_valid_local_obstacles(model: LocalObstacleModel) -> None:
    """Reject privileged runtime safety when a deployable-nav result is required."""

    if not model.contract.deployment_valid:
        raise RuntimeError(
            "deployment-valid local obstacles required, but runtime source is "
            f"{model.contract.source!r}: {model.contract.detail}"
        )
