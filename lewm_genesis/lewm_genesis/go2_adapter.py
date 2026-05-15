"""Go2 asset resolution helpers for Genesis rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_go2_urdf(platform_manifest: dict[str, Any], workspace_root: Path) -> Path:
    """Resolve the Go2 URDF/Xacro path named by the platform manifest."""

    root = Path(workspace_root)
    urdf_value = (
        platform_manifest.get("robot", {}).get("urdf_xacro")
        or "third_party/unitree_go2_ros2/unitree_go2_description/urdf/unitree_go2_robot.xacro"
    )
    candidate = root / str(urdf_value)
    if candidate.is_file():
        return candidate

    upstream = platform_manifest.get("robot", {}).get("upstream_base", {})
    local_path = root / str(upstream.get("local_path", "third_party/unitree_go2_ros2"))
    package_path = str(platform_manifest.get("robot", {}).get("description_package", "unitree_go2_description"))
    fallback = local_path / package_path / "urdf" / "unitree_go2_robot.xacro"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"could not resolve Go2 URDF from manifest path: {urdf_value}")
