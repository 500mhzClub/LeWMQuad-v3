#!/usr/bin/env python3
"""Run one provenance-template arm on a preregistered development panel."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("config/go2_generalization_v3/fast_development_v1.json"),
    )
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--template-result", type=Path, default=None)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=Path(".generated/scene_corpus/go2_generalization_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".generated/go2_fast_development/v1/baseline"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--set-arg", action="append", default=[])
    parser.add_argument("--add-flag", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = _parse_args()
    panel_path = _resolve(args.panel)
    panel = json.loads(panel_path.read_text())
    if panel.get("schema") != "lewm_go2_fast_development_panel_v0":
        raise SystemExit("unsupported panel schema")
    template_path = _resolve(
        args.template_result
        if args.template_result is not None
        else Path(panel["baseline_template"]["path"])
    )
    expected_template_sha = (
        panel["baseline_template"]["sha256"]
        if args.template_result is None
        else None
    )
    if expected_template_sha is not None and _sha256(template_path) != expected_template_sha:
        raise ValueError("baseline template hash mismatch")
    development_path = _resolve(Path(panel["development_manifest"]["path"]))
    if _sha256(development_path) != panel["development_manifest"]["sha256"]:
        raise ValueError("development manifest hash mismatch")
    development = json.loads(development_path.read_text())
    records = {
        record["scene_id"]: record for record in development["validation_scenes"]
    }
    scene_corpus = _resolve(args.scene_corpus)
    for selected in panel["scenes"]:
        record = records.get(selected["scene_id"])
        if record is None or record["manifest_sha256"] != selected["manifest_sha256"]:
            raise ValueError(f"panel scene drift: {selected['scene_id']}")
        manifest_path = (
            scene_corpus
            / "development"
            / record["family"]
            / record["scene_id"]
            / "manifest.json"
        )
        manifest_payload = json.loads(manifest_path.read_text())
        if manifest_payload.get("manifest_sha256") != record["manifest_sha256"]:
            raise ValueError(f"materialized manifest drift: {record['scene_id']}")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []

    def run(index: int, selected: dict[str, Any]) -> dict[str, Any]:
        scene_id = str(selected["scene_id"])
        output = output_dir / f"{scene_id}_{args.arm}_result.json"
        child_log = output_dir / f"{scene_id}_{args.arm}.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_go2_result_provenance_with_dataset.py"),
            "--template-result",
            str(template_path),
            "--output",
            str(output),
            "--child-log",
            str(child_log),
            f"--set-arg=--scene-corpus={scene_corpus}",
            "--set-arg=--split=development",
            f"--set-arg=--scene-id={scene_id}",
            f"--set-arg=--max-ticks={int(panel['tick_budget'])}",
        ]
        command.extend(f"--set-arg={value}" for value in args.set_arg)
        command.extend(f"--add-flag={value}" for value in args.add_flag)
        if available_cpus:
            command = ["taskset", "-c", str(available_cpus[index % len(available_cpus)]), *command]
        if args.dry_run:
            return {"scene_id": scene_id, "returncode": None, "command": command}
        environment = dict(os.environ)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        completed = subprocess.run(command, env=environment, check=False)
        return {
            "scene_id": scene_id,
            "returncode": int(completed.returncode),
            "output": str(output),
            "log": str(child_log),
            "command": command,
        }

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(run, index, selected): selected
            for index, selected in enumerate(panel["scenes"])
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['scene_id']}: returncode={result['returncode']}",
                flush=True,
            )
    results.sort(key=lambda item: item["scene_id"])
    summary = {
        "schema": "lewm_go2_fast_development_run_v0",
        "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "arm": str(args.arm),
        "template_result": str(template_path),
        "template_sha256": _sha256(template_path),
        "scene_corpus": str(scene_corpus),
        "workers": int(args.workers),
        "results": results,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if all(result["returncode"] in (0, None) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
