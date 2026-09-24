from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import os
import weakref

import numpy as np

import pytest

from lewm.safety import physical_handoff_stratified_generator_successor_v1_contract as C
from lewm.safety import physical_handoff_stratified_generator_successor_v1_metrics as M
from scripts import evaluate_physical_handoff_stratified_generator_successor_v1 as E
from scripts import run_physical_handoff_stratified_generator_successor_v1 as R
from lewm_genesis import scene_builder as GENESIS_SCENE_BUILDER


def _runtime() -> dict:
    return C.build_runtime_contract("a" * 40)


def _frozen_nonoverlap_projection() -> dict:
    authority = C.PREDECESSOR_IDENTITY_PROJECTION_AUTHORITY
    seed_rows = [
        {
            "registry": name,
            "count": value["count"],
            "canonical_sorted_unique_no_lf_sha256": value[
                "canonical_sorted_unique_no_lf_sha256"
            ],
            "successor_overlap_count": 0,
        }
        for name, value in sorted(
            C.REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY.items()
        )
    ]
    return C.validate_identity_and_seed_nonoverlap(
        {
            "seed_nonoverlap": {
                "generated_seed_count": C.MAX_CANDIDATE_COUNT,
                "generated_seed_unique_count": C.MAX_CANDIDATE_COUNT,
                "registries": seed_rows,
                "all_overlap_counts_zero": True,
            },
            "broad_predecessor_identity_projection": copy.deepcopy(
                authority["broad_predecessor_identity_projection"]
            ),
            "v4_registered_and_fixture_identity_count": authority[
                "v4_registered_and_fixture_identity_count"
            ],
            "v4_registered_and_fixture_identity_projection_sha256": authority[
                "v4_registered_and_fixture_identity_projection_sha256"
            ],
            "successor_identity_count": authority["successor_identity_count"],
            "successor_identity_projection_sha256": authority[
                "successor_identity_projection_sha256"
            ],
            "successor_namespace": C.IDENTITY_NAMESPACE,
            "successor_namespace_required_prefix": f"{C.IDENTITY_NAMESPACE}-",
            "broad_predecessor_identity_overlap_count": 0,
            "broad_predecessor_scene_hash_overlap_count": 0,
            "v4_registered_and_fixture_identity_overlap_count": 0,
            "v4_fixture_seed_overlap_count": 0,
            "structured_semantic_overlap_is_not_an_identity_gate": True,
            "all_identity_and_seed_overlap_counts_zero": True,
        }
    )


def _configure_isolated_successor_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str]:
    root = tmp_path / C.OUTPUT_BASENAME
    material = tmp_path / f"{C.OUTPUT_BASENAME}_material"
    source_freeze = "a" * 40
    nonoverlap = _frozen_nonoverlap_projection()
    prohibited = tuple(
        str(root.parent / f"{root.name}{suffix}")
        for suffix in (
            "_regeneration_receipt.json",
            "_custody_receipt.json",
            "_terminal_custody_bundle.json",
        )
    )
    authority = copy.deepcopy(C.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY)
    authority.pop("content_digest")
    authority["paths"] = list(prohibited)
    monkeypatch.setattr(C, "PROHIBITED_EXTERNAL_PUBLICATION_PATHS", prohibited)
    monkeypatch.setattr(
        C,
        "PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY",
        C.attach_content_digest(authority),
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(E, "DEFAULT_OUTPUT_ROOT", root)
    monkeypatch.setattr(E, "DEFAULT_MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_freeze)
    monkeypatch.setattr(
        R,
        "_validate_v4_context_files",
        lambda: C.validate_v4_context(C.build_v4_context()),
    )
    monkeypatch.setattr(
        R, "_validate_fresh_identity_domain", lambda: copy.deepcopy(nonoverlap)
    )
    monkeypatch.setattr(R, "_V4_EVIDENCE_INODE_CACHE", set())
    monkeypatch.setattr(R, "_V4_PHYSICAL_SHARD_SHA256_CACHE", set())
    monkeypatch.setattr(
        E,
        "_rebuild_predecessor_identity_and_seed_nonoverlap",
        lambda: copy.deepcopy(nonoverlap),
    )
    monkeypatch.setattr(
        E,
        "_observe_source_freeze",
        lambda runtime: C.build_source_freeze_observation(
            source_freeze_commit=runtime["source_freeze_commit"],
            source_freeze_tree_oid="b" * 40,
            source_closure_content_digest="c" * 64,
            source_closure_file_sha256="d" * 64,
            observed_head_commit_at_scientific_reduction=runtime[
                "source_freeze_commit"
            ],
        ),
    )
    monkeypatch.setattr(
        E,
        "_require_no_v4_hardlink_reuse",
        lambda *_args, **_kwargs: {
            "v4_physical_shard_file_count": 512,
            "v4_physical_shard_sha256_overlap_count": 0,
            "v4_physical_shard_copy_reuse_detected": False,
        },
    )
    return root, material, source_freeze


class _InitialTippedSuccessorBackend:
    def __init__(self) -> None:
        from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v4 import (
            _synthetic_qualification_backend_runtime,
        )

        self.runtime = _synthetic_qualification_backend_runtime()

    def qualify_successor_candidate(self, spec: dict) -> dict:
        from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v4 import (
            _initial_tipped_arrays,
        )

        arrays = _initial_tipped_arrays(spec)
        return {
            "mode": "INITIAL_REJECTION",
            "candidate_spec_id": spec["candidate_spec_id"],
            "disposition": "INITIAL_BOUNDARY_TIPPED",
            "reason": "synthetic tipped initial boundary",
            "stage_reached": "INITIAL_BOUNDARY",
            "arrays": arrays,
            "diagnostics": {
                "intended_pose_representation": "xyz_plus_quaternion_xyzw",
                "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
                "available_simulator_diagnostics": sorted(arrays),
                "previous_applied_command_sha256": hashlib.sha256(
                    arrays["previous_applied_command"].tobytes(order="C")
                ).hexdigest(),
                "previous_applied_command_dtype": "<f8",
                "previous_applied_command_shape": [3],
            },
            "runtime_evidence": copy.deepcopy(self.runtime),
        }


def _qualified_successor_backend() -> object:
    """Build a truthful fake backend with one rejection then four Q per stream."""

    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend,
        _fake_snapshot,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v3 import (
        _snapshot_payload,
        _trace,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v4 import (
        _synthetic_qualification_backend_runtime,
    )
    from scripts import run_physical_graph_edge_handoff_qualification_v3 as V3

    zero_flags = {name: False for name in C.TERMINATION_FLAG_ORDER}

    class QualifiedSuccessorBackend(_FakePhysicalBackend):
        runtime = _synthetic_qualification_backend_runtime()

        @staticmethod
        def _snapshot_packet(spec: dict) -> tuple[dict, dict, dict]:
            teacher = _FakePhysicalBackend._teacher(spec)
            snapshot = _fake_snapshot(spec, teacher)
            snapshot["payload_bytes"] = _snapshot_payload(
                step_index=int(spec["candidate_index"])
            )
            semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
            probe_trace = _trace(
                final_digest=semantics["snapshot_semantic_digest_v1"]
            )
            comparison = V3.METRICS.compare_behavioural_probe_traces(
                probe_trace, copy.deepcopy(probe_trace)
            )
            trials = [
                {
                    "trial_index": trial_index,
                    "completed": True,
                    "trace": copy.deepcopy(probe_trace),
                    "termination_flags": copy.deepcopy(zero_flags),
                    "final_snapshot_semantic_evidence": semantics[
                        "semantic_evidence"
                    ],
                    "snapshot_behavioural_digest_v1": (
                        V3.METRICS.snapshot_behavioural_digest(probe_trace)
                    ),
                    "trace_member_manifests": R.V4._trace_member_manifests(  # noqa: SLF001
                        probe_trace
                    ),
                    "trial_pair_comparison": comparison,
                }
                for trial_index in range(2)
            ]
            probe_arrays, probe_metadata = R.V4._probe_arrays_and_metadata(  # noqa: SLF001
                semantics=semantics, trials=trials
            )
            return snapshot, probe_arrays, probe_metadata

        def qualify_successor_candidate(self, spec: dict) -> dict:
            if int(spec["attempt_index"]) == 0:
                return _InitialTippedSuccessorBackend.qualify_successor_candidate(
                    self, spec
                )
            snapshot, probe_arrays, probe_metadata = self._snapshot_packet(spec)
            value = super().qualify(spec)
            teacher_trace = value["teacher_trace"]
            value["graph"]["teacher_positive_route_progress"] = (
                R.V1._teacher_route_progress_m(  # noqa: SLF001
                    teacher_trace["base_pose_world"],
                    spec["geometry"]["selected_directed_edge"][
                        "opening_segment_world"
                    ],
                )
                > 0.0
            )
            value["graph"]["teacher_competing_port_entered"] = (
                R.V1._first_competing_crossing(  # noqa: SLF001
                    teacher_trace["base_pose_world"],
                    spec["geometry"]["competing_directed_edges"],
                )
                is not None
            )
            snapshot_sha = hashlib.sha256(snapshot["payload_bytes"]).hexdigest()
            value.update(
                {
                    "mode": "TEACHER",
                    "initial_decision_state_sha256": snapshot_sha,
                    "snapshot": snapshot,
                    "probe_arrays": probe_arrays,
                    "probe_metadata": probe_metadata,
                    "teacher_termination_flags": copy.deepcopy(zero_flags),
                    "teacher_completed_without_termination": True,
                }
            )
            value["runtime_evidence"]["teacher_snapshot_sha256"] = snapshot_sha
            return value

        def reset_fixture(self, spec: dict, payload: bytes) -> dict:
            """Shape the fake reset tape with the exact frozen V1 arithmetic."""

            raw = super().reset_fixture(spec, payload)
            timestamps = []
            for sample_index in range(C.RESET_FIXTURE_PHYSICS_SAMPLES):
                policy_step = sample_index // C.PHYSICS_STEPS_PER_POLICY_STEP
                physics_step = sample_index % C.PHYSICS_STEPS_PER_POLICY_STEP
                base_ns = policy_step * int(round(0.02 * 1.0e9))
                timestamps.append(
                    float(base_ns) / 1.0e9
                    + (physics_step + 1) * C.TEACHER_TRACE_DT_S
                )
            timestamp_tape = np.ascontiguousarray(
                np.asarray(timestamps, dtype=np.float64)
            )
            previous = np.zeros(3, dtype=np.float32)
            requested_tick = np.asarray(C.RESET_FIXTURE_COMMAND, dtype=np.float32)
            lower = np.asarray([-0.3, 0.0, -0.5], dtype=np.float32)
            upper = np.asarray([0.3, 0.0, 0.5], dtype=np.float32)
            delta = np.asarray([0.25, 0.0, 0.35], dtype=np.float32)
            requested_ticks = []
            applied_ticks = []
            for _ in range(C.RESET_FIXTURE_COMMAND_TICKS):
                requested = requested_tick.copy()
                bounded = np.clip(requested, lower, upper)
                bounded = np.clip(bounded, previous - delta, previous + delta)
                requested_ticks.append(requested.copy())
                applied_ticks.append(np.asarray(bounded, dtype=np.float32).copy())
                previous = np.asarray(bounded, dtype=np.float32).copy()
            requested_tape = np.ascontiguousarray(
                np.repeat(
                    np.asarray(requested_ticks, dtype=np.float32),
                    C.PHYSICS_STEPS_PER_COMMAND_TICK,
                    axis=0,
                ).astype(np.float64)
            )
            applied_tape = np.ascontiguousarray(
                np.repeat(
                    np.asarray(applied_ticks, dtype=np.float32),
                    C.PHYSICS_STEPS_PER_COMMAND_TICK,
                    axis=0,
                ).astype(np.float64)
            )
            for wrapper in raw["reset_trials"]:
                trace = wrapper["trace"]
                trace["timestamp_s"] = timestamp_tape.copy()
                trace["requested_command"] = requested_tape.copy()
                trace["post_slew_applied_command"] = applied_tape.copy()
                digests = R.V1._trace_digest_projection(trace)  # noqa: SLF001
                metadata = wrapper["metadata"]
                metadata["requested_command_sequence_sha256"] = digests[
                    "requested_command"
                ]
                metadata["post_slew_applied_command_sequence_sha256"] = digests[
                    "post_slew_applied_command"
                ]
                metadata["contact_sequence_sha256"] = digests[
                    "physics_contact"
                ]
            return raw

        @staticmethod
        def _shape_candidate_tape(trace: dict, branch_index: int) -> None:
            timestamps = []
            for sample_index in range(C.RESET_FIXTURE_PHYSICS_SAMPLES):
                policy_step = sample_index // C.PHYSICS_STEPS_PER_POLICY_STEP
                physics_step = sample_index % C.PHYSICS_STEPS_PER_POLICY_STEP
                base_ns = policy_step * int(round(0.02 * 1.0e9))
                timestamps.append(
                    float(base_ns) / 1.0e9
                    + (physics_step + 1) * C.TEACHER_TRACE_DT_S
                )
            previous = np.zeros(3, dtype=np.float32)
            lower = np.asarray([-0.3, 0.0, -0.5], dtype=np.float32)
            upper = np.asarray([0.3, 0.0, 0.5], dtype=np.float32)
            delta = np.asarray([0.25, 0.0, 0.35], dtype=np.float32)
            requested_ticks = []
            applied_ticks = []
            for raw in R.V1._candidate_requested_commands(branch_index):  # noqa: SLF001
                requested = np.asarray(raw, dtype=np.float32)
                clipped = np.clip(requested, lower, upper).astype(np.float32)
                applied = np.clip(
                    clipped, previous - delta, previous + delta
                ).astype(np.float32)
                requested_ticks.append(requested.copy())
                applied_ticks.append(applied.copy())
                previous = applied
            trace["timestamp_s"] = np.ascontiguousarray(
                np.asarray(timestamps, dtype=np.float64)
            )
            trace["requested_command"] = np.ascontiguousarray(
                np.repeat(
                    np.asarray(requested_ticks, dtype=np.float32), 50, axis=0
                ).astype(np.float64)
            )
            trace["post_slew_applied_command"] = np.ascontiguousarray(
                np.repeat(
                    np.asarray(applied_ticks, dtype=np.float32), 50, axis=0
                ).astype(np.float64)
            )

        def fanout(self, spec: dict, payload: bytes) -> list[dict]:
            rows = super().fanout(spec, payload)
            for row in rows:
                self._shape_candidate_tape(
                    row["trace"], int(row["candidate_index"])
                )
            return rows

        def repeat(
            self, spec: dict, payload: bytes, indices: list[int]
        ) -> list[dict]:
            rows = super().repeat(spec, payload, indices)
            for row in rows:
                self._shape_candidate_tape(
                    row["trace"], int(row["candidate_index"])
                )
            return rows

    return QualifiedSuccessorBackend()


def _allow_truthful_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_material = E._validate_persisted_material
    original_call = E._call

    def validate_material(*args: object, **kwargs: object) -> dict:
        kwargs["allow_fake_runtime"] = True
        return original_material(*args, **kwargs)

    def call(name: str, *args: object, **kwargs: object) -> object:
        if name == "recompute_metrics":
            kwargs["allow_fake_runtime"] = True
        return original_call(name, *args, **kwargs)

    monkeypatch.setattr(E, "_validate_persisted_material", validate_material)
    monkeypatch.setattr(E, "_call", call)
    monkeypatch.setattr(E, "_require_real_generator_runtime_marker", lambda _path: None)


def test_round_major_registration_indices_and_stream_paths_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path / "material")
    assert R.stream_index(C.FAMILY_IDS[0], 0) == 0
    assert R.stream_index(C.FAMILY_IDS[-1], 15) == 63
    assert R.candidate_index(C.FAMILY_IDS[0], 0, 0) == 0
    assert R.candidate_index(C.FAMILY_IDS[-1], 15, 0) == 63
    assert R.candidate_index(C.FAMILY_IDS[0], 0, 1) == 64
    spec = C.build_candidate_spec(C.FAMILY_IDS[2], 5, 17)
    assert R.candidate_index(C.FAMILY_IDS[2], 5, 17) == spec["candidate_index"]
    path = R._candidate_directory(C.FAMILY_IDS[2], 5, 17)
    assert path.name == "attempt-17"
    assert path.parent.name == spec["stream_id"]


def test_fresh_identity_domain_checks_broad_predecessors_and_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "candidate_spec_id": "phsgs-v1-a-spec",
            "scene_id": "phsgs-v1-a-scene",
            "state_id": "phsgs-v1-a-state",
            "episode_id": "phsgs-v1-a-episode",
            "graph_id": "phsgs-v1-a-graph",
        }
    ]
    prior = {
        "scene_identity": {"old-scene"},
        "scene_identity_sha256": set(),
        "episode_or_state_identity": {"old-state"},
        "numeric_seed": {17},
        "textual_path_geometry_or_source_identity": {"old-path"},
        "structured_waypoint_or_sequence_path": [["STRAIGHT_PASSAGE", 0]],
    }
    monkeypatch.setattr(C, "build_candidate_identity_manifest", lambda: candidates)
    monkeypatch.setattr(C, "validate_seed_nonoverlap", lambda _value: {"ok": True})
    monkeypatch.setattr(
        C, "validate_identity_and_seed_nonoverlap", lambda value: value
    )
    monkeypatch.setattr(C.V4, "build_candidate_specs", lambda: [])
    monkeypatch.setattr(R.V4, "_nonregistered_family_fixture_specs", lambda: [])
    monkeypatch.setattr(
        R,
        "_predecessor_identity_registries",
        lambda: ({}, prior, {"bound": True}),
    )
    result = R._validate_fresh_identity_domain()
    assert result["successor_namespace"] == "phsgs-v1"
    assert result["all_identity_and_seed_overlap_counts_zero"] is True
    assert result["structured_semantic_overlap_is_not_an_identity_gate"] is True

    prior["episode_or_state_identity"].add("phsgs-v1-a-state")
    with pytest.raises(R.ExperimentError, match="broad predecessor"):
        R._validate_fresh_identity_domain()


def test_fresh_identity_domain_rejects_non_successor_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        field: f"pgehq-v4-overlap-{field}"
        for field in ("candidate_spec_id", "scene_id", "state_id", "episode_id", "graph_id")
    }
    monkeypatch.setattr(C, "build_candidate_identity_manifest", lambda: [candidate])
    monkeypatch.setattr(C, "validate_seed_nonoverlap", lambda _value: {"ok": True})
    monkeypatch.setattr(
        C, "validate_identity_and_seed_nonoverlap", lambda value: value
    )
    monkeypatch.setattr(
        R,
        "_predecessor_identity_registries",
        lambda: (
            {},
            {
                "scene_identity": set(),
                "scene_identity_sha256": set(),
                "episode_or_state_identity": set(),
                "numeric_seed": set(),
                "textual_path_geometry_or_source_identity": set(),
                "structured_waypoint_or_sequence_path": [],
            },
            {},
        ),
    )
    with pytest.raises(R.ExperimentError, match="phsgs-v1 namespace"):
        R._validate_fresh_identity_domain()


def test_panel_index_is_compact_state_order_not_sparse_candidate_index() -> None:
    specs = [
        C.build_candidate_spec(family, stratum, 17 + (stratum % 3))
        for family in C.FAMILY_IDS
        for stratum in range(C.STRATA_PER_FAMILY)
    ]
    ordered = R._panel_ordered_selected_specs(
        {"selected_candidate_specs": list(reversed(specs))}
    )
    assert [row["state_id"] for row in ordered] == sorted(
        row["state_id"] for row in specs
    )
    assert [index for index, _row in enumerate(ordered)] == list(range(64))
    assert any(int(row["candidate_index"]) > 255 for row in ordered)
    assert any(index != int(row["candidate_index"]) for index, row in enumerate(ordered))


def test_shutdown_genesis_keeps_wrapper_state_coherent_for_reinitialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    class FakeGenesis:
        def init(self, **kwargs: object) -> None:
            events.append(("init", copy.deepcopy(kwargs)))

        def destroy(self) -> None:
            events.append(("destroy", None))

    fake = FakeGenesis()
    monkeypatch.setattr(GENESIS_SCENE_BUILDER, "_GENESIS_INITIALIZED", False)
    monkeypatch.setattr(GENESIS_SCENE_BUILDER, "_import_genesis", lambda: fake)
    monkeypatch.setattr(
        GENESIS_SCENE_BUILDER,
        "_resolve_backend",
        lambda _gs, backend: f"resolved-{backend}",
    )
    first_seed = int(C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)["procedural_seed"])
    next_seed = int(C.build_candidate_spec(C.FAMILY_IDS[0], 0, 1)["procedural_seed"])

    GENESIS_SCENE_BUILDER.initialize_genesis(backend="cpu", seed=first_seed)
    GENESIS_SCENE_BUILDER.shutdown_genesis()
    assert GENESIS_SCENE_BUILDER._GENESIS_INITIALIZED is False
    GENESIS_SCENE_BUILDER.shutdown_genesis()
    GENESIS_SCENE_BUILDER.initialize_genesis(backend="cpu", seed=next_seed)

    assert events == [
        (
            "init",
            {"backend": "resolved-cpu", "seed": first_seed & 0x7FFF_FFFF},
        ),
        ("destroy", None),
        (
            "init",
            {"backend": "resolved-cpu", "seed": next_seed & 0x7FFF_FFFF},
        ),
    ]
    assert GENESIS_SCENE_BUILDER._GENESIS_INITIALIZED is True


def test_shutdown_genesis_clears_wrapper_state_when_native_destroy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingGenesis:
        @staticmethod
        def destroy() -> None:
            raise OSError("synthetic Genesis destroy failure")

    monkeypatch.setattr(GENESIS_SCENE_BUILDER, "_GENESIS_INITIALIZED", True)
    monkeypatch.setattr(
        GENESIS_SCENE_BUILDER, "_import_genesis", lambda: FailingGenesis()
    )
    with pytest.raises(OSError, match="synthetic Genesis destroy failure"):
        GENESIS_SCENE_BUILDER.shutdown_genesis()
    assert GENESIS_SCENE_BUILDER._GENESIS_INITIALIZED is False

    monkeypatch.setattr(GENESIS_SCENE_BUILDER, "_GENESIS_INITIALIZED", True)

    def fail_import() -> object:
        raise ImportError("synthetic Genesis reimport failure")

    monkeypatch.setattr(GENESIS_SCENE_BUILDER, "_import_genesis", fail_import)
    with pytest.raises(ImportError, match="synthetic Genesis reimport failure"):
        GENESIS_SCENE_BUILDER.shutdown_genesis()
    assert GENESIS_SCENE_BUILDER._GENESIS_INITIALIZED is False


def test_preregistration_discloses_lifecycle_only_engineering_boundary() -> None:
    text = R._preregistration_text()  # noqa: SLF001
    authority = C.QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY

    assert "qualification-candidate implementation lifecycle" in text
    assert "lewm_genesis.scene_builder.shutdown_genesis()" in text
    assert "lifecycle-only engineering correction" in text
    assert "changes no physical" in text
    assert str(authority["content_digest"]) in text


def test_real_backend_owns_and_releases_only_qualification_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    created: list[object] = []

    class FakeScene:
        def __init__(self, index: int) -> None:
            self.index = index
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1
            events.append(f"destroy:{self.index}")

    class Box:
        pass

    def parent_session(_self: object, _spec: dict) -> object:
        index = len(created)
        scene = FakeScene(index)
        build = Box()
        build.scene = scene
        ctx = Box()
        ctx.build = build
        session = Box()
        session.ctx = ctx
        created.append(session)
        events.append(f"create:{index}")
        return session

    def qualify(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        for _ in range(3):
            self._session(spec)
        events.append("raw-candidate-complete")
        return {"candidate_spec_id": spec["candidate_spec_id"]}

    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_session", parent_session
    )
    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_qualify", qualify
    )
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    monkeypatch.setattr(
        R, "shutdown_genesis", lambda: events.append("shutdown_genesis")
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    assert backend.qualify_successor_candidate(spec) == {
        "candidate_spec_id": spec["candidate_spec_id"]
    }
    assert events == [
        "create:0",
        "create:1",
        "create:2",
        "raw-candidate-complete",
        "destroy:2",
        "destroy:1",
        "destroy:0",
        "shutdown_genesis",
        "gc.collect",
    ]
    assert backend._owned_qualification_sessions is None
    assert [session.ctx.build.scene.destroy_count for session in created] == [1, 1, 1]

    events.clear()
    outside = backend._session(spec)
    assert backend._owned_qualification_sessions is None
    assert events == ["create:3"]
    outside.ctx.build.scene.destroy()
    assert events == ["create:3", "destroy:3"]
    assert all(
        name not in R.GenesisSuccessorPhysicalBackend.__dict__
        for name in ("reset_fixture", "fanout", "repeat")
    )


@pytest.mark.parametrize(
    ("session_count", "failure_index"),
    ((0, None), (3, None), (3, 1)),
)
def test_owned_session_references_are_gone_before_global_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    session_count: int,
    failure_index: int | None,
) -> None:
    events: list[str] = []
    session_refs: list[weakref.ReferenceType[object]] = []

    class FakeScene:
        def __init__(self, index: int) -> None:
            self.index = index

        def destroy(self) -> None:
            events.append(f"destroy:{self.index}")
            if self.index == failure_index:
                raise OSError("synthetic teardown failure")

    class Box:
        pass

    def parent_session(_self: object, _spec: dict) -> object:
        index = len(session_refs)
        build = Box()
        build.scene = FakeScene(index)
        ctx = Box()
        ctx.build = build
        session = Box()
        session.ctx = ctx
        session_refs.append(weakref.ref(session))
        return session

    def qualify(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        for _ in range(session_count):
            self._session(spec)
        events.append("raw-candidate-complete")
        return {"candidate_spec_id": spec["candidate_spec_id"]}

    def shutdown() -> None:
        assert all(reference() is None for reference in session_refs)
        events.append("shutdown_genesis")

    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_session", parent_session
    )
    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", qualify)
    monkeypatch.setattr(R, "shutdown_genesis", shutdown)
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    if failure_index is None:
        assert backend.qualify_successor_candidate(spec) == {
            "candidate_spec_id": spec["candidate_spec_id"]
        }
    else:
        with pytest.raises(R.ExperimentError, match="scene teardown failed"):
            backend.qualify_successor_candidate(spec)
    assert events == (
        ["raw-candidate-complete"]
        + [f"destroy:{index}" for index in reversed(range(session_count))]
        + ["shutdown_genesis", "gc.collect"]
    )
    assert all(reference() is None for reference in session_refs)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_real_backend_preserves_qualification_error_while_releasing_sessions(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool,
) -> None:
    events: list[str] = []
    scenes: list[object] = []

    class FakeScene:
        def __init__(self, index: int) -> None:
            self.index = index
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1
            events.append(f"destroy:{self.index}")
            if cleanup_fails and self.index == 1:
                raise OSError("synthetic teardown failure")

    class Box:
        pass

    def parent_session(_self: object, _spec: dict) -> object:
        scene = FakeScene(len(scenes))
        scenes.append(scene)
        build = Box()
        build.scene = scene
        ctx = Box()
        ctx.build = build
        session = Box()
        session.ctx = ctx
        return session

    def fail(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        for _ in range(3):
            self._session(spec)
        events.append("qualification-error")
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_session", parent_session
    )
    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", fail)
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    monkeypatch.setattr(
        R, "shutdown_genesis", lambda: events.append("shutdown_genesis")
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    with pytest.raises(RuntimeError, match="synthetic native failure") as captured:
        backend.qualify_successor_candidate(spec)
    assert events == [
        "qualification-error",
        "destroy:2",
        "destroy:1",
        "destroy:0",
        "shutdown_genesis",
        "gc.collect",
    ]
    assert backend._owned_qualification_sessions is None
    assert [scene.destroy_count for scene in scenes] == [1, 1, 1]
    notes = getattr(captured.value, "__notes__", [])
    if cleanup_fails:
        assert notes == [
            "successor qualification lifecycle cleanup also failed: "
            "scene teardown: OSError: synthetic teardown failure"
        ]
    else:
        assert notes == []


def test_real_backend_cleanup_failure_after_success_stops_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    scenes: list[object] = []

    class FakeScene:
        def __init__(self, index: int) -> None:
            self.index = index
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1
            events.append(f"destroy:{self.index}")
            if self.index == 1:
                raise OSError("synthetic teardown failure")

    class Box:
        pass

    def parent_session(_self: object, _spec: dict) -> object:
        scene = FakeScene(len(scenes))
        scenes.append(scene)
        build = Box()
        build.scene = scene
        ctx = Box()
        ctx.build = build
        session = Box()
        session.ctx = ctx
        return session

    def qualify(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        for _ in range(3):
            self._session(spec)
        events.append("raw-candidate-complete")
        return {"candidate_spec_id": spec["candidate_spec_id"]}

    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_session", parent_session
    )
    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_qualify", qualify
    )
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    monkeypatch.setattr(
        R, "shutdown_genesis", lambda: events.append("shutdown_genesis")
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    with pytest.raises(R.ExperimentError, match="scene teardown failed") as captured:
        backend.qualify_successor_candidate(spec)
    assert isinstance(captured.value.__cause__, OSError)
    assert events == [
        "raw-candidate-complete",
        "destroy:2",
        "destroy:1",
        "destroy:0",
        "shutdown_genesis",
        "gc.collect",
    ]
    assert backend._owned_qualification_sessions is None
    assert [scene.destroy_count for scene in scenes] == [1, 1, 1]


def test_real_backend_preserves_primary_when_global_shutdown_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail(self: object, _spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        events.append("qualification-error")
        raise RuntimeError("synthetic native failure")

    def shutdown() -> None:
        events.append("shutdown_genesis")
        raise OSError("synthetic global shutdown failure")

    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", fail)
    monkeypatch.setattr(R, "shutdown_genesis", shutdown)
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    with pytest.raises(RuntimeError, match="synthetic native failure") as captured:
        backend.qualify_successor_candidate(spec)
    assert events == [
        "qualification-error",
        "shutdown_genesis",
        "gc.collect",
    ]
    assert backend._owned_qualification_sessions is None
    assert getattr(captured.value, "__notes__", []) == [
        "successor qualification lifecycle cleanup also failed: "
        "Genesis process-global shutdown: OSError: "
        "synthetic global shutdown failure"
    ]


def test_primary_and_cause_tracebacks_release_sessions_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session_refs: list[weakref.ReferenceType[object]] = []
    primary = RuntimeError("synthetic wrapped qualification failure")

    class FakeScene:
        def destroy(self) -> None:
            events.append("destroy")

    class Box:
        pass

    def parent_session(_self: object, _spec: dict) -> object:
        build = Box()
        build.scene = FakeScene()
        ctx = Box()
        ctx.build = build
        session = Box()
        session.ctx = ctx
        session_refs.append(weakref.ref(session))
        return session

    def inner_failure(session: object) -> None:
        assert session is not None
        raise ValueError("synthetic nested cause")

    def fail(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        session = self._session(spec)
        try:
            inner_failure(session)
        except ValueError as cause:
            raise primary from cause

    def shutdown() -> None:
        assert all(reference() is None for reference in session_refs)
        events.append("shutdown_genesis")

    monkeypatch.setattr(
        R.V4.GenesisGo2PhysicalBackend, "_session", parent_session
    )
    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", fail)
    monkeypatch.setattr(R, "shutdown_genesis", shutdown)
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    with pytest.raises(RuntimeError) as captured:
        backend.qualify_successor_candidate(spec)
    assert captured.value is primary
    assert isinstance(captured.value.__cause__, ValueError)
    assert captured.value.__traceback__ is not None
    assert captured.value.__cause__.__traceback__ is not None
    assert events == ["destroy", "shutdown_genesis", "gc.collect"]
    assert all(reference() is None for reference in session_refs)


def test_real_backend_global_shutdown_failure_after_success_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def qualify(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        events.append("raw-candidate-complete")
        return {"candidate_spec_id": spec["candidate_spec_id"]}

    def shutdown() -> None:
        events.append("shutdown_genesis")
        raise OSError("synthetic global shutdown failure")

    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", qualify)
    monkeypatch.setattr(R, "shutdown_genesis", shutdown)
    monkeypatch.setattr(
        R.gc, "collect", lambda: events.append("gc.collect") or 0
    )
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 0)

    with pytest.raises(
        R.ExperimentError, match="Genesis process-global shutdown failed"
    ) as captured:
        backend.qualify_successor_candidate(spec)
    assert isinstance(captured.value.__cause__, OSError)
    assert events == [
        "raw-candidate-complete",
        "shutdown_genesis",
        "gc.collect",
    ]
    assert backend._owned_qualification_sessions is None


def test_global_shutdown_failure_stops_stream_before_persistence_or_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "qualification").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_require_initialized", lambda: (_runtime(), {}))
    calls: list[int] = []

    def qualify(self: object, spec: dict, *, require_registered: bool) -> dict:
        assert require_registered is False
        calls.append(int(spec["attempt_index"]))
        return {}

    def shutdown() -> None:
        raise OSError("synthetic global shutdown failure")

    monkeypatch.setattr(R.V4.GenesisGo2PhysicalBackend, "_qualify", qualify)
    monkeypatch.setattr(R, "shutdown_genesis", shutdown)
    backend = object.__new__(R.GenesisSuccessorPhysicalBackend)
    backend.backend = "cpu"
    backend._owned_qualification_sessions = None

    with pytest.raises(
        R.ExperimentError, match="Genesis process-global shutdown failed"
    ):
        R.qualify_stream_stage(
            C.FAMILY_IDS[0], 0, backend=backend, fake_runtime=True
        )
    stream = R._stream_directory(C.FAMILY_IDS[0], 0)
    assert calls == [0]
    assert stream.is_dir()
    assert list(stream.iterdir()) == []
    assert not R._candidate_directory(C.FAMILY_IDS[0], 0, 0).exists()
    assert not R._candidate_directory(C.FAMILY_IDS[0], 0, 1).exists()
    assert not (stream / "stream_completion.json").exists()


def test_stream_runs_past_first_qualified_and_stops_exactly_at_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "qualification").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_require_initialized", lambda: (_runtime(), {}))
    calls: list[int] = []

    def qualify(spec: dict, **_kwargs: object) -> dict:
        attempt = int(spec["attempt_index"])
        calls.append(attempt)
        candidate_dir = R._candidate_directory(
            spec["family"], spec["stratum_index"], attempt
        )
        candidate_dir.mkdir()
        return {
            "qualified": attempt in {0, 2, 4, 6},
            "pool_index": spec["candidate_index"],
            "state_disposition": {
                "hard_stop": False,
                "continuation_authorized": True,
            },
        }

    monkeypatch.setattr(R, "_qualify_candidate", qualify)
    monkeypatch.setattr(
        R,
        "_terminal_record",
        lambda metadata, **_kwargs: {
            "candidate_index": metadata["pool_index"],
            "qualified": metadata["qualified"],
            "hard_stop": False,
            "continuation_authorized": True,
        },
    )
    summary = R.qualify_stream_stage(
        C.FAMILY_IDS[0], 0, backend=object(), fake_runtime=True
    )
    assert calls == list(range(7))
    assert summary["qualified_count"] == 4
    assert summary["attempt_count"] == 7
    assert summary["termination_reason"] == "TARGET_QUALIFIED_REACHED"
    assert not R._candidate_directory(C.FAMILY_IDS[0], 0, 7).exists()


def test_successor_fake_reset_uses_frozen_v1_timestamp_and_float32_tapes() -> None:
    backend = _qualified_successor_backend()
    spec = C.build_candidate_spec(C.FAMILY_IDS[0], 0, 1)
    qualification = backend.qualify_successor_candidate(spec)
    raw = backend.reset_fixture(spec, qualification["snapshot"]["payload_bytes"])
    expected_timestamp = M._expected_reset_timestamp_tape(0.0)
    expected_requested, expected_applied = M._expected_reset_command_tapes(
        np.zeros(3, dtype=np.float64)
    )
    for wrapper in raw["reset_trials"]:
        trace = wrapper["trace"]
        assert np.array_equal(trace["timestamp_s"], expected_timestamp)
        assert np.array_equal(trace["requested_command"], expected_requested)
        assert np.array_equal(
            trace["post_slew_applied_command"], expected_applied
        )
        assert trace["requested_command"][0, 0] == np.float64(
            np.float32(0.2)
        )
    fanout = backend.fanout(spec, qualification["snapshot"]["payload_bytes"])
    assert len(fanout) == len(C.CANDIDATE_IDS)
    for row in fanout:
        expected_requested, expected_applied, _nominal, _ticks = (
            M._expected_candidate_command_tapes(
                np.zeros(3, dtype=np.float64), int(row["candidate_index"])
            )
        )
        trace = row["trace"]
        assert np.array_equal(trace["timestamp_s"], expected_timestamp)
        assert np.array_equal(trace["requested_command"], expected_requested)
        assert np.array_equal(
            trace["post_slew_applied_command"], expected_applied
        )
    repeat = backend.repeat(
        spec, qualification["snapshot"]["payload_bytes"], [0, 0, 2, 2]
    )
    for row in repeat:
        expected_requested, expected_applied, _nominal, _ticks = (
            M._expected_candidate_command_tapes(
                np.zeros(3, dtype=np.float64), int(row["candidate_index"])
            )
        )
        trace = row["trace"]
        assert np.array_equal(trace["timestamp_s"], expected_timestamp)
        assert np.array_equal(trace["requested_command"], expected_requested)
        assert np.array_equal(
            trace["post_slew_applied_command"], expected_applied
        )


def test_stream_with_no_qualified_candidate_runs_all_64_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "qualification").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_require_initialized", lambda: (_runtime(), {}))
    calls: list[int] = []

    def qualify(spec: dict, **_kwargs: object) -> dict:
        calls.append(int(spec["attempt_index"]))
        R._candidate_directory(
            spec["family"], spec["stratum_index"], spec["attempt_index"]
        ).mkdir()
        return {
            "qualified": False,
            "pool_index": spec["candidate_index"],
            "state_disposition": {
                "hard_stop": False,
                "continuation_authorized": True,
            },
        }

    monkeypatch.setattr(R, "_qualify_candidate", qualify)
    monkeypatch.setattr(
        R,
        "_terminal_record",
        lambda metadata, **_kwargs: {
            "candidate_index": metadata["pool_index"],
            "qualified": False,
            "hard_stop": False,
            "continuation_authorized": True,
        },
    )
    summary = R.qualify_stream_stage(
        C.FAMILY_IDS[-1], 15, backend=object(), fake_runtime=True
    )
    assert calls == list(range(64))
    assert summary["attempt_count"] == 64
    assert summary["qualified_count"] == 0
    assert summary["termination_reason"] == "ATTEMPT_LIMIT_REACHED"


def test_stream_persists_hard_terminal_then_stops_before_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "qualification").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_require_initialized", lambda: (_runtime(), {}))
    calls: list[int] = []

    def qualify(spec: dict, **_kwargs: object) -> dict:
        attempt = int(spec["attempt_index"])
        calls.append(attempt)
        R._candidate_directory(
            spec["family"], spec["stratum_index"], attempt
        ).mkdir()
        return {
            "qualified": False,
            "pool_index": spec["candidate_index"],
            "state_disposition": {
                "hard_stop": True,
                "continuation_authorized": False,
            },
        }

    monkeypatch.setattr(R, "_qualify_candidate", qualify)
    monkeypatch.setattr(
        R,
        "_terminal_record",
        lambda metadata, **_kwargs: {
            "candidate_index": metadata["pool_index"],
            "qualified": False,
            "hard_stop": True,
            "continuation_authorized": False,
        },
    )
    with pytest.raises(R.ExperimentError, match="hard terminal disposition"):
        R.qualify_stream_stage(
            C.FAMILY_IDS[0], 0, backend=object(), fake_runtime=True
        )
    assert calls == [0]
    assert R._candidate_directory(C.FAMILY_IDS[0], 0, 0).is_dir()
    assert not R._candidate_directory(C.FAMILY_IDS[0], 0, 1).exists()
    assert not (
        R._stream_directory(C.FAMILY_IDS[0], 0) / "stream_completion.json"
    ).exists()


def test_stream_completion_rebuild_rejects_coherent_tamper() -> None:
    runtime = _runtime()
    rows = [
        {
            "candidate_index": C.candidate_index(C.FAMILY_IDS[0], 0, attempt),
            "qualified": False,
            "hard_stop": False,
            "continuation_authorized": True,
        }
        for attempt in range(C.MAX_ATTEMPTS_PER_STREAM)
    ]
    completion = R._stream_completion_document(
        runtime, C.FAMILY_IDS[0], 0, rows
    )
    assert R._validate_stream_completion_document(
        completion, runtime, C.FAMILY_IDS[0], 0, rows
    ) == completion
    tampered = copy.deepcopy(completion)
    tampered.pop("content_digest")
    tampered["qualified_count"] = 1
    tampered = C.attach_content_digest(tampered)
    with pytest.raises(R.ExperimentError, match="reopened terminal prefix"):
        R._validate_stream_completion_document(
            tampered, runtime, C.FAMILY_IDS[0], 0, rows
        )


def test_existing_partial_stream_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    stream = material / "qualification" / C.build_candidate_spec(
        C.FAMILY_IDS[0], 0, 0
    )["stream_id"]
    stream.mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_require_initialized", lambda: (_runtime(), {}))
    with pytest.raises(R.ExperimentError, match="stream is not fresh"):
        R.qualify_stream_stage(C.FAMILY_IDS[0], 0, backend=object(), fake_runtime=True)


def test_cli_is_direct_and_has_no_fake_or_orchestrator_switch() -> None:
    parser = R.build_parser()
    help_text = parser.format_help()
    assert "--fake" not in help_text
    source = Path(R.__file__).read_text()
    for forbidden in (
        "multiprocessing",
        "ProcessPoolExecutor",
        "subprocess.Popen",
        "sys.addaudithook",
        "run_v4_downstream_stage",
        "_patch_module",
        "types.FunctionType",
        "PREEXECUTION",
        "custody_bundle",
        "custom_finalizer",
    ):
        assert forbidden not in source
    assert set(parser._subparsers._group_actions[0].choices) == {
        "freeze-docs",
        "initialize",
        "qualify-stream",
        "reduce-generator",
        "freeze-panel",
        "qualify-selected-reset",
        "encode-canonical-pixels",
        "fanout-state",
        "select-development-target",
        "score-heldout",
        "repeat-state",
        "assemble-row-evidence",
        "report",
    }
    initialize = source[
        source.index("def initialize_stage") : source.index(
            "def _terminal_record"
        )
    ]
    assert initialize.index("_bound_v4_evidence_inodes()") < initialize.index(
        "OUTPUT_ROOT.mkdir"
    )
    context = source[
        source.index("def _validate_v4_context_files") : source.index(
            "def _sorted_projection_sha256"
        )
    ]
    assert context.count('"rev-list", "--parents", "-n", "1"') == 2


def test_file_binding_is_single_link_and_rejects_hardlinks(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"fresh")
    binding = R._file_binding(source, relative_to=tmp_path)
    assert binding == {
        "path": "source.bin",
        "bytes": 5,
        "sha256": "d098ab5e44b9aabb755f76d806598f43573c662b35e4a2eab1e312ec9ad195e2",
        "nlink": 1,
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }
    alias = tmp_path / "alias.bin"
    os.link(source, alias)
    with pytest.raises(R.ExperimentError, match="ordinary regular"):
        R._file_binding(source, relative_to=tmp_path)


def test_successor_material_rejects_byte_copy_of_v4_physical_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    copied = material / "selected" / "state" / "metadata.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"copied-v4-shard")
    digest = R.sha256_file(copied)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_bound_v4_evidence_inodes", lambda: set())
    monkeypatch.setattr(
        R, "_bound_v4_physical_shard_sha256s", lambda: {digest}
    )
    with pytest.raises(R.ExperimentError, match="copies a bound V4"):
        R._require_successor_inode_nonreuse(copied)


def test_material_roundtrip_binds_all_file_custody_fields_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "qualification" / "stream").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_bound_v4_evidence_inodes", lambda: set())
    monkeypatch.setattr(R, "_bound_v4_physical_shard_sha256s", lambda: set())
    directory = material / "qualification" / "stream" / "attempt-00"
    metadata = {
        "schema": f"{C.OUTPUT_BASENAME}.roundtrip_material.v1",
        "experiment_id": C.EXPERIMENT_ID,
    }
    R._write_material_shard(
        directory, metadata, {"sample": np.asarray([1.25], dtype="<f8")}
    )
    reopened, arrays = R._load_material_shard(directory)
    assert arrays["sample"].dtype.str == "<f8"
    assert reopened["payload"]["path"].endswith("attempt-00/payload.npz")
    for field in C.MATERIAL_FILE_BINDING_FIELDS:
        assert reopened["payload"][field] == R._payload_binding(
            directory / "payload.npz"
        )[field]
    assert reopened["persisted_array_evidence"]["shard_id"] == (
        "qualification/stream/attempt-00"
    )

    for field, bad_value in (
        ("nlink", 2),
        ("resolved_path_ancestor_symlink_count", 1),
    ):
        tampered = copy.deepcopy(reopened)
        tampered["payload"][field] = bad_value
        tampered.pop("content_digest")
        (directory / "metadata.json").write_bytes(
            R.canonical_bytes(C.attach_content_digest(tampered))
        )
        with pytest.raises(R.ExperimentError, match="payload binding drift"):
            R._load_material_shard(directory)
        (directory / "metadata.json").write_bytes(R.canonical_bytes(reopened))


def test_broken_symlink_ancestor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "absent", target_is_directory=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", broken / "material")
    with pytest.raises(R.ExperimentError, match="symlink ancestor"):
        R._require_no_symlink_ancestors(broken / "material" / "value.json")


def test_preencoding_outcome_gate_distinguishes_empty_and_nonempty_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    (material / "fanout").mkdir(parents=True)
    (material / "repeatability").mkdir()
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    assert R._physical_candidate_outcome_exists() is False
    (material / "fanout" / "opened").mkdir()
    assert R._physical_candidate_outcome_exists() is True
    (material / "fanout" / "opened").rmdir()
    (material / "repeatability").rmdir()
    (material / "repeatability").symlink_to(material / "absent")
    with pytest.raises(R.ExperimentError, match="material root drift"):
        R._physical_candidate_outcome_exists()


def test_runner_preflight_consumes_exact_prohibited_publication_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / C.OUTPUT_BASENAME
    paths = tuple(
        str(root.parent / f"{root.name}{suffix}")
        for suffix in (
            "_regeneration_receipt.json",
            "_custody_receipt.json",
            "_terminal_custody_bundle.json",
        )
    )
    authority = copy.deepcopy(C.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY)
    authority.pop("content_digest")
    authority["paths"] = list(paths)
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(C, "PROHIBITED_EXTERNAL_PUBLICATION_PATHS", paths)
    monkeypatch.setattr(
        C,
        "PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY",
        C.attach_content_digest(authority),
    )
    R._require_no_external_successor_publication()
    for value in paths:
        path = Path(value)
        path.write_bytes(b"prohibited\n")
        with pytest.raises(R.ExperimentError, match="external regeneration/custody"):
            R._require_no_external_successor_publication()
        path.unlink()

    corrupt = copy.deepcopy(C.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY)
    corrupt["content_digest"] = "0" * 64
    monkeypatch.setattr(C, "PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY", corrupt)
    with pytest.raises(R.ExperimentError, match="successor API failed"):
        R._require_no_external_successor_publication()


def test_generator_status_is_never_influenced_by_downstream_evidence() -> None:
    source = Path(R.__file__).read_text()
    qualifier = source[source.index("def qualify_stream_stage"):source.index("def reduce_generator_stage")]
    assert "ranker" not in qualifier
    assert "fanout" not in qualifier
    assert "heldout" not in qualifier


def test_heldout_and_repeat_simulators_open_only_after_full_frozen_gates() -> None:
    source = Path(R.__file__).read_text()
    fanout = source[
        source.index("def fanout_state_stage") : source.index(
            "def _load_encoding_material_validation"
        )
    ]
    assert fanout.index('"validate_development_target_selection"') < fanout.index(
        "collector.fanout"
    )
    heldout = source[
        source.index("def heldout_scores_stage") : source.index(
            "def repeat_state_stage"
        )
    ]
    assert heldout.index('"validate_development_target_selection"') < heldout.index(
        "runtime_ranker.score"
    )
    repeat = source[
        source.index("def repeat_state_stage") : source.index(
            "def assemble_row_evidence_stage"
        )
    ]
    assert repeat.index('"validate_development_target_selection"') < repeat.index(
        '"validate_heldout_scores"'
    ) < repeat.index("collector.repeat")
    assert "source_fanout_material=fanout_validation" in repeat

    assembly = source[
        source.index("def assemble_row_evidence_stage") : source.index(
            "def report_stage"
        )
    ]
    assert "source_fanout_material=fanout_validation" in assembly


def test_synthetic_generator_terminal_full_e2e_truthful_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the exact 4096-attempt terminal branch without a simulator."""

    root, material, source_freeze = _configure_isolated_successor_roots(
        tmp_path, monkeypatch
    )
    runtime = R.initialize_stage()
    assert runtime["source_freeze_commit"] == source_freeze
    backend = _InitialTippedSuccessorBackend()
    for family in C.FAMILY_IDS:
        for stratum in range(C.STRATA_PER_FAMILY):
            completion = R.qualify_stream_stage(
                family, stratum, backend=backend, fake_runtime=True
            )
            assert completion["attempt_count"] == C.MAX_ATTEMPTS_PER_STREAM
            assert completion["qualified_count"] == 0
    generator = R.reduce_generator_stage(fake_runtime=True)
    assert generator["status"] == C.GENERATOR_FEASIBILITY_NO_GO
    assert generator["terminal_record_count"] == C.MAX_CANDIDATE_COUNT
    assert generator["disposition_counts"]["INITIAL_BOUNDARY_TIPPED"] == (
        C.MAX_CANDIDATE_COUNT
    )
    assert sum(
        count
        for disposition, count in generator["disposition_counts"].items()
        if disposition != "INITIAL_BOUNDARY_TIPPED"
    ) == 0
    assert set(path.name for path in root.iterdir()) == set(
        C.GENERATOR_TERMINAL_OUTPUT_LEAVES[:5]
    )
    with pytest.raises(
        E.RegenerationError,
        match="production generator runtime evidence is absent or marked fake",
    ):
        E.build_reduction(root, material_root=material)
    assert not (root / "metrics.json").exists()
    assert all(not Path(value).exists() for value in C.PROHIBITED_EXTERNAL_PUBLICATION_PATHS)

    _allow_truthful_fake_evaluator(monkeypatch)
    result = E.reduce_publish_and_validate(root, material_root=material)
    assert result["primary_classification"] == C.GENERATOR_FEASIBILITY_NO_GO
    assert result["next_decision"] == C.PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION
    assert result["models_trained"] == 0
    assert result["runtime_environments"]["any_fake_runtime"] is True
    assert set(path.name for path in root.iterdir()) == set(
        C.GENERATOR_TERMINAL_OUTPUT_LEAVES
    )
    files, directories = E._material_tree_inventory(material)
    expected = C.expected_material_inventory_counts(
        C.MAX_CANDIDATE_COUNT, panel_available=False
    )
    assert len(files) == expected["file_count"] == 66 + 2 * C.MAX_CANDIDATE_COUNT
    assert len(directories) == expected["directory_count"] == 69 + C.MAX_CANDIDATE_COUNT
    assert all(not Path(value).exists() for value in C.PROHIBITED_EXTERNAL_PUBLICATION_PATHS)


def test_synthetic_available_full_downstream_e2e_truthful_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise all native successor stages without opening a real simulator."""

    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakeEncoder,
        _FakeRanker,
    )

    root, material, source_freeze = _configure_isolated_successor_roots(
        tmp_path, monkeypatch
    )
    runtime = R.initialize_stage()
    assert runtime["source_freeze_commit"] == source_freeze
    backend = _qualified_successor_backend()
    for family in C.FAMILY_IDS:
        for stratum in range(C.STRATA_PER_FAMILY):
            completion = R.qualify_stream_stage(
                family, stratum, backend=backend, fake_runtime=True
            )
            assert completion["attempt_count"] == 5
            assert completion["qualified_count"] == C.TARGET_QUALIFIED_PER_STREAM

    generator = R.reduce_generator_stage(fake_runtime=True)
    attempted = C.STREAM_COUNT * 5
    assert generator["status"] == C.GENERATOR_PANEL_AVAILABLE
    assert generator["terminal_record_count"] == attempted
    assert generator["qualified_count"] == C.PANEL_STATE_COUNT * 4
    assert generator["teacher_execution_count"] == C.PANEL_STATE_COUNT * 4
    assert generator["disposition_counts"]["INITIAL_BOUNDARY_TIPPED"] == (
        C.PANEL_STATE_COUNT
    )
    assert generator["disposition_counts"]["QUALIFIED"] == (
        C.PANEL_STATE_COUNT * 4
    )

    _runtime_value, _records, _generator_value, handoff = R._load_generator_gate()
    selected_specs = R._panel_ordered_selected_specs(handoff)
    assert len(selected_specs) == C.PANEL_STATE_COUNT
    assert any(int(spec["candidate_index"]) > 255 for spec in selected_specs)
    for spec in selected_specs:
        R.capture_selected_state_stage(
            str(spec["state_id"]), backend=backend, fake_runtime=True
        )
    panel = R.freeze_panel_stage(fake_runtime=True)
    assert len(panel["states"]) == C.PANEL_STATE_COUNT
    assert [int(row["panel_index"]) for row in panel["states"]] == list(
        range(C.PANEL_STATE_COUNT)
    )
    assert len(
        R._ordinary_json(root / "teacher_trace_index.json")["records"]
    ) == C.PANEL_STATE_COUNT * 4

    R.encode_canonical_pixels_stage(encoder=_FakeEncoder(), fake_runtime=True)
    development = [
        row for row in panel["states"] if row["role"] == "DEVELOPMENT"
    ]
    heldout = [
        row
        for row in panel["states"]
        if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    assert (len(development), len(heldout)) == (48, 16)
    for state in development:
        R.fanout_state_stage(
            str(state["state_id"]), backend=backend, fake_runtime=True
        )
    selection = R.development_target_selection_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert selection["selection_frozen"] is True
    for state in heldout:
        R.fanout_state_stage(
            str(state["state_id"]), backend=backend, fake_runtime=True
        )
    heldout_rows = R.heldout_scores_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert len(heldout_rows) == 64
    for state in heldout:
        R.repeat_state_stage(
            str(state["state_id"]), backend=backend, fake_runtime=True
        )
    assembled = R.assemble_row_evidence_stage(fake_runtime=True)
    assert assembled == {
        "candidate_fanout_count": 768,
        "heldout_score_count": 64,
        "repeatability_count": 64,
        "models_trained": 0,
    }

    with pytest.raises(
        E.RegenerationError,
        match="production generator runtime evidence is absent or marked fake",
    ):
        E.build_reduction(root, material_root=material)
    assert not (root / "metrics.json").exists()
    assert all(
        not Path(value).exists()
        for value in C.PROHIBITED_EXTERNAL_PUBLICATION_PATHS
    )

    _allow_truthful_fake_evaluator(monkeypatch)
    result = E.reduce_publish_and_validate(root, material_root=material)
    assert result["primary_classification"] in C.PRIMARY_CLASSIFICATIONS
    assert result["next_decision"] == C.NEXT_DECISION_BY_CLASSIFICATION[
        result["primary_classification"]
    ]
    assert result["models_trained"] == 0
    assert result["runtime_environments"]["any_fake_runtime"] is True
    assert set(path.name for path in root.iterdir()) == set(C.SUCCESS_OUTPUT_LEAVES)
    files, directories = E._material_tree_inventory(material)
    expected = C.expected_material_inventory_counts(
        attempted, panel_available=True
    )
    assert len(files) == expected["file_count"] == 356 + 2 * attempted
    assert len(directories) == expected["directory_count"] == 214 + attempted
    assert all(
        not Path(value).exists()
        for value in C.PROHIBITED_EXTERNAL_PUBLICATION_PATHS
    )
