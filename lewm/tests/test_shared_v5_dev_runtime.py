from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from lewm.models.shared_v5_target_observation_head_v1 import (
    SharedV5TargetObservationHeadConfigV1,
    SharedV5TargetObservationHeadV1,
    initialize_deterministic_mock_weights_v1,
)
from lewm.navigation.shared_v5_dev_runtime import (
    DevelopmentPhysicalFuseReceipt,
    G4CandidateBatch,
    Pose2D,
    RuntimeArtifactBindings,
    SharedV5DevMazeRuntime,
    SharedV5DevRuntimeConfigurationError,
    SharedV5DevRuntimeOrderError,
    TargetConfirmationCalibration,
)
import scripts.run_go2_shared_v5_dev_maze as runner_script


def _h(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _MapFrame:
    cell_size_m = 0.05
    content_sha256 = _h("physical-map-frame")


class _ConfigurationFrame:
    def world_to_cell(self, xy):
        return (int(float(xy[0]) // 0.1), int(float(xy[1]) // 0.1))

    def cell_center(self, cell):
        return ((cell[0] + 0.5) * 0.1, (cell[1] + 0.5) * 0.1)


class _Memory:
    def __init__(self):
        self.map_frame = _MapFrame()
        self.revision = 0

    @property
    def physical_content_sha256(self):
        return _h(f"physical-content-{self.revision}")

    def to_dict(self):
        return {"revision": self.revision, "physical_content_sha256": self.physical_content_sha256}


class _Projection:
    def __init__(self, memory):
        self.calls = 0
        self.frame = _ConfigurationFrame()
        self._memory = memory

    def project(self):
        self.calls += 1
        return SimpleNamespace(
            configuration_map_frame=self.frame,
            free_cells=frozenset({(1, 1), (2, 1), (3, 1)}),
            physical_revision=self._memory.revision,
            physical_map_frame_sha256=self._memory.map_frame.content_sha256,
            physical_content_sha256=self._memory.physical_content_sha256,
        )


class _Planner:
    def __init__(self):
        self.calls = []

    def connected_component(self, snapshot, start):
        self.calls.append(("component", snapshot, start))
        return SimpleNamespace(cells=frozenset(snapshot.free_cells))

    def frontier_cells(self, snapshot, component):
        self.calls.append(("frontiers", snapshot, component))
        return SimpleNamespace(cells=((2, 1), (3, 1)))

    def astar(self, snapshot, start, goal):
        self.calls.append(("astar", snapshot, start, goal))
        cells = ((1, 1), (2, 1), (3, 1))
        return SimpleNamespace(cells=cells[: cells.index(goal) + 1])


class _Backend:
    def __init__(self):
        self.events = []
        self.commands = []
        self.snapshot_calls = 0
        self.stop_invocations = 0
        self.stop_effects = 0
        self.stopped = False
        self.fail_preprocess = False
        self.fail_evidence = self.fail_stop = False

    def reset(self):
        self.events.append("reset")
        self.stopped = False
        self.claimed = set()

    def render_rgb(self):
        self.events.append("render")
        return torch.zeros(12, 12, 3, dtype=torch.uint8)

    def preprocess_rgb(self, frame):
        assert tuple(frame.shape) == (12, 12, 3)
        self.events.append("preprocess")
        if self.fail_preprocess:
            raise LookupError("preserved controller fault")
        return torch.zeros(1, 3, 112, 112)

    def camera_calibration_tensors(self, image):
        self.events.append("camera")
        device = image.device
        return (
            torch.tensor([[0.326, 0.0, 0.443]], device=device),
            torch.eye(3, device=device)[None],
            torch.tensor([0.0], device=device),
        )

    def pose_xy_yaw(self):
        self.events.append("pose")
        return Pose2D(0.15, 0.15, 0.0)

    def apply_command(self, command):
        self.events.append("apply")
        self.commands.append(command)

    def attempt_claim(self, *, tick_index, color, intent_sha256):
        assert color not in self.claimed
        self.claimed.add(color)
        core = {"schema": "lewm_shared_v5_dev_claim_attempt_v1", "tick_index": tick_index,
                "color": color, "intent_sha256": intent_sha256}
        return {**core, "content_sha256": runner_script._canonical_sha256(core)}
    def navigation_evidence(self, *, tick_index):
        if self.fail_evidence: raise LookupError("post-claim evidence fault")
        core = {"schema": "lewm_shared_v5_dev_backend_evidence_v1", "tick_index": tick_index,
                "coverage_pose_xy_yaw": [0.15, 0.15, 0.0],
                "visibility_opportunity_sha256": _h(f"rgb-{tick_index}"),
                "collision_observed": False, "fall_observed": False,
                "source_sha256": _h("mock-backend")}
        return {**core, "content_sha256": runner_script._canonical_sha256(core)}

    def stop(self):
        self.stop_invocations += 1
        if self.fail_stop: raise OSError("terminal stop fault")
        if self.stopped:
            return
        self.stopped = True
        self.stop_effects += 1
        self.events.append("stop")

    def observer_snapshot(self):
        self.snapshot_calls += 1
        return {"commands": len(self.commands)}


class _Model:
    def __init__(self):
        self.forward_count = 0
        self.encoder_count = 0
        self.frames = []

    def forward_frame(self, image, origin, basis, ground):
        assert tuple(image.shape) == (1, 3, 112, 112)
        assert tuple(origin.shape) == (1, 3)
        assert tuple(basis.shape) == (1, 3, 3)
        assert tuple(ground.shape) == (1,)
        self.forward_count += 1
        self.encoder_count += 1
        frame = SimpleNamespace(
            patch_tokens=torch.full((1, 4, 3), float(self.forward_count)),
            bev=torch.full((1, 2, 2, 2), float(self.forward_count)),
            evidence=object(),
        )
        self.frames.append(frame)
        return frame


class _TargetHead:
    def __init__(self, *, confirmed=False):
        self.confirmed = confirmed
        self.identities = []

    def __call__(self, patch, bev):
        self.identities.append((id(patch), id(bev)))
        presence = 0.95 if self.confirmed else 0.10
        return SimpleNamespace(
            colors=("red", "yellow", "blue", "green"),
            presence_probability=torch.full((1, 4), presence),
            quality=torch.full((1, 4), 0.95),
            uncertainty=torch.full((1, 4), 0.10),
            bearing_mean_rad=torch.zeros(1, 4),
            range_mean_m=torch.full((1, 4), 0.25),
        )


class _Fuser:
    def __init__(self, *, mode="valid"):
        self.calls = []
        self.mode = mode

    def fuse(self, *, evidence, pose, tick_index, memory, **camera):
        assert type(memory) is _Memory
        assert set(camera) == {
            "camera_origin_body_m",
            "camera_basis_body_fru",
            "ground_plane_z_body_m",
        }
        self.calls.append((evidence, pose, tick_index, memory))
        if self.mode == "wrong_memory":
            other = _Memory()
            other.revision = 1
            return DevelopmentPhysicalFuseReceipt(
                memory=other,
                physical_map_frame_sha256=other.map_frame.content_sha256,
                revision_before=0,
                revision_after=1,
                physical_content_sha256=other.physical_content_sha256,
            )
        before = memory.revision
        memory.revision += 1
        return DevelopmentPhysicalFuseReceipt(
            memory=memory,
            physical_map_frame_sha256=memory.map_frame.content_sha256,
            revision_before=before,
            revision_after=memory.revision + (1 if self.mode == "stale" else 0),
            physical_content_sha256=memory.physical_content_sha256,
        )


class _Scores:
    def selected_row_indices(self):
        return (1,)


class _G4Head:
    def __init__(self):
        self.identities = []

    def __call__(self, patch, bev, candidates):
        self.identities.append((id(patch), id(bev), candidates))
        return _Scores()


class _CandidateBuilder:
    def __init__(self):
        self.calls = []
        self.batches = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch = object()
        self.batches.append(batch)
        return G4CandidateBatch(cells=tuple(kwargs["nearest_cells"]), head_batch=batch)


def _artifacts(*, g4: bool) -> RuntimeArtifactBindings:
    return RuntimeArtifactBindings(
        shared_checkpoint_sha256=_h("shared"),
        g2_report_sha256=_h("g2"),
        physical_calibration_sha256=_h("physical"),
        target_head_checkpoint_sha256=_h("target"),
        target_calibration_sha256=_h("target-calibration"),
        g4_head_checkpoint_sha256=_h("g4") if g4 else None,
        g4_calibration_sha256=_h("g4-calibration") if g4 else None,
    )


def _runtime(*, g4: bool = True, confirmed: bool = False, fuser_mode="valid"):
    model = _Model()
    target = _TargetHead(confirmed=confirmed)
    fuser = _Fuser(mode=fuser_mode)
    g4_head = _G4Head() if g4 else None
    builder = _CandidateBuilder() if g4 else None
    memory = _Memory()
    runtime = SharedV5DevMazeRuntime(
        model=model,
        target_head=target,
        physical_fuser=fuser,
        physical_memory=memory,
        projection=_Projection(memory),
        planner=_Planner(),
        target_calibration=TargetConfirmationCalibration(0.8, 0.8, 0.5, 4.0),
        artifacts=_artifacts(g4=g4),
        target_color="red",
        g4_head=g4_head,
        g4_candidate_builder=builder,
        frontier_cap=2,
    )
    runtime.development_authority = {"observer": {"source_path": "lewm/tests/test_shared_v5_dev_runtime.py"}}
    return runtime, model, target, fuser, g4_head, builder


def test_one_render_preprocess_encoder_and_cached_identity_per_visual_tick() -> None:
    runtime, model, target, fuser, g4, builder = _runtime()
    backend = _Backend()
    result = runtime.run_controller(backend, visual_ticks=3)

    result.counters.assert_one_frame_per_tick()
    assert result.counters.visual_ticks == 3
    assert result.counters.g4_head_calls == 3
    assert model.forward_count == model.encoder_count == 3
    assert len(target.identities) == len(g4.identities) == len(fuser.calls) == 3
    for index, frame in enumerate(model.frames):
        assert target.identities[index] == (id(frame.patch_tokens), id(frame.bev))
        assert g4.identities[index][:2] == (id(frame.patch_tokens), id(frame.bev))
        assert g4.identities[index][2] is builder.batches[index]
    assert backend.events[0] == "reset"
    assert backend.events.count("render") == backend.events.count("preprocess") == 3
    assert all(decision.goal_configuration_cell == (3, 1) for decision in result.decisions)
    assert runtime._sealed is True
    assert backend.stopped is True
    assert backend.stop_invocations == backend.stop_effects == 1
    with pytest.raises(SharedV5DevRuntimeOrderError, match="sealed"):
        runtime.tick(backend)


def test_controller_fault_is_preserved_while_backend_stops_and_seals(tmp_path) -> None:
    runtime, *_ = _runtime(g4=False)
    backend = _Backend()
    backend.fail_preprocess = True
    with pytest.raises(LookupError, match="preserved controller fault") as caught:
        runtime.run_controller(backend, visual_ticks=2)
    assert caught.value.lewm_terminal_fault_record.physical_revision == 0
    assert runtime._sealed is True
    assert backend.stopped is True
    assert backend.stop_invocations == backend.stop_effects == 1
    tick_path = tmp_path / "tick-fault.json"; runner_script._publish_exclusive(tick_path, runner_script._jsonable_fault(caught.value.lewm_terminal_fault_record, runtime.development_authority))
    assert tick_path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError): runner_script._publish_exclusive(tick_path, {})
    with pytest.raises(SharedV5DevRuntimeOrderError, match="sealed"):
        runtime.tick(backend)
    stop_runtime, *_ = _runtime(g4=False); stop_backend = _Backend(); stop_backend.fail_stop = True
    with pytest.raises(OSError, match="terminal stop fault") as stopped:
        stop_runtime.run_controller(stop_backend, visual_ticks=1)
    assert stopped.value.lewm_terminal_fault_record.stage == "stop"
    runner_script._publish_exclusive(tmp_path / "stop-fault.json", runner_script._jsonable_fault(stopped.value.lewm_terminal_fault_record, stop_runtime.development_authority))


@pytest.mark.parametrize("mode", ["stale", "wrong_memory"])
def test_fuse_receipt_rejects_stale_revision_or_wrong_memory(mode) -> None:
    runtime, *_ = _runtime(g4=False, fuser_mode=mode)
    backend = _Backend()
    with pytest.raises(
        RuntimeError,
        match="physical fuse|exactly one",
    ):
        runtime.run_controller(backend, visual_ticks=1)
    assert runtime._sealed is True
    assert backend.stop_effects == 1
    assert runtime.terminal_fault.physical_revision == (1 if mode == "stale" else 0)
    assert runtime.terminal_fault.post_fuse_mutation is (mode == "stale")


def test_reset_tick_order_and_observer_isolation(monkeypatch) -> None:
    runtime, *_ = _runtime(g4=False)
    backend = _Backend()
    with pytest.raises(SharedV5DevRuntimeOrderError, match="reset"):
        runtime.tick(backend)

    seen = []

    def observer(*, controller_run, observer_snapshot):
        seen.append((runtime._sealed, len(controller_run.decisions), observer_snapshot))
        with pytest.raises(SharedV5DevRuntimeOrderError, match="sealed"):
            runtime.tick(backend)
        return {"score": -999, "attempted_controller_feedback": True}

    monkeypatch.setattr(runner_script, "_load_callable", lambda *_args, **_kwargs: observer)
    run, observed = runner_script.run_controller_then_observer(
        runtime,
        backend,
        visual_ticks=2,
        observer_spec="fake_observer:score",
    )
    assert seen == [(True, 2, {"commands": 2})]
    assert observed["score"] == -999
    assert [decision.tick_index for decision in run.decisions] == [0, 1]
    assert backend.snapshot_calls == 1
    assert backend.stop_invocations == 2
    assert backend.stop_effects == 1
    assert backend.events.index("reset") < backend.events.index("render")


def test_trained_target_checkpoint_roundtrip_and_untrained_rejection(tmp_path) -> None:
    config = SharedV5TargetObservationHeadConfigV1(
        patch_feature_dim=3,
        bev_feature_dim=2,
        hidden_dim=8,
        color_embedding_dim=4,
    )
    source = SharedV5TargetObservationHeadV1(config)
    initialize_deterministic_mock_weights_v1(source, seed=17)
    source.eval()
    checkpoint = tmp_path / "target.pt"
    torch.save(
        {
            "schema": "lewm_go2_shared_v5_target_head_checkpoint_v1",
            "trained": True,
            "config": asdict(config),
            "config_sha256": config.content_sha256,
            "shared_feature_contract_sha256": _h("shared-feature"),
            "state_dict_sha256": runner_script.tensor_state_dict_sha256(source.state_dict()),
            "state_dict": source.state_dict(),
        },
        checkpoint,
    )
    restored, digest, _, _ = runner_script.load_target_head(
        checkpoint,
        device=torch.device("cpu"),
        shared_feature_sha256=_h("shared-feature"),
    )
    assert digest == runner_script.file_sha256(checkpoint)
    patch = torch.linspace(0.0, 1.0, 12).reshape(1, 4, 3)
    bev = torch.linspace(0.0, 1.0, 8).reshape(1, 2, 2, 2)
    with torch.inference_mode():
        expected = source(patch, bev)
        actual = restored(patch, bev)
    assert torch.equal(expected.presence_probability, actual.presence_probability)
    assert torch.equal(expected.range_mean_m, actual.range_mean_m)

    untrained = tmp_path / "untrained.pt"
    torch.save(
        {
            "schema": "lewm_go2_shared_v5_target_head_checkpoint_v1",
            "trained": False,
            "config": asdict(config),
            "config_sha256": config.content_sha256,
            "shared_feature_contract_sha256": _h("shared-feature"),
            "state_dict_sha256": runner_script.tensor_state_dict_sha256(source.state_dict()),
            "state_dict": source.state_dict(),
        },
        untrained,
    )
    with pytest.raises(SharedV5DevRuntimeConfigurationError, match="not explicitly marked trained"):
        runner_script.load_target_head(untrained, device=torch.device("cpu"),
                                       shared_feature_sha256=_h("shared-feature"))


def test_claim_is_at_most_once_and_tick_evidence_is_chained(tmp_path) -> None:
    runtime, *_ = _runtime(g4=False, confirmed=True)
    runtime.planner.connected_component = lambda snapshot, start: SimpleNamespace(cells=frozenset({start}))
    backend = _Backend()
    result = runtime.run_controller(backend, visual_ticks=2)
    assert len(result.claim_journal) == len(backend.claimed) == 1
    assert result.decisions[0].claim_receipt is not None
    assert result.decisions[1].claim_receipt is None
    assert result.tick_evidence[1].previous_chain_sha256 == result.tick_evidence[0].chain_sha256
    assert all(not row.collision_observed and not row.fall_observed for row in result.tick_evidence)
    fault_runtime, *_ = _runtime(g4=False, confirmed=True); fault_runtime.planner.connected_component = runtime.planner.connected_component
    fault_backend = _Backend(); fault_backend.fail_evidence = True
    with pytest.raises(LookupError, match="post-claim") as failed: fault_runtime.run_controller(fault_backend, visual_ticks=1)
    fault = failed.value.lewm_terminal_fault_record
    assert len(fault.claim_journal) == 1 and fault.last_claim_intent_sha256 and fault.last_command.primitive == "hold"
    runner_script._publish_exclusive(tmp_path / "claim-fault.json", runner_script._jsonable_fault(fault, fault_runtime.development_authority))


def test_reset_rejects_nonzero_memory_and_old_loose_g2_report(tmp_path) -> None:
    runtime, *_ = _runtime(g4=False)
    runtime.physical_memory.revision = 1
    with pytest.raises(SharedV5DevRuntimeOrderError, match="revision zero"):
        runtime.run_controller(_Backend(), visual_ticks=1)
    loose = tmp_path / "g2.json"
    loose.write_text('{"status":"PASS","post_g2_qualified":true,"extra":1}')
    with pytest.raises(Exception, match="publication"):
        runner_script.load_shared_v5(tmp_path / "missing.pt", loose, device=torch.device("cpu"))


@pytest.mark.parametrize("mutation", ["role", "extra"])
def test_calibration_rejects_role_swap_and_extra_field(tmp_path, mutation) -> None:
    core = {"schema": "lewm_go2_shared_v5_calibration_binding_v1", "role": "target_confirmation",
            "qualified": True, "checkpoint_sha256": _h("checkpoint"), "config_sha256": _h("config"),
            "shared_feature_contract_sha256": _h("feature"), "payload": {"target_confirmation": {}}}
    if mutation == "role":
        core["role"] = "g4_frontier_value"
    value = {**core, "content_sha256": runner_script._canonical_sha256(core)}
    if mutation == "extra":
        value["unexpected"] = True
    path = tmp_path / "calibration.json"
    path.write_text(__import__("json").dumps(value))
    with pytest.raises(SharedV5DevRuntimeConfigurationError):
        runner_script.load_qualified_calibration(
            path, purpose="target calibration", role="target_confirmation",
            checkpoint_sha256=_h("checkpoint"), config_sha256=_h("config"),
            shared_feature_sha256=_h("feature"),
        )


def test_scene_authority_rejects_heldout_role_before_scene_load(tmp_path, monkeypatch) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "manifest.json"; manifest.write_text('{"scene_id":"x","manifest_sha256":"' + _h("scene") + '","split":"heldout"}')
    platform = tmp_path / "platform.yaml"
    platform.write_text("platform: test\n")
    core = {"schema": "lewm_go2_shared_v5_development_scene_authority_v1", "authorized": True,
            "role": "development", "scene": {"path": "scene", "scene_id": "x", "manifest_sha256": _h("scene"), "manifest_file_sha256": runner_script.file_sha256(manifest), "genesis_file_sha256": _h("genesis")},
            "platform_manifest": {"path": "platform.yaml", "file_sha256": runner_script.file_sha256(platform)},
            "observer": None}
    authority = tmp_path / "authority.json"
    authority.write_text(__import__("json").dumps({**core, "content_sha256": runner_script._canonical_sha256(core)}))
    args = SimpleNamespace(development_authority=authority, repo_root=tmp_path, scene=scene,
                           platform_manifest=platform, observer=None)
    monkeypatch.setattr(runner_script, "load_shared_v5", lambda *a, **k: pytest.fail("artifact load preceded role rejection"))
    with pytest.raises(SharedV5DevRuntimeConfigurationError, match="not development"):
        runner_script.build_runtime_stack(args)
def test_launcher_constructs_stack_with_verified_physical_calibration(monkeypatch) -> None:
    captured = {}; authority = {"scene": {"scene_id": "dev", "manifest_sha256": _h("scene")}, "observer": None}; binding = {"feature": _h("feature"), "config": _h("config"), "physical_head": _h("physical")}
    physical = {"free_probability_threshold": 0.2, "occupied_probability_threshold": 0.8, "camera_transform_sha256": _h("camera"), "physical_head_state_sha256": binding["physical_head"]}; target = {"target_confirmation": {"minimum_presence_probability": .8, "minimum_quality": .8, "maximum_uncertainty": .5, "maximum_range_m": 4.0}}
    monkeypatch.setattr(runner_script, "load_development_authority", lambda a: (authority, _h("authority"))); monkeypatch.setattr(runner_script, "load_shared_v5", lambda *a, **k: (object(), _h("shared"), _h("g2"), binding)); monkeypatch.setattr(runner_script, "load_target_head", lambda *a, **k: (object(), _h("target"), _h("target-config"), _h("target-state")))
    monkeypatch.setattr(runner_script, "load_qualified_calibration", lambda path, *, role, **k: (target if role == "target_confirmation" else physical, _h(role)))
    pack = SimpleNamespace(scene_id="dev", manifest_sha256=_h("scene"), split="development"); monkeypatch.setitem(__import__("sys").modules, "lewm_genesis.scene_loader", SimpleNamespace(load_scene_pack=lambda *a, **k: pack))
    stack = {name: object() for name in ("backend", "physical_fuser", "physical_memory", "projection", "planner")}; monkeypatch.setitem(__import__("sys").modules, "lewm.navigation.genesis_shared_v5_dev_stack", SimpleNamespace(build_kinematic_development_stack=lambda **k: captured.update(k) or stack))
    monkeypatch.setattr(runner_script, "SharedV5DevMazeRuntime", lambda **k: SimpleNamespace())
    args = SimpleNamespace(device="cpu", repo_root=None, shared_checkpoint=None, g2_report=None, target_head_checkpoint=None, target_calibration=None, physical_calibration=None, g4_head_checkpoint=None, g4_calibration=None, scene=None, platform_manifest=None, genesis_backend="cpu", target_color="red", frontier_cap=1)
    runtime, backend = runner_script.build_runtime_stack(args); assert captured["physical_calibration"] == {"qualified": True, **physical} and backend is stack["backend"] and runtime.development_authority is authority
