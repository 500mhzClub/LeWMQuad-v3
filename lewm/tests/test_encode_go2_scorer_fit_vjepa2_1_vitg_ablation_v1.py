"""Source-only tests for the separate V-JEPA 2.1 ViT-g latent encoder."""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import encode_go2_scorer_fit_vjepa2_1_vitg_ablation_v1 as ENCODER


def resource_receipt(maximum_passing_batch_size: int = 4) -> dict:
    return {
        "status": "PASS_EXACT_PATH_RESOURCE_SMOKE",
        "encoder_contract_digest": "a" * 64,
        "encoder_contract": {"preprocessing": {"crop": [0, 28, 224, 196]}},
        "source_binding": {"commit": "d" * 40, "files": {}},
        "receipt_sha256": "b" * 64,
        "checkpoint_binding": {
            "path": "/synthetic/vjepa2_1_vitg_384.pt",
            "sha256": "c" * 64,
            "byte_count": 8_000_000_000,
        },
        "inference_dtype": "torch.bfloat16",
        "execution_mode": "bfloat16_autocast_fp32_weights",
        "parameter_dtype": "torch.float32",
        "maximum_passing_batch_size": maximum_passing_batch_size,
    }


def project_source_binding() -> dict:
    return {
        "source_commit": "d" * 40,
        "clean": True,
        "files": {
            path: {"sha256": hashlib.sha256(path.encode()).hexdigest(),
                   "byte_count": len(path)}
            for path in ENCODER.PROJECT_SOURCE_PATHS
        },
    }


def frozen_contract(maximum_passing_batch_size: int = 4) -> dict:
    return ENCODER._signed(ENCODER.encoding_contract(
        resource_receipt(maximum_passing_batch_size),
        project_source_binding=project_source_binding()),
        ENCODER.ENCODING_CONTRACT_SELF_KEY)


def minimal_view(rows: int = ENCODER.EXPECTED_ROWS) -> dict:
    value = {
        "training_view_digest": ENCODER.EXPECTED_TRAINING_VIEW_DIGEST,
        "oracle_v1_3_digest": "1" * 64,
        "scorer_fit_oracle_v1_3_contract_digest": "2" * 64,
        "authority_digest": "3" * 64,
        "rows": [{} for _ in range(rows)],
    }
    for index, key in enumerate(ENCODER.SOURCE_DIGEST_KEYS, start=4):
        value[key] = f"{index:x}" * 64
    return value


def fake_record(name: str, value: str = "d") -> dict:
    return {
        "training_view_row_digest": hashlib.sha256(name.encode()).hexdigest(),
        "state_id": f"state-{name}",
        "state_identity_digest": hashlib.sha256(
            f"state-{name}".encode()).hexdigest(),
        "candidate_index": 0,
        "source_kind": "V2_VALID_ADOPTION",
        "path": f"latents/horizon/{name}.f16",
        "sha256": value * 64,
        "byte_count": ENCODER.SHARD_BYTES,
        "shape": list(ENCODER.HORIZON_SHAPE),
    }


def test_contract_is_separate_exact_shape_and_one_bounded_loader():
    contract = ENCODER.encoding_contract(
        resource_receipt(), project_source_binding=project_source_binding())
    assert ENCODER.GENERATED_ROOT == Path(
        ".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1")
    assert ENCODER.HORIZON_SHAPE == (4, 768, 1408)
    assert ENCODER.SHARD_BYTES == 8_650_752
    assert ENCODER.TOTAL_LATENT_BYTES == 12_457_082_880
    assert contract["training_view_digest"] == (
        "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c")
    assert contract["latent_storage_dtype"] == "float16"
    assert contract["selected_batch_frames"] == 4
    assert contract["execution_mode"] == "bfloat16_autocast_fp32_weights"
    assert contract["parameter_dtype"] == "torch.float32"
    assert contract["loader_workers"] == {
        "default": 4, "minimum": 4, "maximum": 8, "shuffle": False}
    assert contract["minimum_device_free_memory_bytes"] == 26 * (1 << 30)
    assert contract["simulator_runs"] == contract["renders_generated"] == 0
    assert contract["predictor_checkpoints_opened"] == 0


def test_training_view_loader_accepts_only_the_exact_1440_view(monkeypatch):
    view = minimal_view()
    monkeypatch.setattr(ENCODER, "_load_preserved_training_view",
                        lambda root: view)
    monkeypatch.setattr(ENCODER.BASE, "validate_training_view_structure",
                        lambda value: value)
    assert ENCODER.load_training_view(root=Path("/synthetic")) is view

    wrong_count = minimal_view(rows=1_439)
    monkeypatch.setattr(ENCODER, "_load_preserved_training_view",
                        lambda root: wrong_count)
    with pytest.raises(ENCODER.VitGEncodingError, match="1,440"):
        ENCODER.load_training_view(root=Path("/synthetic"))
    wrong_digest = minimal_view()
    wrong_digest["training_view_digest"] = "f" * 64
    monkeypatch.setattr(ENCODER, "_load_preserved_training_view",
                        lambda root: wrong_digest)
    with pytest.raises(ENCODER.VitGEncodingError, match="another training view"):
        ENCODER.load_training_view(root=Path("/synthetic"))


def test_encoding_contract_is_frozen_once_before_runtime(
        tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ENCODER, "_current_project_source_binding",
        lambda root: project_source_binding())
    first = ENCODER.freeze_encoding_contract(
        resource_receipt=resource_receipt(), root=tmp_path)
    repeated = ENCODER.freeze_encoding_contract(
        resource_receipt=resource_receipt(), root=tmp_path)
    path = ENCODER.encoding_contract_path(tmp_path)
    assert first == repeated
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert first[ENCODER.ENCODING_CONTRACT_SELF_KEY] == (
        ENCODER.canonical_digest({
            key: value for key, value in first.items()
            if key != ENCODER.ENCODING_CONTRACT_SELF_KEY
        }))
    with pytest.raises(ENCODER.VitGEncodingError, match="already different"):
        ENCODER.freeze_encoding_contract(
            resource_receipt=resource_receipt(2), root=tmp_path)


def test_atomic_f16_is_missing_only_content_bound_and_read_only(tmp_path: Path):
    path = tmp_path / "one.f16"
    value = np.zeros(ENCODER.HORIZON_SHAPE, dtype=np.float16)
    digest, byte_count = ENCODER.atomic_missing_f16(path, value)
    assert byte_count == ENCODER.SHARD_BYTES == path.stat().st_size
    assert digest == ENCODER.file_sha256(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert ENCODER._adoptable_unindexed_shard(path)
    before = path.read_bytes()
    with pytest.raises(ENCODER.VitGEncodingError, match="refusing to replace"):
        ENCODER.atomic_missing_f16(path, np.ones_like(value))
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))
    os.chmod(path, 0o644)
    assert not ENCODER._adoptable_unindexed_shard(path)


def test_content_digest_is_deterministic_path_and_value_sensitive():
    first = fake_record("first", "d")
    second = fake_record("second", "e")
    observed = ENCODER.latent_content_digest([first, second])
    assert observed == ENCODER.latent_content_digest([first, second])
    changed_value = dict(second, sha256="f" * 64)
    changed_path = dict(second, path="latents/horizon/changed.f16")
    assert observed != ENCODER.latent_content_digest([first, changed_value])
    assert observed != ENCODER.latent_content_digest([first, changed_path])


def test_partial_index_binds_source_preprocess_checkpoint_and_execution():
    record = fake_record("first")
    execution = {
        "loader_workers": 4,
        "resumed_shard_count": 0,
        "adopted_unindexed_shard_count": 0,
        "new_shard_count": 1,
        "invalid_existing_shard_count": 0,
        "encoded_frame_count": 4,
        "wall_seconds": 2.0,
        "new_frames_per_second": 2.0,
        "peak_vram_bytes": 20_000_000_000,
        "peak_process_rss_bytes": 10_000_000_000,
        "peak_child_worker_rss_bytes": 1_000_000_000,
    }
    index = ENCODER._index_payload(
        minimal_view(), [record], resource_receipt=resource_receipt(),
        contract=frozen_contract(), complete=False, execution=execution)
    assert index["horizon_shape"] == [1, 4, 768, 1408]
    assert index["target_encoder_checkpoint_sha256"] == "c" * 64
    assert index["encoding_contract_digest"] == (
        frozen_contract()[ENCODER.ENCODING_CONTRACT_SELF_KEY])
    assert index["selected_batch_frames"] == 4
    assert len(index["encoder_source_binding_digest"]) == 64
    assert len(index["preprocess_contract_digest"]) == 64
    assert index["latent_content_digest"] == (
        ENCODER.latent_content_digest([record]))
    assert index["execution"] == execution
    assert ENCODER._validate_signed(
        index, ENCODER.LATENT_INDEX_SELF_KEY, "synthetic index") == index


def test_monotonic_index_allows_append_but_not_removal_or_mutation():
    first = fake_record("first")
    second = fake_record("second")
    ENCODER._require_monotonic([first], [first, second])
    with pytest.raises(ENCODER.VitGEncodingError, match="monotonically"):
        ENCODER._require_monotonic([first], [])
    with pytest.raises(ENCODER.VitGEncodingError, match="monotonically"):
        ENCODER._require_monotonic(
            [first], [dict(first, sha256="e" * 64)])


def test_managed_root_enforces_free_space_and_rejects_nested_alias(
        tmp_path: Path, monkeypatch):
    logical = tmp_path / ENCODER.GENERATED_ROOT
    logical.mkdir(parents=True)
    monkeypatch.setattr(
        ENCODER.shutil, "disk_usage",
        lambda path: SimpleNamespace(total=100_000_000_000, used=0,
                                     free=ENCODER.MIN_FREE_STORAGE_BYTES))
    assert ENCODER._managed_root(
        tmp_path, require_free_space=True) == logical.absolute()
    monkeypatch.setattr(
        ENCODER.shutil, "disk_usage",
        lambda path: SimpleNamespace(total=100_000_000_000, used=0,
                                     free=ENCODER.MIN_FREE_STORAGE_BYTES - 1))
    with pytest.raises(ENCODER.VitGEncodingError, match="free bytes"):
        ENCODER._managed_root(tmp_path, require_free_space=True)

    nested = logical / "encoded_training_view"
    nested.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ENCODER.VitGEncodingError, match="nested output ancestor"):
        ENCODER._guarded_output(
            "encoded_training_view/latents/horizon/x.f16", root=tmp_path)


def test_loader_pool_is_ordered_bounded_and_after_frame_validation(monkeypatch):
    torch = pytest.importorskip("torch")

    captured = {}

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured.update(kwargs)
            self.dataset = dataset

        def __iter__(self):
            return iter((torch.zeros((1, 4, 3, 8, 8)),))

    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeLoader)
    batches = list(ENCODER._default_preprocessed_batches(
        [["a", "b", "c", "d"]], loader_workers=4))
    assert [tuple(value.shape) for value in batches] == [(4, 3, 8, 8)]
    assert captured["num_workers"] == 4
    assert captured["shuffle"] is False
    assert captured["batch_size"] == 1
    assert captured["worker_init_fn"] is ENCODER._loader_worker_init

    source = inspect.getsource(ENCODER.encode_training_view)
    assert source.index("BASE.validate_frame_inputs") < source.index(
        "_runtime_device")
    assert source.index("freeze_encoding_contract") < source.index(
        "_runtime_device")
    assert "MIN_LOADER_WORKERS <= loader_workers <= MAX_LOADER_WORKERS" in source
    assert "batch_frames is None or batch_frames == selected_batch" in source


@pytest.mark.parametrize(
    ("maximum", "expected"),
    ((1, ((0, 1), (1, 2), (2, 3), (3, 4))),
     (2, ((0, 2), (2, 4))),
     (4, ((0, 4),))),
)
def test_smoke_selected_batch_chunks_each_fixed_horizon_row(maximum, expected):
    receipt = resource_receipt(maximum)
    assert ENCODER.selected_batch_frames(receipt) == maximum
    assert ENCODER.encoding_contract(
        receipt, project_source_binding=project_source_binding()
    )["selected_batch_frames"] == maximum
    assert ENCODER._frame_chunk_ranges(maximum) == expected


@pytest.mark.parametrize("maximum", (1, 2, 4))
def test_default_encoder_uses_fp32_inputs_and_selected_chunks(
        maximum, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

    observed = []

    def fake_extract(_arm, batch):
        assert batch.dtype == torch.float32
        observed.append(int(batch.shape[0]))
        return torch.zeros(
            (batch.shape[0], ENCODER.TOKENS, ENCODER.TOKEN_DIM),
            dtype=torch.bfloat16, device=batch.device)

    monkeypatch.setattr(runtime, "extract_final_dense_tokens_v1", fake_extract)
    output = ENCODER._default_encode_pixels(
        object(), torch.zeros((4, 3, 2, 2)), torch.device("cpu"),
        torch.bfloat16, maximum)
    assert observed == [stop - start for start, stop in
                        ENCODER._frame_chunk_ranges(maximum)]
    assert output.shape == ENCODER.HORIZON_SHAPE
    assert output.dtype == np.float16


def test_public_loader_contract_and_no_forbidden_runtime_route():
    signature = inspect.signature(
        ENCODER.load_and_validate_encoded_training_view_for_consumption)
    assert tuple(signature.parameters) == ("root", "verify_encoder_checkpoint")
    source = Path(ENCODER.__file__).read_text()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports
    assert "simulator" not in imports
    assert "renderer" not in imports
    assert "sealed" not in source.lower()
    assert ENCODER.latent_index_path().name == "latent_index.json"
    assert ENCODER.encoding_receipt_path().name == "encoding_receipt.json"
    assert set(ENCODER.RECORD_KEYS) == {
        "training_view_row_digest", "state_id", "state_identity_digest",
        "candidate_index", "source_kind", "path", "sha256", "byte_count",
        "shape",
    }
