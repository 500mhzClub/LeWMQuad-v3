from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_go2_dynamic_cell_square_projection as preparation
from scripts import diagnose_go2_dynamic_cell_square_projection as runner


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sources() -> list[dict[str, str]]:
    return [
        {
            "role": role,
            "path": path,
            "sha256": (
                runner.DYNAMIC_GEOMETRY_SHA256
                if role == "dynamic_geometry"
                else _sha(role)
            ),
        }
        for role, path in preparation.SOURCE_MAP
    ]


def _shards() -> list[dict]:
    return [
        {
            "path": str(preparation.ROOT / f".synthetic/shard_{index:02d}.npz"),
            "sha256": _sha(f"shard-{index}"),
            "selected_tuples": [],
        }
        for index in range(20)
    ]


def _machine_manifest() -> dict:
    sources, shards = _sources(), _shards()
    ledger = preparation._preparation_ledger(
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    return preparation.assemble_machine_manifest(
        human_manifest_sha256=_sha("human"),
        created_at_utc="2026-07-11T00:00:00+00:00",
        source_entries=sources,
        shard_entries=shards,
        preparation_ledger=ledger,
        output_absence={
            "paths": [
                {
                    "path": str(preparation.ROOT / relative),
                    "exists": False,
                }
                for relative in (
                    preparation.CANDIDATE_RELATIVE_PATH,
                    preparation.FINAL_RESULT_RELATIVE_PATH,
                    preparation.FAILURE_RESULT_RELATIVE_PATH,
                )
            ],
            "all_absent": True,
        },
    )


def _rehash(record: dict) -> None:
    record.pop("content_sha256", None)
    record["content_sha256"] = preparation.canonical_json_sha256(record)


FINALIZER_NUMERIC_PATHS = (
    *(
        ("expected_role_byte_open_counts", role)
        for role in (
            "binding",
            "candidate",
            "diagnostic_test",
            "dynamic_geometry",
            "finalizer",
            "finalizer_test",
            "geometry_test",
            "human_manifest",
            "label_shard",
            "machine_manifest",
            "predecessor_result",
            "preparation_test",
        )
    ),
    *(
        ("expected_label_access", field)
        for field in (
            "label_shard_pre_hash_byte_opens",
            "label_shard_post_hash_byte_opens",
            "label_shard_npz_parses",
            "selected_label_rows_read",
            "metadata_only_shard_stats",
        )
    ),
    (
        "expected_label_access",
        "array_decompression_counts",
        "current_labels",
    ),
    (
        "expected_label_access",
        "array_decompression_counts",
        "next_labels",
    ),
)


def test_source_map_is_exact_nine_role_contract() -> None:
    assert preparation.SOURCE_MAP == (
        ("dynamic_geometry", "lewm/benchmarks/go2_dynamic_cell_square_projection.py"),
        (
            "diagnostic_core",
            "lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py",
        ),
        ("preparation", "scripts/prepare_go2_dynamic_cell_square_projection.py"),
        ("runner", "scripts/diagnose_go2_dynamic_cell_square_projection.py"),
        ("finalizer", "scripts/finalize_go2_dynamic_cell_square_projection.py"),
        ("geometry_test", "lewm/tests/test_go2_dynamic_cell_square_projection.py"),
        (
            "diagnostic_test",
            "lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py",
        ),
        (
            "preparation_test",
            "lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py",
        ),
        (
            "finalizer_test",
            "lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py",
        ),
    )
    assert len({role for role, _ in preparation.SOURCE_MAP}) == 9
    assert len({path for _, path in preparation.SOURCE_MAP}) == 9


def test_execution_bootstraps_have_no_numpy_torch_or_repository_imports() -> None:
    forbidden_prefixes = ("numpy", "torch", "lewm", "scripts")
    for relative in (
        "scripts/prepare_go2_dynamic_cell_square_projection.py",
        "scripts/diagnose_go2_dynamic_cell_square_projection.py",
        "scripts/finalize_go2_dynamic_cell_square_projection.py",
    ):
        tree = ast.parse((preparation.ROOT / relative).read_text())
        top_level_modules = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_modules.append(str(node.module))
        assert not any(
            module.startswith(forbidden_prefixes)
            for module in top_level_modules
        ), (relative, top_level_modules)


def test_runner_and_finalizer_templates_have_only_registered_placeholders() -> None:
    sources, shards = _sources(), _shards()
    runner = preparation._read_template(
        phase="runner",
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    finalizer = preparation._read_template(
        phase="finalizer",
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    runner_placeholders = [item for item in runner if "sha256_source" in item]
    finalizer_placeholders = [item for item in finalizer if "sha256_source" in item]
    assert runner_placeholders == [preparation.SELF_TEMPLATE_ENTRY]
    assert set(map(json.dumps, finalizer_placeholders)) == {
        json.dumps(preparation.SELF_TEMPLATE_ENTRY),
        json.dumps(preparation.CANDIDATE_TEMPLATE_ENTRY),
    }
    assert len([item for item in runner if item["role"] == "label_shard"]) == 20
    assert len([item for item in finalizer if item["role"] == "label_shard"]) == 20


def test_manifest_template_substitution_requires_verified_bytes_and_exact_self() -> None:
    template = [
        {"path": "a", "role": "a", "sha256": _sha("a")},
        dict(preparation.SELF_TEMPLATE_ENTRY),
    ]
    with pytest.raises(ValueError, match="verified before"):
        preparation.instantiate_read_template(
            template,
            verified_manifest_sha256=_sha("manifest"),
            manifest_bytes_verified=False,
        )
    actual = preparation.instantiate_read_template(
        template,
        verified_manifest_sha256=_sha("manifest"),
        manifest_bytes_verified=True,
    )
    assert actual[-1]["sha256"] == _sha("manifest")

    for changed in (
        [*template, dict(preparation.SELF_TEMPLATE_ENTRY)],
        [template[0], {**preparation.SELF_TEMPLATE_ENTRY, "role": "wrong"}],
        [template[0], {"path": preparation.MACHINE_MANIFEST_RELATIVE_PATH, "role": "machine_manifest", "sha256": _sha("literal")}],
    ):
        with pytest.raises(ValueError, match="placeholder"):
            preparation.instantiate_read_template(
                changed,
                verified_manifest_sha256=_sha("manifest"),
                manifest_bytes_verified=True,
            )


def test_phase_contract_freezes_exact_numeric_label_access() -> None:
    template = preparation._read_template(
        phase="runner",
        source_entries=_sources(),
        shard_entries=_shards(),
        human_sha256=_sha("human"),
    )
    contract = preparation._phase_contract(
        phase="runner",
        template=template,
        output_relative_path=preparation.CANDIDATE_RELATIVE_PATH,
    )
    assert contract["authorized_read_path_template_sha256"] == (
        preparation.canonical_json_sha256(template)
    )
    assert contract["expected_role_byte_open_counts"]["label_shard"] == 40
    assert contract["expected_label_access"] == {
        "label_shard_pre_hash_byte_opens": 20,
        "label_shard_post_hash_byte_opens": 20,
        "label_shard_npz_parses": 20,
        "array_decompression_counts": {
            "current_labels": 20,
            "next_labels": 20,
        },
        "selected_label_rows_read": 320,
        "metadata_only_shard_stats": 0,
    }


def test_runner_strict_machine_validator_accepts_exact_contract() -> None:
    manifest = _machine_manifest()
    phase, records = runner._validate_machine_manifest(
        manifest, manifest_sha256=_sha("manifest")
    )
    assert phase is manifest["phase_contracts"]["runner"]
    assert len([record for record in records if record["role"] == "label_shard"]) == 20


@pytest.mark.parametrize(
    "mutation",
    (
        "float_count",
        "bool_count",
        "label_access",
        "source_digest",
        "extra_role",
        "prep_etc_passwd",
        "prep_etc_shadow",
        "prep_source_digest",
        "output_absence_bool",
        "timestamp_bool",
        "timestamp_nonutc",
        "runtime_bool",
        "runtime_float",
        "source_map_count_bool",
        "source_map_count_float",
        "finalizer_source_digest",
        "finalizer_write_path",
        "finalizer_extra_role",
    ),
)
def test_runner_machine_contract_mutations_fail_closed(mutation: str) -> None:
    manifest = _machine_manifest()
    phase = manifest["phase_contracts"]["runner"]
    if mutation == "float_count":
        phase["expected_role_byte_open_counts"]["label_shard"] = 40.0
    elif mutation == "bool_count":
        phase["expected_label_access"]["metadata_only_shard_stats"] = False
    elif mutation == "label_access":
        phase["expected_label_access"]["selected_label_rows_read"] = 319
    elif mutation == "source_digest":
        template = phase["authorized_read_path_template"]
        next(record for record in template if record["role"] == "runner")[
            "sha256"
        ] = _sha("substituted")
        phase["authorized_read_path_template_sha256"] = (
            preparation.canonical_json_sha256(template)
        )
    elif mutation == "extra_role":
        template = phase["authorized_read_path_template"]
        template.append(
            {"path": "extra.py", "role": "extra", "sha256": _sha("extra")}
        )
        template.sort(key=lambda item: (item["path"], item["role"]))
        phase["authorized_read_path_template_sha256"] = (
            preparation.canonical_json_sha256(template)
        )
        phase["expected_roles"].append("extra")
        phase["expected_role_byte_open_counts"]["extra"] = 1
    elif mutation.startswith("prep_etc_"):
        reads = manifest["preparation_access_ledger"]["authorized_read_paths"]
        next(record for record in reads if record["role"] == "binding")["path"] = (
            "/etc/passwd" if mutation.endswith("passwd") else "/etc/shadow"
        )
        reads.sort(key=lambda item: (item["path"], item["role"]))
        manifest["preparation_access_ledger"][
            "authorized_read_path_set_sha256"
        ] = preparation.canonical_json_sha256(reads)
    elif mutation == "prep_source_digest":
        reads = manifest["preparation_access_ledger"]["authorized_read_paths"]
        next(record for record in reads if record["role"] == "runner")[
            "sha256"
        ] = _sha("wrong-preparation-source")
        manifest["preparation_access_ledger"][
            "authorized_read_path_set_sha256"
        ] = preparation.canonical_json_sha256(reads)
    elif mutation == "output_absence_bool":
        manifest["output_absence"]["paths"][0]["exists"] = 0
    elif mutation == "timestamp_bool":
        manifest["created_at_utc"] = True
    elif mutation == "timestamp_nonutc":
        manifest["created_at_utc"] = "2026-07-11T00:00:00"
    elif mutation == "runtime_bool":
        manifest["runtime_environment"]["python_version"][0] = True
    elif mutation == "runtime_float":
        manifest["runtime_environment"]["python_version"][0] = float(
            manifest["runtime_environment"]["python_version"][0]
        )
    elif mutation == "source_map_count_bool":
        manifest["source_map"]["entry_count"] = True
    elif mutation == "source_map_count_float":
        manifest["source_map"]["entry_count"] = 9.0
    elif mutation == "finalizer_source_digest":
        finalizer_phase = manifest["phase_contracts"]["finalizer"]
        template = finalizer_phase["authorized_read_path_template"]
        next(item for item in template if item["role"] == "finalizer")[
            "sha256"
        ] = _sha("wrong-finalizer-source")
        finalizer_phase["authorized_read_path_template_sha256"] = (
            preparation.canonical_json_sha256(template)
        )
    elif mutation == "finalizer_write_path":
        finalizer_phase = manifest["phase_contracts"]["finalizer"]
        finalizer_phase["authorized_write_paths"][0]["path"] = "wrong.json"
        finalizer_phase["authorized_write_paths"].sort(
            key=lambda item: (item["path"], item["role"])
        )
        finalizer_phase["authorized_write_path_set_sha256"] = (
            preparation.canonical_json_sha256(
                finalizer_phase["authorized_write_paths"]
            )
        )
    elif mutation == "finalizer_extra_role":
        finalizer_phase = manifest["phase_contracts"]["finalizer"]
        finalizer_phase["expected_roles"].append("extra")
        finalizer_phase["expected_role_byte_open_counts"]["extra"] = 1
    _rehash(manifest)
    with pytest.raises(ValueError):
        runner._validate_machine_manifest(
            manifest, manifest_sha256=_sha("manifest")
        )


@pytest.mark.parametrize("numeric_path", FINALIZER_NUMERIC_PATHS)
@pytest.mark.parametrize("replacement_type", ("bool", "float"))
def test_runner_rejects_every_nonexact_finalizer_numeric_leaf(
    numeric_path: tuple[str, ...], replacement_type: str
) -> None:
    manifest = _machine_manifest()
    container = manifest["phase_contracts"]["finalizer"]
    for key in numeric_path[:-1]:
        container = container[key]
    original = container[numeric_path[-1]]
    container[numeric_path[-1]] = (
        bool(original) if replacement_type == "bool" else float(original)
    )
    _rehash(manifest)
    with pytest.raises(ValueError, match="finalizer phase"):
        runner._validate_machine_manifest(
            manifest, manifest_sha256=_sha("manifest")
        )


def test_runner_shard_records_must_equal_predecessor_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "path": f"/repo/shard_{index:02d}.npz",
            "sha256": _sha(f"shard-{index}"),
            "selected_tuples": [],
            "selected_row_count": 0,
            "family_side_counts": {},
        }
        for index in range(20)
    ]
    manifest_hash = runner.canonical_json_sha256(entries)
    monkeypatch.setattr(runner, "EXPECTED_LABEL_MANIFEST_SHA256", manifest_hash)
    predecessor = {
        "label_shard_manifest": {
            "entry_count": 20,
            "entries": entries,
            "manifest_sha256": manifest_hash,
        }
    }
    records = sorted(
        [
            {
                "path": entry["path"],
                "role": "label_shard",
                "sha256": entry["sha256"],
            }
            for entry in entries
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    runner.validate_template_shards_against_predecessor(records, predecessor)
    for replacement in (True, 20.0):
        changed_predecessor = copy.deepcopy(predecessor)
        changed_predecessor["label_shard_manifest"]["entry_count"] = replacement
        with pytest.raises(ValueError, match="commitment"):
            runner._manifest_entries(changed_predecessor)
    for field in ("path", "sha256"):
        changed = [dict(item) for item in records]
        changed[0][field] = "wrong" if field == "path" else _sha("wrong")
        with pytest.raises(ValueError, match="differs"):
            runner.validate_template_shards_against_predecessor(
                changed, predecessor
            )


def test_finalizer_phase_has_exact_normal_and_failure_write_paths() -> None:
    template = preparation._read_template(
        phase="finalizer",
        source_entries=_sources(),
        shard_entries=_shards(),
        human_sha256=_sha("human"),
    )
    contract = preparation._phase_contract(
        phase="finalizer",
        template=template,
        output_relative_path=preparation.FINAL_RESULT_RELATIVE_PATH,
    )
    assert contract["authorized_write_paths"] == sorted(
        [
            {
                "path": preparation.FINAL_RESULT_RELATIVE_PATH,
                "role": "finalizer_output",
                "sha256": None,
            },
            {
                "path": preparation.FAILURE_RESULT_RELATIVE_PATH,
                "role": "failure_diagnostic_output",
                "sha256": None,
            },
        ],
        key=lambda item: (item["path"], item["role"]),
    )


def test_metadata_only_preparation_ledger_has_zero_shard_bytes() -> None:
    ledger = preparation._preparation_ledger(
        source_entries=_sources(),
        shard_entries=_shards(),
        human_sha256=_sha("human"),
    )
    assert ledger["metadata_only_shard_stats"] == 20
    assert ledger["role_byte_open_counts"]["label_shard"] == 0
    assert ledger["label_shard_pre_hash_byte_opens"] == 0
    assert ledger["label_shard_post_hash_byte_opens"] == 0
    assert ledger["label_shard_npz_parses"] == 0
    assert ledger["array_decompression_counts"] == {}
    assert not ledger["denied_attempt_records"]
    assert not any(ledger["forbidden_role_open_counts"].values())


@pytest.mark.parametrize(
    "path",
    (
        "docs//synthetic.json",
        "docs/./synthetic.json",
        "docs/other/../synthetic.json",
        "docs/synthetic.json/",
        "/etc/passwd",
        "/etc/shadow",
    ),
)
def test_preparation_rejects_raw_path_aliases_and_outside_paths(path: str) -> None:
    with pytest.raises(ValueError):
        preparation._anchored_absolute(path)


@pytest.mark.parametrize(
    "timestamp",
    (True, 1.0, "2026-07-11T00:00:00", "2026-07-11T00:00:00Z"),
)
def test_preparation_rejects_noncanonical_timestamp_types(timestamp: object) -> None:
    sources, shards = _sources(), _shards()
    ledger = preparation._preparation_ledger(
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    with pytest.raises(ValueError):
        preparation.assemble_machine_manifest(
            human_manifest_sha256=_sha("human"),
            created_at_utc=timestamp,  # type: ignore[arg-type]
            source_entries=sources,
            shard_entries=shards,
            preparation_ledger=ledger,
            output_absence={
                "paths": [
                    {
                        "path": str(preparation.ROOT / relative),
                        "exists": False,
                    }
                    for relative in (
                        preparation.CANDIDATE_RELATIVE_PATH,
                        preparation.FINAL_RESULT_RELATIVE_PATH,
                        preparation.FAILURE_RESULT_RELATIVE_PATH,
                    )
                ],
                "all_absent": True,
            },
        )


@pytest.mark.parametrize("outside", ("/etc/passwd", "/etc/shadow"))
def test_preparation_assembly_rejects_forged_read_graph(outside: str) -> None:
    sources, shards = _sources(), _shards()
    ledger = preparation._preparation_ledger(
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    ledger = copy.deepcopy(ledger)
    ledger["authorized_read_paths"][0]["path"] = outside
    ledger["authorized_read_paths"].sort(
        key=lambda item: (item["path"], item["role"])
    )
    ledger["authorized_read_path_set_sha256"] = preparation.canonical_json_sha256(
        ledger["authorized_read_paths"]
    )
    with pytest.raises(ValueError, match="derived graph"):
        preparation.assemble_machine_manifest(
            human_manifest_sha256=_sha("human"),
            created_at_utc="2026-07-11T00:00:00+00:00",
            source_entries=sources,
            shard_entries=shards,
            preparation_ledger=ledger,
            output_absence=_machine_manifest()["output_absence"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("exists", 0), ("all_absent", 1), ("path", "/tmp/candidate.json")),
)
def test_preparation_rejects_nonexact_output_absence_proof(
    field: str, value: object
) -> None:
    sources, shards = _sources(), _shards()
    ledger = preparation._preparation_ledger(
        source_entries=sources,
        shard_entries=shards,
        human_sha256=_sha("human"),
    )
    absence = copy.deepcopy(_machine_manifest()["output_absence"])
    if field == "all_absent":
        absence[field] = value
    else:
        absence["paths"][0][field] = value
    with pytest.raises(ValueError, match="absence"):
        preparation.assemble_machine_manifest(
            human_manifest_sha256=_sha("human"),
            created_at_utc="2026-07-11T00:00:00+00:00",
            source_entries=sources,
            shard_entries=shards,
            preparation_ledger=ledger,
            output_absence=absence,
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        preparation._strict_json_bytes(b'{"a":1,"a":2}', name="synthetic")
    with pytest.raises(ValueError, match="nonfinite"):
        preparation._strict_json_bytes(b'{"a":Infinity}', name="synthetic")


def test_preparation_cli_has_only_human_manifest_hash() -> None:
    args = preparation._parse_args(["--human-manifest-sha256", _sha("human")])
    assert vars(args) == {"human_manifest_sha256": _sha("human")}
    with pytest.raises(SystemExit):
        preparation._parse_args([])
    with pytest.raises(SystemExit):
        preparation._parse_args(
            ["--human-manifest-sha256", _sha("human"), "--output", "alternate"]
        )


def test_machine_manifest_writer_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    preparation._write_exclusive(output, b"first\n")
    with pytest.raises(FileExistsError):
        preparation._write_exclusive(output, b"second\n")
    assert output.read_bytes() == b"first\n"
