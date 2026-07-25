from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_rgb_jepa_encoder_pretraining_v1_source_closure.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_jepa_encoder_pretraining_v1_source_closure",
    CHECKER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load RGB JEPA source-closure checker")
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class SourceClosureTest(unittest.TestCase):
    def test_dynamic_import_edges_are_explicit_roots(self) -> None:
        self.assertTrue(
            checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
                checker.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
            )
        )

    def test_ast_closure_contains_every_required_source(self) -> None:
        sources = checker.discover_source_closure()
        self.assertEqual(tuple(sorted(sources)), sources)
        self.assertTrue(
            set(checker.contract.SOURCE_PATHS).issubset(sources)
        )
        self.assertTrue(
            checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(sources)
        )

    def test_manifest_candidate_validates_under_contract(self) -> None:
        value = checker.build_manifest()
        raw = checker.contract.canonical_json_bytes(value) + b"\n"
        self.assertEqual(
            checker.contract.validate_source_manifest(raw),
            value,
        )
        self.assertEqual(value["generated_input_open_count"], 0)
        self.assertEqual(value["checkpoint_or_tensor_open_count"], 0)
        self.assertEqual(value["sealed_or_heldout_open_count"], 0)


if __name__ == "__main__":
    unittest.main()
