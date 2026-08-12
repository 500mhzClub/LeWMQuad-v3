"""Pure provenance-order tests for the scorer-fit latent encoder."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import encode_go2_branch_corpus_v1_2 as encoder


class BranchCorpusEncoderProvenanceTests(unittest.TestCase):
    def test_live_selection_replay_failure_precedes_rows_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "scorer_fit"
            out.mkdir()
            manifest = {"pool": "scorer_fit", "states": []}
            (out / "state_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_active_state_manifest_for_consumption",
                    side_effect=RuntimeError(
                        "later replacement capture prefix is not canonical"
                    )) as replay, \
                    mock.patch.object(
                        encoder, "_load_selector_successor_receipts"
                    ) as selector, \
                    mock.patch.object(encoder, "_verify_frames") as frames, \
                    mock.patch.object(encoder.json, "loads") as raw_json:
                with self.assertRaisesRegex(
                        RuntimeError, "later replacement capture prefix"):
                    encoder._load_inputs(
                        out, allow_partial=False, pool="scorer_fit")
            replay.assert_called_once_with(
                out / "state_manifest.json", pool="scorer_fit")
            selector.assert_not_called()
            frames.assert_not_called()
            raw_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
