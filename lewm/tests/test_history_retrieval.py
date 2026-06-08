from __future__ import annotations

import unittest

import numpy as np

from scripts.probe_lewm_history_retrieval import history_descriptors


class HistoryRetrievalTests(unittest.TestCase):
    def test_history_descriptors_use_terminal_suffixes(self) -> None:
        history = np.arange(2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3)
        descriptors = history_descriptors(history, (4, 8))

        np.testing.assert_array_equal(descriptors["terminal_raw"], history[:, -1])
        np.testing.assert_array_equal(descriptors["h4_mean"], history[:, -4:].mean(axis=1))
        np.testing.assert_array_equal(descriptors["h4_concat"], history[:, -4:].reshape(2, -1))
        self.assertEqual(descriptors["h8_concat"].shape, (2, 24))


if __name__ == "__main__":
    unittest.main()

