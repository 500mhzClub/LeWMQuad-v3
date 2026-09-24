from __future__ import annotations

import inspect
from pathlib import Path
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - source-only host
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch is unavailable")
class AttentiveReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_go2_utility_scorer_v1_3_attentive_readout_v1 as module
        cls.m = module

    class TinyPooler(nn.Module if nn is not None else object):
        def __init__(self, *, num_queries, embed_dim, **_kwargs):
            super().__init__()
            self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        def forward(self, x):
            return self.query_tokens.expand(len(x), -1, -1) + x.mean(1, keepdim=True)

    def test_architecture_shape_order_and_parameter_count(self):
        model = self.m.FinalLayerAttentiveUtilityScorer(self.TinyPooler)
        latent = torch.randn(2, 4, 768, 1024)
        context = torch.randn(2, 43)
        outputs = model(latent, context)
        self.assertEqual([tuple(value.shape) for value in outputs], [(2,), (2,), (2,)])
        self.assertEqual(tuple(model.horizon_embeddings.shape), (4, 512))
        self.assertFalse(model.horizon_embeddings.requires_grad)

    def test_pinned_official_pooler_parameter_counts(self):
        model = self.m.FinalLayerAttentiveUtilityScorer()
        self.assertEqual(sum(p.numel() for p in model.pooler.parameters()),
                         12_348_416)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         13_684_739)

    def test_fixed_horizon_embedding_is_deterministic(self):
        left = self.m.fixed_horizon_embeddings()
        right = self.m.fixed_horizon_embeddings()
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(left.dtype, torch.float32)
        self.assertEqual(tuple(left.shape), (4, 512))
        self.assertEqual(len(torch.unique(left, dim=0)), 4)

    def test_architecture_keyed_seed_is_stable(self):
        left = self.m.attentive_seed()
        right = self.m.attentive_seed()
        self.assertEqual(left, right)
        self.assertGreaterEqual(left[0], 0)
        self.assertLess(left[0], 2 ** 31)
        self.assertEqual(len(left[1]), 64)

    def test_entire_new_architecture_uses_one_construction_seed(self):
        seed, _digest = self.m.attentive_seed()
        self.m.FROZEN.configure_determinism(seed)
        left = self.m.FinalLayerAttentiveUtilityScorer(self.TinyPooler)
        self.m.FROZEN.configure_determinism(seed)
        right = self.m.FinalLayerAttentiveUtilityScorer(self.TinyPooler)
        self.assertTrue(all(torch.equal(value, right.state_dict()[name])
                            for name, value in left.state_dict().items()))
        frozen = self.m.FROZEN.UtilityScorer(use_latent=True)
        self.assertFalse(torch.equal(
            left.state_dict()["context.0.weight"],
            frozen.state_dict()["context.0.weight"]))

    def test_effective_batch_loss_matches_microbatch_accumulation(self):
        torch.manual_seed(9)
        model_full = nn.Linear(5, 3)
        model_micro = nn.Linear(5, 3)
        model_micro.load_state_dict(model_full.state_dict())
        x = torch.randn(8, 5)
        y = torch.randn(8, 3)
        nn.functional.mse_loss(model_full(x), y, reduction="mean").backward()
        for start in range(0, 8, 2):
            loss = nn.functional.mse_loss(
                model_micro(x[start:start + 2]), y[start:start + 2],
                reduction="sum") / y.numel()
            loss.backward()
        for left, right in zip(model_full.parameters(), model_micro.parameters()):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=1e-7, rtol=1e-6))

    def test_decision_rules_are_predeclared(self):
        strong = self.m.exploratory_decision(
            safety_auc=.8, pairwise_gain=.08, family_consistent=True)
        mixed = self.m.exploratory_decision(
            safety_auc=.8, pairwise_gain=.08, family_consistent=False)
        none = self.m.exploratory_decision(
            safety_auc=.7, pairwise_gain=.04, family_consistent=True)
        self.assertEqual(strong["classification"], "STRONG_READOUT_SIGNAL")
        self.assertEqual(mixed["classification"], "MIXED_READOUT_SIGNAL")
        self.assertEqual(none["classification"], "NO_READOUT_SIGNAL")

    def test_training_precedes_authorisation_and_calibration(self):
        source = inspect.getsource(self.m.run_once)
        self.assertLess(source.index("_load_required_diagnostics("),
                        source.index("FinalLayerAttentiveUtilityScorer()"))
        self.assertLess(source.index("train_once("),
                        source.index("evaluation_auth ="))
        self.assertLess(source.index("evaluation_auth ="),
                        source.index("_evaluate_streaming("))
        training = inspect.getsource(self.m.train_once)
        self.assertIn("TOTAL_UPDATES", training)
        self.assertIn("final_epoch_only_no_selection", training)
        self.assertIn("maximum_attempts\": 1", training)
        self.assertIn("manual_seed(DATA_ORDER_SEED)", training)

    def test_published_terminal_matches_consumption_schema(self):
        run = inspect.getsource(self.m.run_once)
        validate = inspect.getsource(self.m.validate_result_for_consumption)
        for field in (
            '"diagnostic_prerequisites": prerequisites',
            '"official_pooler_binding_digest"',
            '"initialisation"', '"training"',
            '"evaluation_authorisation_digest"',
            '"training_execution_count": 1',
            '"calibration_evaluation_count": 1',
        ):
            self.assertIn(field, run)
        self.assertIn("return validate_result_for_consumption(root=root)", run)
        for lineage in (
            '"attempt_digest"', '"initial_state_digest"',
            '"data_order_witness"', '"diagnostic_prerequisites"',
            '"attentive_result_digest"',
        ):
            self.assertIn(lineage, validate)

    def test_no_predictor_planner_or_package_route(self):
        source = Path(self.m.__file__).read_text()
        self.assertNotIn("load_predictor", source)
        self.assertNotIn("predictor_shard", source)
        self.assertNotIn("scorer_package.pt", source)
        self.assertNotIn("final_eval", source)
        self.assertNotIn("sealed", source.lower())


if __name__ == "__main__":
    unittest.main()
