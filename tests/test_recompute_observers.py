"""Validate scientific observations against independent model outputs."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/cross_model"))
import recompute_mechanisms as measurements


class ObserverParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8)
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config).eval()
        self.sequences = [[5, 6, 7, 8], [2, 9, 10, 11]]
        self.ids = torch.tensor(self.sequences)
        self.ctx = SimpleNamespace(
            key="observer_test", spec={"marker_id": 3},
            adapter=SimpleNamespace(model=self.model, layers=list(self.model.model.layers), tokenizer=None),
            pairs=SimpleNamespace(heldout=[SimpleNamespace(clean_text="", corrupt_text="")] * 2),
            direction=torch.ones(32), unit=torch.ones(32) / 32**0.5)

    def test_attention_matches_eager_without_changing_logits(self):
        with torch.no_grad():
            reference = self.model(self.ids, use_cache=False, logits_to_keep=1).logits
        spans = {"R": [0], "T": [1], "F": [2], "U": [3], "history": []}
        with patch.dict(measurements.CONFIGS, {"observer_test": {"readout_layers": [0, 1]}}), \
                patch.object(measurements, "token_regions", return_value=spans):
            observer = measurements.ReadoutObserver(self.ctx, self.sequences, "clean")
            try:
                observer.begin([0, 1], torch.tensor([3, 3]), 4)
                with torch.no_grad():
                    actual = self.model(self.ids, use_cache=False, logits_to_keep=1).logits
                observer.end()
            finally:
                observer.close()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.model.set_attn_implementation("eager")
        with torch.no_grad():
            eager = self.model(self.ids, use_cache=False, output_attentions=True, logits_to_keep=1)
        for layer, weights in enumerate(eager.attentions):
            torch.testing.assert_close(observer.attention[layer][:, :, :4], weights[:, :, -1].float(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(observer.attention[layer].sum(-1), torch.ones(2, 4), rtol=1e-6, atol=1e-6)
            self.assertTrue(torch.isfinite(observer.dla(layer)).all())

    def test_formation_writes_account_for_residual_change(self):
        observer = measurements.FormationObserver(self.ctx, 2, [0, 1])
        try:
            observer.begin([0, 1], torch.tensor([3, 3]), 4)
            with torch.no_grad():
                self.model(self.ids, use_cache=False, logits_to_keep=1)
            observer.end()
        finally:
            observer.close()
        for layer in (0, 1):
            torch.testing.assert_close(
                observer.projections["post"][:, layer] - observer.projections["pre"][:, layer],
                observer.projections["mlp"][:, layer] + observer.projections["attn"][:, layer],
                rtol=1e-4, atol=1e-6)

    def test_pre_layer_is_previous_block_output(self):
        from tool_call_vector.run import LayerSweep
        adapter = SimpleNamespace(model=self.model, layers=list(self.model.model.layers),
                                  tokenizer=None, device=torch.device("cpu"))
        pre = LayerSweep(adapter, 3, [1], 4096, hook="pre", last_token_only=True).capture(self.sequences)
        post = LayerSweep(adapter, 3, [0], 4096, hook="post", last_token_only=True).capture(self.sequences)
        torch.testing.assert_close(pre["states"][1], post["states"][0], rtol=0, atol=0)


class TranscoderQuality(unittest.TestCase):
    def test_ve_matches_training_global_variance(self):
        tc = measurements.Transcoder.__new__(measurements.Transcoder)
        tc.enc = torch.eye(2)
        tc.enc_bias = torch.zeros(2)
        tc.dec = torch.zeros(2, 2)
        tc.dec_bias = torch.tensor([10.0, 100.0])
        tc.features, tc.topk, tc.topk_input = 2, 0, "relu"
        x = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
        y = torch.tensor([[9.0, 100.0], [11.0, 102.0]])
        means, quality = tc.measure(x, y)
        predicted = tc.dec_bias.expand_as(y)
        mse = (predicted-y).square().mean()
        expected = 1.0-float(mse/y.var(unbiased=False))
        self.assertAlmostEqual(quality["variance_explained"], expected, places=7)
        self.assertAlmostEqual(quality["variance_explained_per_channel"], -0.5)
        self.assertEqual(quality["mean_active_features"], 1.5)
        torch.testing.assert_close(means, torch.tensor([1.5, 1.5]))


if __name__ == "__main__":
    unittest.main()
