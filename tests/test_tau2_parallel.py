"""Parallel Tau2 arms preserve serial interventions and conditional metrics."""
import sys
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments/cross_model/transfer"))
import tau2_parallel as parallel


class ParallelTau2(unittest.TestCase):
    def test_independent_arm_can_start_before_domain_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/"run"
            run.mkdir()
            (run/"fixed_layers.json").write_text(json.dumps({"test": {"layer": 1, "hook": "pre"}}))
            vector = root/"results/tool_call_vector/test"
            vector.mkdir(parents=True)
            torch.save({"directions": {1: torch.arange(4).float()}, "hook": "pre"}, vector/"directions.pt")
            module = SimpleNamespace(LOCKED={"test": {"layer": 1, "hook": "pre", "token_budget": 99}})
            with patch.object(parallel.transfer, "REPO_ROOT", root), \
                 patch.object(parallel.transfer, "load_vector_module", return_value=module):
                _, spec, _, report, coding, provenance = parallel.context(
                    "test", run, require_transfer=False, token_budget=4)
                self.assertIsNone(report)
                self.assertEqual(spec["token_budget"], 4)
                torch.testing.assert_close(coding, torch.arange(4).float())
                with self.assertRaisesRegex(ValueError, "Completed multi-domain"):
                    parallel.context("test", run)
                (run/"fixed_layers.json").write_text(json.dumps({"test": {"layer": 1, "hook": "post"}}))
                with self.assertRaisesRegex(ValueError, "Locked layer/hook"):
                    parallel.context("test", run, require_transfer=False, token_budget=4)

    def test_chunked_scoring_equals_serial_fixed_layer(self):
        torch.manual_seed(18)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                              num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=8, pad_token_id=0)).eval()
        adapter = SimpleNamespace(model=model, layers=list(model.model.layers),
                                  tokenizer=None, device=torch.device("cpu"))
        module = parallel.transfer.load_vector_module()
        sweep = module.LayerSweep(adapter, 3, [1], 2, hook="pre", last_token_only=True)
        sequences = [[2, 3, 4], [2, 4, 5, 6], [3, 4, 7], [2, 8, 9, 10]]
        coding = torch.randn(16)
        serial_base = sweep.capture(sequences)
        chunk_base = [sweep.capture(sequences[:2]), sweep.capture(sequences[2:])]
        part = {"baseline_parts": chunk_base, "variant_parts": {}}
        for key in parallel.VARIANTS:
            direction = parallel.transfer.orthogonal_match(coding, parallel.transfer.RANDOM_SEED) if key.startswith("random") else float(key)*coding
            serial = parallel.transfer.run_delta_arm(sweep, 1, sequences, serial_base, -direction)
            part["variant_parts"][key] = [sweep.intervene(chunk, 1, base["states"][1]-direction)
                                          for chunk, base in zip((sequences[:2], sequences[2:]), chunk_base)]
            self.assertEqual(parallel.side_metrics(part, key), serial)

    def test_combination_uses_each_sides_baseline_denominator(self):
        provenance = dict(model_key="test", layer=1, hook="post", coding_sha256="vector",
                          token_budget=2, domain="telecom", family="test", template="test",
                          system_sha256="system", tools_sha256="tools", source_sha256="source")
        def part(side, before, after):
            base = {"tool_top1": torch.tensor(before).bool(), "tool_logit": torch.ones(len(before))}
            scored = {"tool_top1": torch.tensor(after).bool(), "tool_logit": torch.zeros(len(after))}
            return {"provenance": dict(provenance, side=side), "baseline_parts": [base],
                    "variant_parts": {key: [scored] for key in parallel.VARIANTS}}
        call = part("call", [1,1,0], [0,1,1])
        text = part("text", [0,0,1,0], [1,0,1,1])
        result, strengths = parallel.combine(call, text)
        self.assertEqual(result["removal"]["n_baseline_call"], 2)
        self.assertEqual(result["induction"]["n_baseline_quiet"], 3)
        self.assertEqual(result["removal"]["suppression_among_calls"], 0.5)
        self.assertAlmostEqual(result["induction"]["induction_among_quiet"], 2/3)
        self.assertAlmostEqual(strengths["1.0"]["score"], (0.5+2/3)/2)
        text["provenance"]["layer"] = 2
        with self.assertRaisesRegex(ValueError, "layer"):
            parallel.combine(call, text)

    def test_merging_shards_covers_full_arms_and_rejects_duplicate_ranges(self):
        provenance = dict(model_key="test", layer=1, hook="post", coding_sha256="vector", token_budget=2)
        shards = {}
        for side in ("call", "text"):
            for index in (0,1):
                base = {"tool_top1": torch.tensor([0,1]*12+[0]).bool(), "tool_logit": torch.ones(25)}
                scored = {"tool_top1": torch.full((25,), side=="text"), "tool_logit": torch.zeros(25)}
                shards[f"{side}_{index}_of_2.pt"] = {
                    "provenance": dict(provenance,side=side,shard_index=index,shard_count=2,
                                       sample_start=index*100,sample_stop=(index+1)*100,
                                       source_sha256="source",domain="telecom",family="test",template="test",
                                       system_sha256="system",tools_sha256="tools"),
                    "n":100,"baseline_parts":[base]*4,
                    "variant_parts":{key:[scored]*4 for key in parallel.VARIANTS}}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            report = {"multi_domain": {"completed": True}, "verb_free": {"completed": True}}
            context = (None,dict(layer=1,hook="post",token_budget=2),destination,report,torch.ones(4),provenance)
            with patch.object(parallel,"context",return_value=context), \
                 patch.object(parallel,"digest",return_value="source"), \
                 patch.object(parallel.torch,"load",side_effect=lambda path,**kw: shards[path.name]), \
                 patch.object(parallel.transfer,"checkpoint") as checkpoint:
                parallel.merge("test",destination,2)
                self.assertEqual(report["tau2"]["removal"]["n"],200)
                self.assertEqual(report["tau2"]["induction"]["n"],200)
                self.assertEqual(report["tau2"]["score"],1.0)
                self.assertTrue(report["multi_domain"]["completed"])
                checkpoint.assert_called_once()
                shards["call_1_of_2.pt"]["provenance"]["sample_start"] = 0
                with self.assertRaisesRegex(ValueError,"sample_start"):
                    parallel.merge("test",destination,2)


if __name__ == "__main__":
    unittest.main()
