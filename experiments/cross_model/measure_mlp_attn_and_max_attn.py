#!/usr/bin/env python3
"""Measure MLP/Attn write ratio (Formation) and Max Attn shift pp (Downstream Readout) without Transcoder."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments/cross_model"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from tool_call_vector.run import LOCKED, load_text_dir
from mi4tc.model import _find_decoder_layers


CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3_4b": {
        "formation_layers": [22, 23, 24, 25],
        "commitment_layer": 26,
        "readout_layers": [26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "target_span": "F",
        "key_head": (29, 9),
    },
    "qwen3_8b": {
        "formation_layers": [20, 21, 22, 23],
        "commitment_layer": 24,
        "readout_layers": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "target_span": "F",
        "key_head": (29, 9),
    },
    "qwen3_14b": {
        "formation_layers": [32, 33],
        "commitment_layer": 34,
        "readout_layers": [32, 33, 34, 35, 36, 37, 38, 39],
        "target_span": "F",
        "key_head": (34, 8),
    },
    "qwen35_4b": {
        "formation_layers": [27, 28, 29, 30],
        "commitment_layer": 31,
        "readout_layers": [23, 27, 31],
        "target_span": "F",
        "key_head": (27, 0),
    },
    "qwen35_9b": {
        "formation_layers": [27, 28, 29, 30],
        "commitment_layer": 31,
        "readout_layers": [23, 27, 31],
        "target_span": "F",
        "key_head": (23, 0),
    },
    "granite_3p3_8b": {
        "formation_layers": [31, 32, 33, 34],
        "commitment_layer": 35,
        "readout_layers": [32, 33, 34, 35, 36, 37, 38, 39],
        "target_span": "T",
        "key_head": (34, 25),
    },
    "mistral_3p2_24b": {
        "formation_layers": [22, 23, 24, 25],
        "commitment_layer": 25,
        "readout_layers": [20, 25, 26, 30, 35],
        "target_span": "T",
        "key_head": (20, 19),
    },
}


def load_pairs(model_key: str, spec: dict[str, Any], max_pairs: int) -> list[tuple[str, str]]:
    dataset = spec["dataset"]
    layout = spec["layout"]
    if layout == "text_dir":
        _, held_rows = load_text_dir(dataset, 0, max_pairs)
        return [(clean, corrupt) for _, clean, corrupt in held_rows]
    else:
        manifest = dataset / "manifest.jsonl" if (dataset / "manifest.jsonl").exists() else dataset / "pairs.jsonl"
        items = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split", "heldout") == "heldout":
                    clean = (dataset / row["clean_relpath"]).read_text(encoding="utf-8")
                    corrupt = (dataset / row["corrupt_relpath"]).read_text(encoding="utf-8")
                    items.append((clean, corrupt))
                    if max_pairs > 0 and len(items) >= max_pairs:
                        break
        return items


def encode_text(model_key: str, tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if "mistral" in model_key:
        enc = tokenizer(text, add_special_tokens=False)
        input_ids = enc["input_ids"]
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        offsets = []
        curr = 0
        for t in tokens:
            offsets.append((curr, curr + len(t)))
            curr += len(t)
        return input_ids, offsets
    else:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return enc["input_ids"], enc.get("offset_mapping", [])


def identify_spans(model_key: str, text: str, offsets: list[tuple[int, int]]) -> dict[str, list[int]]:
    spans: dict[str, list[int]] = {"R": [], "T": [], "F": [], "U": []}
    
    if "qwen35" in model_key:
        t_start = text.find("<|start_of_role|>tools")
        if t_start < 0:
            t_start = text.find("<tools>")
        f_start = text.find("For each function call")
        if f_start < 0:
            f_start = text.find("<tool_call>")
        u_start = text.find("<|start_of_role|>user")
        if u_start < 0:
            u_start = text.find("<|im_start|>user")

        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if f_start >= 0 and s >= f_start and (u_start < 0 or s < u_start):
                spans["F"].append(idx)
            elif t_start >= 0 and s >= t_start and (f_start < 0 or s < f_start):
                spans["T"].append(idx)
            elif u_start >= 0 and s >= u_start:
                spans["U"].append(idx)
            else:
                spans["R"].append(idx)

    elif "qwen3" in model_key:
        tools_start = text.find("<tools>\n")
        fmt_start = text.find("For each function call,")
        sys_user_start = text.find("<|im_end|>\n<|im_start|>user\n")
        user_start = sys_user_start + len("<|im_end|>\n<|im_start|>user\n") if sys_user_start >= 0 else -1
        asst_start = text.find("<|im_end|>\n<|im_start|>assistant\n")

        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if fmt_start >= 0 and sys_user_start >= 0 and fmt_start <= s < sys_user_start:
                spans["F"].append(idx)
            elif tools_start >= 0 and fmt_start >= 0 and tools_start <= s < fmt_start:
                spans["T"].append(idx)
            elif user_start >= 0 and asst_start >= 0 and user_start <= s < asst_start:
                spans["U"].append(idx)
            elif tools_start >= 0 and e <= tools_start:
                spans["R"].append(idx)

    elif "granite" in model_key:
        available_open = "<|start_of_role|>available_tools<|end_of_role|>"
        user_open = "<|start_of_role|>user<|end_of_role|>"
        end_tag = "<|end_of_text|>"

        t_start = text.find(available_open)
        t_payload_start = t_start + len(available_open) if t_start >= 0 else 0
        t_end = text.find(end_tag, t_payload_start) if t_start >= 0 else 0
        u_start = text.find(user_open)

        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if t_payload_start <= s < t_end:
                spans["T"].append(idx)
            elif u_start >= 0 and s >= u_start:
                spans["U"].append(idx)
            else:
                spans["R"].append(idx)

    elif "mistral" in model_key:
        tools_open = "[AVAILABLE_TOOLS]"
        tools_close = "[/AVAILABLE_TOOLS]"
        user_open = "[INST]"
        user_close = "[/INST]"

        t_start = text.find(tools_open) + len(tools_open) if text.find(tools_open) >= 0 else 0
        t_end = text.find(tools_close) if text.find(tools_close) >= 0 else 0
        u_start = text.find(user_open) + len(user_open) if text.find(user_open) >= 0 else 0
        u_end = text.find(user_close) if text.find(user_close) >= 0 else len(text)

        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if t_start <= s < t_end:
                spans["T"].append(idx)
            elif u_start <= s < u_end:
                spans["U"].append(idx)
            else:
                spans["R"].append(idx)

    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_key = args.model_key
    device_str = args.device
    cfg = CONFIGS[model_key]
    spec = LOCKED[model_key]

    print(f"[{model_key}] Loading model onto {device_str} with eager attention...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if spec["loader"] == "mistral":
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            spec["path"],
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device_str,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec["path"],
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device_str,
            trust_remote_code=True,
        )
    model.eval()
    layers = _find_decoder_layers(model)
    n_layers = len(layers)

    # Load vector
    vector_path = REPO_ROOT / "results" / "tool_call_vector" / model_key / "directions.pt"
    if not vector_path.exists():
        vector_path = REPO_ROOT / "results" / "transfer" / model_key / "coding_vector.pt"
    if vector_path.exists():
        blob = torch.load(vector_path, map_location="cpu", weights_only=False)
        if "directions" in blob:
            layer = int(spec["layer"])
            coding = blob["directions"].get(layer, list(blob["directions"].values())[0]).float()
        elif "mean_diff" in blob:
            coding = blob["mean_diff"].float()
        else:
            coding = blob.float()
        mu_norm = float(coding.norm().item())
        u_Delta = (coding / mu_norm).to(device_str)
    else:
        raise FileNotFoundError(f"Fit the coding vector before measuring component writes: {vector_path}")

    pairs = load_pairs(model_key, spec, args.max_pairs)
    print(f"[{model_key}] Loaded {len(pairs)} pairs for measurement.", flush=True)

    formation_layers = [l for l in cfg["formation_layers"] if l < n_layers]
    readout_layers = [l for l in cfg["readout_layers"] if l < n_layers]
    target_span = cfg["target_span"]

    mlp_diffs = {l: [] for l in formation_layers}
    attn_diffs = {l: [] for l in formation_layers}
    attn_shifts_per_head: dict[tuple[int, int], list[float]] = {}

    FULL_ATTN_MAP = {23: 5, 27: 6, 31: 7} if "qwen35" in model_key else {}

    print(f"[{model_key}] Measuring formation writes and downstream attention shifts...", flush=True)

    for pair_idx, (clean_text, corrupt_text) in enumerate(pairs):
        c_ids, c_offsets = encode_text(model_key, tokenizer, clean_text)
        k_ids, k_offsets = encode_text(model_key, tokenizer, corrupt_text)

        clean_spans = identify_spans(model_key, clean_text, c_offsets)
        corrupt_spans = identify_spans(model_key, corrupt_text, k_offsets)

        c_target = clean_spans.get(target_span, [])
        k_target = corrupt_spans.get(target_span, [])

        pair_comp_writes = {"clean": {"mlp": {}, "attn": {}}, "corrupt": {"mlp": {}, "attn": {}}}
        pair_head_attns = {"clean": {}, "corrupt": {}}

        for side, ids, target_tokens in [("clean", c_ids, c_target), ("corrupt", k_ids, k_target)]:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device_str)
            handles = []

            for l in formation_layers:
                layer_mod = layers[l]
                mlp_mod = getattr(layer_mod, "mlp", None) or getattr(layer_mod, "feed_forward", None)
                attn_mod = getattr(layer_mod, "self_attn", None) or getattr(layer_mod, "attention", None)

                if mlp_mod is not None:
                    def make_mlp_h(idx):
                        def h(m, a, o):
                            val = o[0] if isinstance(o, tuple) else o
                            proj = (val[0, -1, :].float() @ u_Delta.to(val.device)).item()
                            pair_comp_writes[side]["mlp"][idx] = proj
                        return h
                    handles.append(mlp_mod.register_forward_hook(make_mlp_h(l)))

                if attn_mod is not None:
                    def make_attn_h(idx):
                        def h(m, a, o):
                            val = o[0] if isinstance(o, tuple) else o
                            proj = (val[0, -1, :].float() @ u_Delta.to(val.device)).item()
                            pair_comp_writes[side]["attn"][idx] = proj
                        return h
                    handles.append(attn_mod.register_forward_hook(make_attn_h(l)))

            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
                except Exception:
                    out = model(input_ids=input_ids, use_cache=False)

            for h in handles:
                h.remove()

            if hasattr(out, "attentions") and out.attentions is not None and target_tokens:
                for l in readout_layers:
                    attn_idx = FULL_ATTN_MAP.get(l, l)
                    if attn_idx < len(out.attentions):
                        att_mat = out.attentions[attn_idx]
                        if att_mat is not None and att_mat.ndim == 4:
                            n_heads = att_mat.shape[1]
                            for h in range(n_heads):
                                weight = att_mat[0, h, -1, target_tokens].sum().item()
                                pair_head_attns[side][(l, h)] = weight

        # Accumulate diffs for this pair
        for l in formation_layers:
            m_diff = pair_comp_writes["clean"]["mlp"].get(l, 0.0) - pair_comp_writes["corrupt"]["mlp"].get(l, 0.0)
            a_diff = pair_comp_writes["clean"]["attn"].get(l, 0.0) - pair_comp_writes["corrupt"]["attn"].get(l, 0.0)
            mlp_diffs[l].append(m_diff)
            attn_diffs[l].append(a_diff)

        for key in pair_head_attns["clean"]:
            c_w = pair_head_attns["clean"].get(key, 0.0)
            k_w = pair_head_attns["corrupt"].get(key, 0.0)
            if key not in attn_shifts_per_head:
                attn_shifts_per_head[key] = []
            attn_shifts_per_head[key].append(c_w - k_w)

        if (pair_idx + 1) % 10 == 0 or pair_idx + 1 == len(pairs):
            print(f"[{model_key}] Processed {pair_idx + 1}/{len(pairs)} pairs...", flush=True)

    # 1. MLP/Attn calculation
    tot_mlp = sum(sum(mlp_diffs[l]) / max(1, len(mlp_diffs[l])) for l in formation_layers)
    tot_attn = sum(sum(attn_diffs[l]) / max(1, len(attn_diffs[l])) for l in formation_layers)
    if "qwen35" in model_key:
        mlp_attn_str = "--"
        mlp_attn_ratio = None
    elif abs(tot_attn) > 1e-12:
        mlp_attn_ratio = round(tot_mlp / tot_attn, 2)
        mlp_attn_str = f"{mlp_attn_ratio:.2f}"
    else:
        mlp_attn_ratio = None
        mlp_attn_str = "--"

    # 2. Max Attn shift in pp
    head_shifts = {head: sum(shifts) / len(shifts) * 100.0
                   for head, shifts in attn_shifts_per_head.items() if shifts}
    if not head_shifts:
        raise RuntimeError("Attention matrices are required to compute the target-span shift")
    best_head = max(head_shifts, key=head_shifts.get)
    max_shift_pp = head_shifts[best_head]

    result = {
        "model_key": model_key,
        "formation_layers": formation_layers,
        "tot_mlp_write": round(tot_mlp, 3),
        "tot_attn_write": round(tot_attn, 3),
        "mlp_attn_ratio": mlp_attn_str,
        "max_attn_pp": round(max_shift_pp, 1),
        "best_head": f"L{best_head[0]}H{best_head[1]}",
    }

    out_dir = args.output_dir if args.output_dir is not None else (REPO_ROOT / "results" / "formation_readout" / model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[{model_key}] DONE: MLP/Attn={mlp_attn_str} (tot_mlp={tot_mlp:.2f}, tot_attn={tot_attn:.2f}), Max Attn (pp)=+{max_shift_pp:.1f} (head={result['best_head']})", flush=True)


if __name__ == "__main__":
    main()
