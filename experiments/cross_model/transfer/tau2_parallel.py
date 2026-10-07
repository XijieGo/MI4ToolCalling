#!/usr/bin/env python3
"""Score independent Tau2 call/text arms, with resumable measured predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import run as transfer


ALPHAS = (1.0, 1.5, 3.0)
VARIANTS = ("1.0", "1.5", "3.0", "random_1.0")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_part(path: Path, part: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(part, temporary)
    temporary.replace(path)


def context(model: str, run_dir: Path, *, require_transfer: bool = True, token_budget: int = 0):
    if model == "qwen3_8b":
        raise ValueError("Use transfer/run.py for Qwen3-8B tau2 evaluation")
    module = transfer.load_vector_module()
    spec = dict(module.LOCKED[model], model_key=model)
    fixed = json.loads((run_dir / "fixed_layers.json").read_text())[model]
    if fixed != {"layer": spec["layer"], "hook": spec["hook"]}:
        raise ValueError("Locked layer/hook does not match this rerun")
    destination = transfer.REPO_ROOT / "results/transfer" / model
    blob = torch.load(transfer.REPO_ROOT / "results/tool_call_vector" / model / "directions.pt",
                      map_location="cpu", weights_only=False)
    fresh = blob["directions"][spec["layer"]].float().flatten()
    if blob["hook"] != spec["hook"]:
        raise ValueError("Fresh vector hook must match the fixed layer")
    if require_transfer:
        report, coding = transfer.load_checkpoint(destination, spec["layer"], spec["hook"])
        if report is None or not {"multi_domain", "verb_free"} <= report.keys():
            raise ValueError("Completed multi-domain and verb-free checkpoints are required")
        if not torch.equal(coding, fresh):
            raise ValueError("Transfer vector must equal the newly fitted fixed-layer vector")
        if token_budget and int(report["token_budget"]) != token_budget:
            raise ValueError("Tau2 token budget differs from the transfer checkpoint")
        spec["token_budget"] = int(report["token_budget"])
    else:
        # Independent native call/text arms only need the already fitted
        # vector. They may run alongside the domain/verb-free worker without
        # reading or writing its in-progress checkpoint. Merge still requires
        # both completed transfer datasets and identical vector provenance.
        coding, report = fresh, None
        if token_budget:
            spec["token_budget"] = token_budget
    provenance = {"model_key": model, "layer": spec["layer"], "hook": spec["hook"],
                  "coding_sha256": hashlib.sha256(coding.numpy().tobytes()).hexdigest(),
                  "token_budget": spec["token_budget"]}
    return module, spec, destination, report, coding, provenance


def part_path(run_dir: Path, model: str, side: str, index: int, count: int) -> Path:
    return run_dir / "parallel_tau2" / model / f"{side}_{index}_of_{count}.pt"


def run_side(model: str, side: str, run_dir: Path, index: int, count: int, token_budget: int = 0) -> None:
    if not 0 <= index < count:
        raise ValueError("Shard index out of bounds")
    module, spec, destination, report, coding, provenance = context(
        model, run_dir, require_transfer=not bool(token_budget), token_budget=token_budget)
    source = transfer.REPO_ROOT / "datasets" / model / "tau2_bench" / f"native-{side}-200.jsonl"
    rows = transfer.read_jsonl(source)
    if len(rows) != 200:
        raise ValueError(f"Expected the full 200-row Tau2 {side} arm")
    start, stop = index*(200//count), (index+1)*(200//count)
    rows = rows[start:stop]
    domain = transfer.tau2_domain(rows)
    family = transfer.FAMILY[model]
    provenance.update(side=side, source_sha256=digest(source), domain=domain, family=family,
                      template=str(transfer.TAU2_RAW[domain]),
                      system_sha256=digest(transfer.TAU2_RAW[domain]/"tau2_system_prompt.txt"),
                      tools_sha256=digest(transfer.TAU2_RAW[domain]/"tau2_tool_schemas.json"),
                      sample_start=start, sample_stop=stop, shard_index=index, shard_count=count)
    path = part_path(run_dir, model, side, index, count)
    if path.exists():
        part = torch.load(path, map_location="cpu", weights_only=False)
        if part["provenance"] != provenance:
            raise ValueError("Tau2 checkpoint inputs or fixed vector changed")
    else:
        part = {"provenance": provenance, "baseline_parts": [], "variant_parts": {}, "n": len(rows)}
    n_chunks = len(rows)//25
    if all(len(part["variant_parts"].get(key, [])) == n_chunks for key in VARIANTS):
        print(f"MODEL_DONE {model} side={side} resumed=complete", flush=True)
        return
    started = time.time()
    print(f"load {model} side={side} L{spec['layer']} {spec['hook']}", flush=True)
    adapter, loader = module.load_adapter(Path(spec["path"]), spec["loader"], "cuda:0")
    module.check_marker(adapter, spec["marker"], spec["marker_id"])
    ids = transfer.render_tau2(adapter.tokenizer, rows, domain, family)
    encoded_digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if part.get("rendered_sha256", encoded_digest) != encoded_digest:
        raise ValueError("Native Tau2 rendering changed since the checkpoint")
    part.update(rendered_sha256=encoded_digest, loader=loader)
    sweep = module.LayerSweep(adapter, spec["marker_id"], [spec["layer"]],
                              spec["token_budget"], hook=spec["hook"], last_token_only=True)
    chunks = [ids[start:start+25] for start in range(0, len(ids), 25)]
    for i in range(len(part["baseline_parts"]), len(chunks)):
        part["baseline_parts"].append(sweep.capture(chunks[i]))
        save_part(path, part)
        print(f"{side} shard={index} baseline {(i+1)*25}/{len(rows)}", flush=True)
    random_direction = transfer.orthogonal_match(coding, transfer.RANDOM_SEED)
    sign = -1 if side == "call" else 1
    for key in VARIANTS:
        delta = sign * (random_direction if key.startswith("random") else float(key) * coding)
        scored = part["variant_parts"].setdefault(key, [])
        for i in range(len(scored), len(chunks)):
            baseline = part["baseline_parts"][i]
            replacement = baseline["states"][spec["layer"]] + delta
            scored.append(sweep.intervene(chunks[i], spec["layer"], replacement))
            save_part(path, part)
            print(f"{side} shard={index} {key} {(i+1)*25}/{len(rows)}", flush=True)
    part["elapsed_sec"] = part.get("elapsed_sec", 0) + time.time() - started
    save_part(path, part)
    print(f"MODEL_DONE {model} side={side} elapsed={part['elapsed_sec']:.1f}", flush=True)


def side_metrics(part: dict, key: str) -> dict:
    base = part["baseline_parts"]
    after = part["variant_parts"][key]
    return transfer.arm_metrics(torch.cat([p["tool_top1"] for p in base]),
                                torch.cat([p["tool_top1"] for p in after]),
                                torch.cat([p["tool_logit"] for p in base]),
                                torch.cat([p["tool_logit"] for p in after]))


def combine(call: dict, text: dict) -> tuple[dict, dict]:
    for key in ("model_key", "layer", "hook", "coding_sha256", "token_budget", "domain", "family",
                "template", "system_sha256", "tools_sha256"):
        if call["provenance"][key] != text["provenance"][key]:
            raise ValueError(f"Tau2 sides disagree on {key}")
    if call["provenance"]["side"] != "call" or text["provenance"]["side"] != "text":
        raise ValueError("Expected separate call and text arms")
    strength = {}
    for key in VARIANTS:
        removal, induction = side_metrics(call, key), side_metrics(text, key)
        sup, ind = removal["suppression_among_calls"], induction["induction_among_quiet"]
        strength[key] = {"removal": removal, "induction": induction,
                         "score": None if sup is None or ind is None else 0.5*(sup+ind)}
    p = call["provenance"]
    result = {"domain": p["domain"], "family": p["family"], "template": p["template"],
              "alpha": 1.0, "forward": "left_pad_last_token", **strength["1.0"],
              "random_removal": strength["random_1.0"]["removal"],
              "random_induction": strength["random_1.0"]["induction"],
              "strength": {key: strength[key] for key in ("1.5", "3.0")},
              "parallel_arms": True,
              "source_sha256": {"call": p["source_sha256"], "text": text["provenance"]["source_sha256"]}}
    return result, {key: strength[key] for key in ("1.0", "1.5", "3.0")}


def merge(model: str, run_dir: Path, count: int) -> None:
    module, spec, destination, report, coding, provenance = context(model, run_dir)
    parts = []
    for side in ("call", "text"):
        shards = []
        for index in range(count):
            part = torch.load(part_path(run_dir,model,side,index,count), map_location="cpu", weights_only=False)
            expected = dict(provenance, side=side, shard_index=index, shard_count=count,
                            sample_start=index*(200//count), sample_stop=(index+1)*(200//count))
            for key, value in expected.items():
                if part["provenance"][key] != value:
                    raise ValueError(f"Tau2 {side} provenance mismatch on {key}")
            source = transfer.REPO_ROOT/"datasets"/model/"tau2_bench"/f"native-{side}-200.jsonl"
            if part["provenance"]["source_sha256"] != digest(source):
                raise ValueError("Canonical Tau2 source changed")
            if part["n"] != 200//count or len(part["baseline_parts"]) != 8//count:
                raise ValueError("Incomplete Tau2 baseline")
            if any(len(part["variant_parts"].get(key, [])) != 8//count for key in VARIANTS):
                raise ValueError("Incomplete Tau2 intervention arms")
            if any(sum(len(p["tool_top1"]) for p in arm) != 200//count for arm in
                   [part["baseline_parts"], *[part["variant_parts"][key] for key in VARIANTS]]):
                raise ValueError("Tau2 predictions do not cover the entire shard")
            shards.append(part)
        parts.append({"provenance": shards[0]["provenance"],
                      "baseline_parts": [p for shard in shards for p in shard["baseline_parts"]],
                      "variant_parts": {key: [p for shard in shards for p in shard["variant_parts"][key]]
                                        for key in VARIANTS},
                      "elapsed_sec": max(p.get("elapsed_sec",0) for p in shards)})
    result, strength = combine(*parts)
    alpha_report = {"model_key": model, "layer": spec["layer"], "hook": spec["hook"],
                    "coding_norm": float(coding.norm()), "domain": result["domain"],
                    "forward": "left_pad_last_token", "token_budget": spec["token_budget"],
                    "alphas": strength, "parallel_arms": True}
    (destination/"tau2_alphas.json").write_text(json.dumps(alpha_report, indent=2)+"\n")
    report["tau2"] = result
    report["tau2_parallel_elapsed_sec"] = max(p.get("elapsed_sec", 0) for p in parts)
    transfer.checkpoint(destination, report, coding, spec["layer"], spec["hook"])
    print(f"MODEL_DONE {model} merged=call+text n=200+200", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("call", "text"))
    parser.add_argument("--shard-count", type=int, choices=(1,2,4,8), default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--token-budget", type=int, default=0)
    args = parser.parse_args()
    if args.merge:
        merge(args.model_key, args.run_dir.resolve(), args.shard_count)
    elif args.side:
        run_side(args.model_key, args.side, args.run_dir.resolve(), args.shard_index, args.shard_count, args.token_budget)
    else:
        parser.error("Provide --side or --merge")


if __name__ == "__main__":
    main()
