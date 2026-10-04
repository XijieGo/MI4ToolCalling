"""Tau2 strength sweep on a saved coding vector.

Loads the locked vector already written by the transfer run. Does not refit it.
Scores the call arm at -alpha * mu and the text arm at +alpha * mu. The forward
left-pads and projects only the last real token.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run as transfer


ALPHAS = (1.5, 3.0)
RANDOM_ALPHAS = ()
TOKEN_BUDGET = 24576


def load_vector(model_key: str, output_root: Path) -> tuple[torch.Tensor, int, str]:
    path = output_root / model_key / "coding_vector.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vector = torch.as_tensor(blob["mean_diff"], dtype=torch.float32).flatten().contiguous()
    return vector, int(blob["layer"]), str(blob["hook"])


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def alpha_key(alpha: float) -> str:
    return str(float(alpha))


def sweep_model(
    model_key: str,
    output_root: Path,
    device: str,
    alphas: tuple[float, ...],
    random_alphas: tuple[float, ...],
    token_budget: int,
) -> dict:
    destination = output_root / model_key / "tau2_alphas.json"
    existing: dict = {}
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
    saved = existing.get("alphas") or {}
    pending = [alpha for alpha in alphas if alpha_key(alpha) not in saved]
    pending_random = [alpha for alpha in random_alphas if f"random_{alpha:g}" not in saved]
    if not pending and not pending_random:
        print(f"SKIP {model_key} alphas already saved", flush=True)
        print(f"MODEL_DONE {model_key}", flush=True)
        return existing
    module = transfer.load_vector_module()
    spec = dict(module.LOCKED[model_key])
    spec["model_key"] = model_key
    coding, layer, hook = load_vector(model_key, output_root)
    if layer != int(spec["layer"]) or hook != spec["hook"]:
        raise RuntimeError(f"Saved vector is L{layer} {hook}, locked spec is L{spec['layer']} {spec['hook']}")
    print(f"load {model_key} norm={float(coding.norm()):.3f} L{layer} {hook}", flush=True)
    adapter, loader_name = module.load_adapter(Path(spec["path"]), spec["loader"], device)
    module.check_marker(adapter, spec["marker"], spec["marker_id"])
    root = transfer.REPO_ROOT / "datasets" / model_key / "tau2_bench"
    call_rows = transfer.read_jsonl(root / "native-call-200.jsonl")
    text_rows = transfer.read_jsonl(root / "native-text-200.jsonl")
    domain = transfer.tau2_domain(call_rows)
    if transfer.tau2_domain(text_rows) != domain:
        raise RuntimeError(f"Tau2 arms disagree for {model_key}")
    family = transfer.FAMILY[model_key]
    call_ids = transfer.render_tau2(adapter.tokenizer, call_rows, domain, family)
    text_ids = transfer.render_tau2(adapter.tokenizer, text_rows, domain, family)
    print(f"tau2 {domain} call {transfer.length_stats(call_ids)} text {transfer.length_stats(text_ids)}", flush=True)
    sweep = module.LayerSweep(
        adapter,
        spec["marker_id"],
        [layer],
        token_budget,
        hook=hook,
        last_token_only=True,
    )
    call_base = sweep.capture(call_ids)
    text_base = sweep.capture(text_ids)
    print(
        f"baseline call {int(call_base['tool_top1'].sum())}/{len(call_ids)} "
        f"text {int(text_base['tool_top1'].sum())}/{len(text_ids)}",
        flush=True,
    )
    report: dict = {
        "model_key": model_key,
        "layer": layer,
        "hook": hook,
        "loader": loader_name,
        "coding_norm": float(coding.norm().item()),
        "domain": domain,
        "forward": "left_pad_last_token",
        "token_budget": token_budget,
        "alphas": dict(saved),
    }
    for alpha in pending:
        removal = transfer.run_delta_arm(sweep, layer, call_ids, call_base, -alpha * coding)
        induction = transfer.run_delta_arm(sweep, layer, text_ids, text_base, alpha * coding)
        score = None
        if removal["suppression_among_calls"] is not None and induction["induction_among_quiet"] is not None:
            score = 0.5 * (removal["suppression_among_calls"] + induction["induction_among_quiet"])
        report["alphas"][str(alpha)] = {"removal": removal, "induction": induction, "score": score}
        print(
            f"alpha {alpha:g}: sup {pct(removal['suppression_among_calls'])} "
            f"ind {pct(induction['induction_among_quiet'])} score {pct(score)}",
            flush=True,
        )
        write_report(destination, report)
    if pending_random:
        random_direction = transfer.orthogonal_match(coding, transfer.RANDOM_SEED)
    for alpha in pending_random:
        removal = transfer.run_delta_arm(sweep, layer, call_ids, call_base, -alpha * random_direction)
        induction = transfer.run_delta_arm(sweep, layer, text_ids, text_base, alpha * random_direction)
        report["alphas"][f"random_{alpha:g}"] = {"removal": removal, "induction": induction}
        print(
            f"random {alpha:g}: sup {pct(removal['suppression_among_calls'])} "
            f"ind {pct(induction['induction_among_quiet'])}",
            flush=True,
        )
        write_report(destination, report)
    del adapter
    torch.cuda.empty_cache()
    print(f"MODEL_DONE {model_key}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3_4b,qwen3_8b,qwen35_9b,granite_3p3_8b")
    parser.add_argument("--alphas", default="1.5,3")
    parser.add_argument("--random-alphas", default="")
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--output-root", type=Path, default=transfer.REPO_ROOT / "results" / "transfer")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    alphas = parse_alphas(args.alphas)
    random_alphas = parse_alphas(args.random_alphas)
    started = time.time()
    for model_key in [item.strip() for item in args.models.split(",") if item.strip()]:
        sweep_model(model_key, args.output_root, args.device, alphas, random_alphas, args.token_budget)
    print(f"DONE alphas elapsed={time.time() - started:.0f}", flush=True)


if __name__ == "__main__":
    main()
