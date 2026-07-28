#!/usr/bin/env python3
"""Pilot round 3: does mu_delta transfer across domains?

Builds behaviour-filtered pairs in 5 domains + 1 affordance-reversal condition,
computes mu_delta per domain at L24 prediction position, then measures the full
source x target transfer matrix (raw and norm-matched), cosines, and a
leave-one-domain-out shared vector.
"""
import json, itertools, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(0)
MODEL = "/root/autodl-tmp/Qwen/Qwen3-8B"
LSTAR = 24  # paper's Qwen3-8B locus; hidden_states[24] = output of block 24
REF = Path("./datasets/train/clean/apps_python_1.txt").read_text()
A = REF.find("<tools>\n") + len("<tools>\n"); B = REF.find("</tools>", A)
PRE, POST = REF[:A], REF[B:]
U = POST.find("<|im_start|>user\n") + len("<|im_start|>user\n")
POST_HEAD, POST_TAIL = POST[:U], POST[POST.find("<|im_end|>", U):]

def build(schema, verb, tail, body):
    v = verb[:1].upper() + verb[1:]
    return PRE + schema + POST_HEAD + f"{v} {tail}\n{body}\n" + POST_TAIL

exec(open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot2.py").read().split("CFG = {")[0].split("def build")[0].split("S = {")[1].join(["S = {",""]) if False else "pass")

import importlib.util
spec = importlib.util.spec_from_file_location("p2", "/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot2.py")
# avoid executing pilot2's main(); re-declare the data here instead
src = Path("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot2.py").read_text()
ns = {}
exec(src.split("def main():")[0].replace('exec(open', '#exec(open'), ns)
S, D1_BODIES, D2, D3v2, D4v2, D5v2 = ns["S"], ns["D1_BODIES"], ns["D2"], ns["D3v2"], ns["D4v2"], ns["D5v2"]

DOMAINS = {
 "D1_code":   ("write_file", "the function body in solve.py based on the function definition and docstring below:",
               D1_BODIES, ["write","add","save","build","complete"], ["discuss","explore","inspect","review","study","describe"]),
 "D2_math":   ("calculator", "the answer to the word problem stated below:",
               D2, ["compute","calculate","evaluate","count","total"], ["outline","describe","review","discuss","study"]),
 "D3_search": ("web_search", "the claim shown below about the entity given:",
               D3v2, ["search","verify","lookup","check","confirm"], ["outline","describe","define"]),
 "D4_sql":    ("run_sql", "the SQL query shown below against the database schema given:",
               D4v2, ["run","execute","query","fetch"], ["outline","describe","discuss","review","study","compare","explore","inspect","document"]),
 "D5_email":  ("send_email", "the email message shown below to the recipient given:",
               D5v2, ["send","mail","forward","dispatch","submit"], ["describe","discuss","review","compare","inspect"]),
}
# affordance reversal: verb held constant, only the <tools> schema differs
REVERSAL = ("the function body in solve.py based on the function definition and docstring below:", D1_BODIES,
            [("review","submit_review")], [("review","write_file")])

def main():
    tk = AutoTokenizer.from_pretrained(MODEL)
    tc = tk.convert_tokens_to_ids("<tool_call>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    LAYERS = model.model.layers

    def run(prompts, mu=None, bs=8):
        """returns (top1_is_toolcall list, resid[L*] at last position list)"""
        tops, res = [], []
        h = None
        if mu is not None:
            def pre_hook(mod, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                hs = hs.clone(); hs[:, -1, :] += mu.to(hs.dtype)
                if args: return ((hs,) + args[1:], kwargs)
                kwargs["hidden_states"] = hs; return (args, kwargs)
            h = LAYERS[LSTAR].register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            for i in range(0, len(prompts), bs):
                enc = tk(prompts[i:i+bs], return_tensors="pt", padding=True,
                         padding_side="left", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=(mu is None))
                tops += (o.logits[:, -1, :].argmax(-1) == tc).tolist()
                if mu is None:
                    res.append(o.hidden_states[LSTAR][:, -1, :].float().cpu())
        finally:
            if h: h.remove()
        return tops, (torch.cat(res) if res else None)

    # ---------- build behaviour-filtered pairs ----------
    pairs = {}
    for name, (tool, tail, bodies, ev, av) in DOMAINS.items():
        cl = [(v, b, build(S[tool], v, tail, b)) for v in ev for b in bodies]
        co = [(v, b, build(S[tool], v, tail, b)) for v in av for b in bodies]
        ok_c, _ = run([p for _, _, p in cl]); ok_o, _ = run([p for _, _, p in co])
        cl = [x for x, k in zip(cl, ok_c) if k]; co = [x for x, k in zip(co, ok_o) if not k]
        by_body_c, by_body_o = {}, {}
        for v, b, p in cl: by_body_c.setdefault(b, []).append(p)
        for v, b, p in co: by_body_o.setdefault(b, []).append(p)
        pl = []
        for b in by_body_c:
            for cp, op in zip(by_body_c[b], by_body_o.get(b, [])):
                pl.append((cp, op))
        random.shuffle(pl)
        pairs[name] = pl
        print(f"{name:10s} clean_ok={len(cl):3d} corrupt_ok={len(co):3d} -> {len(pl)} pairs", flush=True)

    # affordance-reversal condition: SAME verb 'review', schema decides
    tail, bodies, cl_spec, co_spec = REVERSAL
    rc = [build(S[t], v, tail, b) for v, t in cl_spec for b in bodies]
    ro = [build(S[t], v, tail, b) for v, t in co_spec for b in bodies]
    okc, _ = run(rc); oko, _ = run(ro)
    rp = [(c, o) for (c, kc), (o, ko) in zip(zip(rc, okc), zip(ro, oko)) if kc and not ko]
    pairs["R_afford"] = rp
    print(f"{'R_afford':10s} -> {len(rp)} pairs (verb fixed = 'review', schema swapped)", flush=True)

    # ---------- mu per domain (train half) + eval on test half ----------
    mus, tests = {}, {}
    for name, pl in pairs.items():
        k = max(4, int(len(pl) * 0.6))
        tr, te = pl[:k], pl[k:]
        if not te: te = pl[-4:]
        _, hc = run([c for c, _ in tr]); _, ho = run([o for _, o in tr])
        mus[name] = (hc - ho).mean(0)
        tests[name] = te
        print(f"{name:10s} train={len(tr)} test={len(te)} ||mu||={mus[name].norm():.2f}", flush=True)

    names = list(mus)
    out = {"pairs": {k: len(v) for k, v in pairs.items()}, "norms": {k: float(v.norm()) for k, v in mus.items()}}

    # ---------- cosine matrix ----------
    print("\n=== cosine(mu_source, mu_target) ===", flush=True)
    print("           " + "".join(n[:9].rjust(11) for n in names))
    cos = {}
    for a in names:
        row = {b: float(torch.nn.functional.cosine_similarity(mus[a], mus[b], dim=0)) for b in names}
        cos[a] = row
        print(f"{a:10s} " + "".join(f"{row[b]:11.3f}" for b in names))
    out["cosine"] = cos

    # ---------- transfer matrix ----------
    for mode in ["raw", "normmatched"]:
        print(f"\n=== transfer: add mu_SOURCE to TARGET corrupt prompts ({mode}) -> top1 <tool_call> %  ===", flush=True)
        print("src\\tgt    " + "".join(n[:9].rjust(11) for n in names))
        M = {}
        for a in names:
            row = {}
            for b in names:
                mu = mus[a].clone()
                if mode == "normmatched":
                    mu = mu / mu.norm() * mus[b].norm()
                t, _ = run([o for _, o in tests[b]], mu=mu.cuda())
                row[b] = sum(t) / len(t)
            M[a] = row
            print(f"{a:10s} " + "".join(f"{row[b]*100:10.1f}%" for b in names))
        out[f"transfer_{mode}"] = M

    # ---------- baseline corrupt rate ----------
    base = {b: sum(run([o for _, o in tests[b]])[0]) / len(tests[b]) for b in names}
    print("\nbaseline corrupt top1 rate:", {k: f"{v*100:.0f}%" for k, v in base.items()}, flush=True)
    out["baseline"] = base

    # ---------- leave-one-domain-out shared vector ----------
    print("\n=== leave-one-domain-out: mu_shared from the other domains -> held-out domain ===", flush=True)
    lodo = {}
    for b in names:
        others = [mus[a] / mus[a].norm() for a in names if a != b]
        sh = torch.stack(others).mean(0)
        sh = sh / sh.norm() * mus[b].norm()
        t, _ = run([o for _, o in tests[b]], mu=sh.cuda())
        lodo[b] = sum(t) / len(t)
        print(f"  held-out {b:10s} flip={lodo[b]*100:5.1f}%   (within-domain {M[b][b]*100:5.1f}%)", flush=True)
    out["lodo"] = lodo

    # ---------- SVD of the stacked unit vectors ----------
    Mstack = torch.stack([mus[a] / mus[a].norm() for a in names])
    sv = torch.linalg.svdvals(Mstack)
    ev = (sv ** 2 / (sv ** 2).sum()).tolist()
    print("\nSVD variance share of stacked unit mu's:", [f"{e:.3f}" for e in ev], flush=True)
    out["svd_share"] = ev

    json.dump(out, open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot3_results.json", "w"), indent=1)

main()
