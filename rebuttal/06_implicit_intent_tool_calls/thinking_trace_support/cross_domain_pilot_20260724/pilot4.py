#!/usr/bin/env python3
"""Pilot 4, serving two constraints the user just set:
 (a) ONE shared analysis-verb pool used identically in every domain
 (b) ONE shared injection layer for every domain -> sweep layers per domain
 (c) try to widen D3's thin analysis pool with alternative instruction tails
"""
import json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(0)
MODEL = "/root/autodl-tmp/Qwen/Qwen3-8B"
REF = Path("./datasets/train/clean/apps_python_1.txt").read_text()
A = REF.find("<tools>\n") + len("<tools>\n"); B = REF.find("</tools>", A)
PRE, POST = REF[:A], REF[B:]
U = POST.find("<|im_start|>user\n") + len("<|im_start|>user\n")
POST_HEAD, POST_TAIL = POST[:U], POST[POST.find("<|im_end|>", U):]
def build(schema, verb, tail, body):
    return PRE + schema + POST_HEAD + f"{verb[:1].upper()+verb[1:]} {tail}\n{body}\n" + POST_TAIL

src = Path("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot2.py").read_text()
ns = {}; exec(src.split("def main():")[0].replace('exec(open', '#exec(open'), ns)
S, D1_BODIES, D2, D3v2, D4v2, D5v2 = ns["S"], ns["D1_BODIES"], ns["D2"], ns["D3v2"], ns["D4v2"], ns["D5v2"]

# ONE shared analysis pool, identical in every domain
SHARED_ANA = ["describe", "discuss", "review", "study", "inspect", "compare", "outline"]

DOM = {
 "D1_code":   ("write_file", "the function body in solve.py based on the function definition and docstring below:",
               D1_BODIES, ["write","add","save","build","complete"]),
 "D2_math":   ("calculator", "the answer to the word problem stated below:",
               D2, ["compute","calculate","evaluate","count","total"]),
 "D3_search": ("web_search", "the claim shown below about the entity given:",
               D3v2, ["search","verify","lookup","check","confirm"]),
 "D4_sql":    ("run_sql", "the SQL query shown below against the database schema given:",
               D4v2, ["run","execute","query","fetch"]),
 "D5_email":  ("send_email", "the email message shown below to the recipient given:",
               D5v2, ["send","mail","forward","dispatch","submit"]),
}
D3_ALT_TAILS = [
 "the claim shown below about the entity given:",
 "the claim about the entity stated below:",
 "the factual record shown below for the entity given:",
 "the statement shown below concerning the entity given:",
]
LAYERS_SWEEP = [14, 16, 18, 20, 22, 23, 24, 25, 26, 28, 30, 32]

def main():
    tk = AutoTokenizer.from_pretrained(MODEL)
    tc = tk.convert_tokens_to_ids("<tool_call>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    BLOCKS = model.model.layers

    def fwd(prompts, mu=None, layer=None, bs=8, want_hs=False):
        tops, hs_all = [], []
        h = None
        if mu is not None:
            def pre_hook(mod, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                x = x.clone(); x[:, -1, :] += mu.to(x.dtype)
                if args: return ((x,) + args[1:], kwargs)
                kwargs["hidden_states"] = x; return (args, kwargs)
            h = BLOCKS[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            for i in range(0, len(prompts), bs):
                enc = tk(prompts[i:i+bs], return_tensors="pt", padding=True,
                         padding_side="left", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=want_hs)
                tops += (o.logits[:, -1, :].argmax(-1) == tc).tolist()
                if want_hs:
                    hs_all.append(torch.stack([o.hidden_states[l][:, -1, :] for l in range(len(o.hidden_states))]).float().cpu())
        finally:
            if h: h.remove()
        return tops, (torch.cat(hs_all, dim=1) if want_hs else None)   # [L+1, N, d]

    out = {}

    # ---------- (a) shared analysis pool, identical across domains ----------
    print("=== SHARED ANALYSIS POOL, identical in all 5 domains (tool_call %, want LOW) ===", flush=True)
    hdr = "verb".ljust(11) + "".join(d.replace("_","")[:8].rjust(10) for d in DOM)
    print(hdr, flush=True)
    grid = {}
    for v in SHARED_ANA:
        row = {}
        for d, (tool, tail, bodies, _) in DOM.items():
            r, _ = fwd([build(S[tool], v, tail, b) for b in bodies])
            row[d] = sum(r)/len(r)
        grid[v] = row
        print(v.ljust(11) + "".join(f"{row[d]*100:9.1f}%" for d in DOM), flush=True)
    out["shared_analysis"] = grid

    print("\n=== execution verbs (want HIGH) ===", flush=True)
    ex = {}
    for d, (tool, tail, bodies, ev) in DOM.items():
        ex[d] = {}
        for v in ev:
            r, _ = fwd([build(S[tool], v, tail, b) for b in bodies])
            ex[d][v] = sum(r)/len(r)
        print(f"{d:10s} " + "  ".join(f"{v}={ex[d][v]*100:.0f}%" for v in ev), flush=True)
    out["exec"] = ex

    # ---------- (c) D3 alternative tails ----------
    print("\n=== D3 alternative instruction tails (analysis side, want LOW) ===", flush=True)
    d3 = {}
    for tail in D3_ALT_TAILS:
        row = {}
        for v in SHARED_ANA:
            r, _ = fwd([build(S["web_search"], v, tail, b) for b in D3v2])
            row[v] = sum(r)/len(r)
        rex, _ = fwd([build(S["web_search"], "verify", tail, b) for b in D3v2])
        d3[tail] = {"analysis": row, "verify": sum(rex)/len(rex)}
        print(f"  '{tail[:52]}'  verify={sum(rex)/len(rex)*100:.0f}% | " +
              " ".join(f"{v[:4]}={row[v]*100:.0f}%" for v in SHARED_ANA), flush=True)
    out["d3_tails"] = d3

    # ---------- (b) layer sweep: is ONE common layer OK for all domains? ----------
    print("\n=== LAYER SWEEP: mu_d computed & injected at layer l, per domain ===", flush=True)
    pairs = {}
    for d, (tool, tail, bodies, ev) in DOM.items():
        ok_ana = [v for v in SHARED_ANA if grid[v][d] == 0.0]
        if not ok_ana: ok_ana = [min(SHARED_ANA, key=lambda v: grid[v][d])]
        cl = [build(S[tool], v, tail, b) for v in ev for b in bodies]
        co = [build(S[tool], v, tail, b) for v in ok_ana for b in bodies]
        okc, _ = fwd(cl); oko, _ = fwd(co)
        cl = [p for p, k in zip(cl, okc) if k]; co = [p for p, k in zip(co, oko) if not k]
        n = min(len(cl), len(co)); random.shuffle(cl); random.shuffle(co)
        pl = list(zip(cl[:n], co[:n])); random.shuffle(pl)
        k = int(n*0.6); pairs[d] = (pl[:k], pl[k:])
        print(f"  {d:10s} usable analysis verbs={ok_ana}  pairs={n} (train {k} / test {n-k})", flush=True)

    mus = {}
    for d, (tr, te) in pairs.items():
        _, hc = fwd([c for c, _ in tr], want_hs=True)
        _, ho = fwd([o for _, o in tr], want_hs=True)
        mus[d] = (hc - ho).mean(1)          # [L+1, d]

    print("\n" + "layer".ljust(8) + "".join(d.replace("_","")[:8].rjust(10) for d in DOM), flush=True)
    sweep = {}
    for l in LAYERS_SWEEP:
        row = {}
        for d, (tr, te) in pairs.items():
            t, _ = fwd([o for _, o in te], mu=mus[d][l].cuda(), layer=l)
            row[d] = sum(t)/len(t)
        sweep[l] = row
        print(str(l).ljust(8) + "".join(f"{row[d]*100:9.1f}%" for d in DOM), flush=True)
    out["layer_sweep"] = sweep

    # cross-domain transfer at the single shared layer 24, norm matched
    print("\n=== transfer at SHARED layer 24 (norm-matched) ===", flush=True)
    print("src\\tgt".ljust(11) + "".join(d.replace("_","")[:8].rjust(10) for d in DOM), flush=True)
    T = {}
    for a in DOM:
        row = {}
        for b in DOM:
            mu = mus[a][24] / mus[a][24].norm() * mus[b][24].norm()
            t, _ = fwd([o for _, o in pairs[b][1]], mu=mu.cuda(), layer=24)
            row[b] = sum(t)/len(t)
        T[a] = row
        print(a.ljust(11) + "".join(f"{row[b]*100:9.1f}%" for b in DOM), flush=True)
    out["transfer_L24"] = T

    print("\ncosine at L24:", flush=True)
    print("".ljust(11) + "".join(d.replace("_","")[:8].rjust(10) for d in DOM), flush=True)
    C = {}
    for a in DOM:
        C[a] = {b: float(torch.nn.functional.cosine_similarity(mus[a][24], mus[b][24], dim=0)) for b in DOM}
        print(a.ljust(11) + "".join(f"{C[a][b]:10.3f}" for b in DOM), flush=True)
    out["cosine_L24"] = C

    json.dump(out, open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot4_results.json","w"), indent=1)

main()
