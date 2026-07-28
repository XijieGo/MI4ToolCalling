import json, sys
from transformers import AutoTokenizer

TOKS = {
 "Qwen3-8B": "/root/autodl-tmp/Qwen/Qwen3-8B",
 "Qwen3-1.7B": "/root/autodl-tmp/Qwen/Qwen3-1.7B",
 "Qwen3.5-9B": "/root/autodl-tmp/Qwen/Qwen3.5-9B",
 "Devstral-24B": "/root/autodl-tmp/Mistral/Devstral-Small-2-24B-Instruct-2512",
 "Granite-3.3-8B": "/root/autodl-tmp/Granite/granite-3.3-8b-instruct",
 "OLMo-3.1-32B": "/root/autodl-tmp/OLMo/Olmo-3.1-32B-Instruct",
}

VERBS = {
"D1_exec":["write","add","save","build","complete","create","implement","update","modify","generate","patch","apply","insert","append","emit"],
"D1_ana":["discuss","explore","inspect","review","study","explain","describe","analyze","examine","assess","compare","summarize","critique","evaluate","outline","interpret","clarify","teach"],
"D2_exec":["compute","calculate","evaluate","solve","count","total","tally","determine"],
"D2_ana":["derive","explain","justify","discuss","outline","describe","motivate","teach","interpret","review"],
"D3_exec":["search","find","verify","check","retrieve","look","confirm","lookup","fetch","browse","locate"],
"D3_ana":["explain","define","describe","discuss","summarize","introduce","characterize","recall","paraphrase"],
"D4_exec":["query","run","execute","fetch","list","select","extract","return","count","retrieve"],
"D4_ana":["explain","describe","review","critique","document","discuss","interpret","teach","outline"],
"D5_exec":["send","reply","forward","deliver","mail","dispatch","submit","transmit","post"],
"D5_ana":["draft","revise","plan","propose","discuss","outline","compose","sketch","review","critique"],
"reversal":["review","summarize","draft","discuss","explain","describe","plan","comment","annotate","document","report","log","record","file","flag"],
}

allv = sorted({v for g in VERBS.values() for v in g})
TAIL = " the function body in solve.py based on the definition below:"

rows = {}
for name, path in TOKS.items():
    try:
        tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"SKIP {name}: {e}", file=sys.stderr); continue
    res = {}
    for v in allv:
        cap = v[:1].upper()+v[1:]
        line = cap + TAIL
        ids = tk(line, add_special_tokens=False)["input_ids"]
        first = tk.decode(ids[:1])
        # single-token iff first decoded token == capitalized verb exactly
        n_solo = len(tk(cap, add_special_tokens=False)["input_ids"])
        res[v] = {"ok": first == cap, "first": first, "n_solo": n_solo}
    rows[name] = res

json.dump({"verbs": VERBS, "results": rows}, open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/verb_tok.json","w"), indent=1)

# print a compact matrix
models = list(rows)
print("verb".ljust(14) + "".join(m[:9].ljust(11) for m in models))
for v in allv:
    line = v.ljust(14)
    for m in models:
        r = rows[m][v]
        line += ("OK" if r["ok"] else "x:"+r["first"].replace(" ","_")[:8]).ljust(11)
    print(line)
