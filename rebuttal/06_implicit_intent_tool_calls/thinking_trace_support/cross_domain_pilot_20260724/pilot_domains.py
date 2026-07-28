#!/usr/bin/env python3
"""Pilot: do the proposed cross-domain scaffolds + verbs flip the call-or-no-call
decision on Qwen3-8B, the paper's main model?

Measures: is <tool_call> the top-1 first generated token.
"""
import json, itertools
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/root/autodl-tmp/Qwen/Qwen3-8B"
REF = Path("./datasets/train/clean/apps_python_1.txt").read_text()

# ---- exact scaffold reconstruction from the real dataset file -----------------
A = REF.find("<tools>\n") + len("<tools>\n")
B = REF.find("</tools>", A)
PRE, POST = REF[:A], REF[B:]                       # POST starts at "</tools>"
U = POST.find("<|im_start|>user\n") + len("<|im_start|>user\n")
POST_HEAD, POST_TAIL = POST[:U], POST[POST.find("<|im_end|>", U):]   # tail = <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>

def build(schema: str, verb: str, tail: str, body: str) -> str:
    v = verb[:1].upper() + verb[1:]
    return PRE + schema + POST_HEAD + f"{v} {tail}\n{body}\n" + POST_TAIL

S = {
"write_file": '{"type":"function","function":{"name":"write_file","description":"Write file.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}',
"calculator": '{"type":"function","function":{"name":"calculator","description":"Evaluate expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"},"precision":{"type":"integer"}},"required":["expression"]}}}',
"web_search": '{"type":"function","function":{"name":"web_search","description":"Search web.","parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}},"required":["query"]}}}',
"run_sql":    '{"type":"function","function":{"name":"run_sql","description":"Run query.","parameters":{"type":"object","properties":{"database":{"type":"string"},"query":{"type":"string"}},"required":["database","query"]}}}',
"send_email": '{"type":"function","function":{"name":"send_email","description":"Send email.","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}}',
# affordance-reversal partners
"submit_review": '{"type":"function","function":{"name":"submit_review","description":"Submit review.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"comments":{"type":"string"}},"required":["file_path","comments"]}}}',
"save_draft":   '{"type":"function","function":{"name":"save_draft","description":"Save draft.","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}}',
"read_file":    '{"type":"function","function":{"name":"read_file","description":"Read file.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"max_bytes":{"type":"integer"}},"required":["file_path"]}}}',
# irrelevant-tool control
"get_weather":  '{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string"}},"required":["city"]}}}',
}

D1_BODIES = [Path(p).read_text().split("docstring below:\n",1)[1].split("\n<|im_end|>")[0]
             for p in sorted(Path("./datasets/test/clean").glob("*_python_*.txt"))[:8]]

D2_BODIES = [
"A storage tank holds 4,860 litres. Each hour a pump removes 7.5% of the volume remaining at the start of that hour. After how many whole hours does the remaining volume first fall below 1,200 litres?",
"Maria buys 14 crates of apples at $23 each and sells 60% of the apples for $1.80 each. Each crate holds 40 apples. The rest spoil. What is her total profit?",
"A train leaves at 09:12 travelling 84 km/h. A second train leaves the same station at 09:40 travelling 112 km/h on the same track. How many kilometres from the station does the second train catch the first?",
"A worker is paid $18.50 per hour for the first 38 hours and 1.6 times that rate beyond it. In a week she works 47 hours and 12% is withheld for tax. What is her take-home pay?",
"A rectangular garden measures 27 m by 19 m. A path 1.5 m wide is laid inside along all four edges. What area of the original garden remains uncovered by the path?",
"A bacterial culture starts at 2,400 cells and increases by 18% every 3 hours. What is the population after 15 hours, rounded to the nearest whole cell?",
"Three friends split a restaurant bill of $184.60. One pays 40%, the second pays 35%, and the third covers the rest plus an 18% tip on the whole bill. How much does the third pay?",
"A car depreciates 22% in its first year and 14% each year after. If it cost $31,500 new, what is its value after 4 years, to the nearest dollar?",
]

D3_BODIES = [
"Entity: Alvin Ailey American Dance Theater\nClaim: The company was founded in 1958 and its signature work Revelations premiered two years later.",
"Entity: The Great Barrier Reef\nClaim: It lies off the coast of Queensland, Australia, and is the largest coral reef system in the world.",
"Entity: Marie Curie\nClaim: She won Nobel Prizes in two different sciences and was the first woman to win a Nobel Prize.",
"Entity: The Trans-Siberian Railway\nClaim: It connects Moscow to Vladivostok and is the longest railway line in the world.",
"Entity: Mount Kilimanjaro\nClaim: It is a dormant volcano in Tanzania and the highest single free-standing mountain in the world.",
"Entity: The Rosetta Stone\nClaim: It carries the same decree in three scripts and was central to deciphering Egyptian hieroglyphs.",
"Entity: Ada Lovelace\nClaim: She wrote an algorithm for Charles Babbage's Analytical Engine and is regarded as an early programmer.",
"Entity: The Amazon River\nClaim: It discharges more water than any other river and runs through Peru, Colombia, and Brazil.",
]

D4_BODIES = [
"Schema:\n  stadium(stadium_id, location, name, capacity, average)\n  singer(singer_id, name, country, song_name, age)\n  concert(concert_id, stadium_id, year)\n  singer_in_concert(concert_id, singer_id)\nRequest: the names of stadiums that have never hosted a concert.",
"Schema:\n  department(dept_id, name, budget, building)\n  instructor(inst_id, name, dept_id, salary)\n  teaches(inst_id, course_id, semester)\nRequest: the name of each department whose average instructor salary exceeds 80000.",
"Schema:\n  customer(cust_id, name, city, signup_date)\n  orders(order_id, cust_id, order_date, total)\n  refund(refund_id, order_id, amount)\nRequest: the customers who placed more than three orders but never received a refund.",
"Schema:\n  airport(code, city, country, elevation)\n  flight(flight_id, origin, destination, minutes)\n  airline(airline_id, name, country)\nRequest: the destination cities reachable from Zurich in under 120 minutes.",
"Schema:\n  book(book_id, title, author_id, year, pages)\n  author(author_id, name, birth_year, country)\n  loan(loan_id, book_id, member_id, due_date)\nRequest: the titles of books by authors born before 1900 that are currently on loan.",
"Schema:\n  product(prod_id, name, category, price)\n  store(store_id, city, region, opened)\n  stock(store_id, prod_id, quantity)\nRequest: the categories with total stock below 50 units across all stores.",
"Schema:\n  patient(pat_id, name, birth_year, city)\n  visit(visit_id, pat_id, clinic_id, visit_date)\n  clinic(clinic_id, name, region, beds)\nRequest: the clinics that saw more than 200 distinct patients last year.",
"Schema:\n  team(team_id, name, league, founded)\n  player(player_id, name, team_id, position)\n  match(match_id, home_id, away_id, season, goals)\nRequest: the teams that scored no goals at home during the 2019 season.",
]

D5_BODIES = [
"Recipient: procurement@northgate-labs.example\nSubject: Revised delivery window for PO-4471\nContent: confirm the new window is 12-16 August, note the two-week slip on the calibration units, and ask for written acknowledgement by Friday.",
"Recipient: facilities@brightline-group.example\nSubject: Server room cooling fault\nContent: report that the unit on level 3 failed overnight, note the temperature reached 34C, and request an engineer visit before Thursday.",
"Recipient: hr@calder-associates.example\nSubject: Onboarding schedule for the September cohort\nContent: propose the induction on 4 September, list the three sessions required, and ask for room bookings to be confirmed this week.",
"Recipient: support@meridian-cloud.example\nSubject: Billing discrepancy on invoice INV-20881\nContent: state that the invoice charges for 40 seats rather than 25, attach the seat report, and request a corrected invoice.",
"Recipient: editors@harbourpress.example\nSubject: Revised manuscript deadline\nContent: explain that the third chapter needs another two weeks, propose 30 October, and confirm the remaining chapters stay on schedule.",
"Recipient: logistics@varden-freight.example\nSubject: Container hold at Rotterdam\nContent: note that container VFR-7712 has been held since Monday, ask for the customs reference, and request a revised arrival estimate.",
"Recipient: alumni@westbrook-college.example\nSubject: Guest lecture on 14 November\nContent: accept the invitation, ask for the expected audience size, and confirm that a projector and microphone will be available.",
"Recipient: safety@lindmark-works.example\nSubject: Quarterly inspection findings\nContent: summarise the two minor findings, note that both were closed on site, and ask for the signed report by end of month.",
]

TAIL = {
 "D1": ("write_file", "the function body in solve.py based on the function definition and docstring below:", D1_BODIES),
 "D2": ("calculator", "the answer to the word problem stated below:", D2_BODIES),
 "D3": ("web_search", "the factual claim about the entity stated below:", D3_BODIES),
 "D4": ("run_sql",    "the SQL query for the database request stated below:", D4_BODIES),
 "D5": ("send_email", "the message to the recipient specified below:", D5_BODIES),
}

VERBS = {
 "D1": (["write","add","save","build","complete","create","insert","patch"],
        ["discuss","explore","inspect","review","study","describe","outline","document","compare"]),
 "D2": (["compute","calculate","evaluate","count","total","return","check"],
        ["discuss","outline","describe","review","study","explore","define","document"]),
 "D3": (["search","find","verify","check","confirm","lookup","retrieve","look"],
        ["discuss","describe","define","review","study","outline","document","explore"]),
 "D4": (["run","execute","query","fetch","list","select","extract","return"],
        ["discuss","describe","review","study","outline","document","explore","inspect"]),
 "D5": (["send","mail","forward","reply","dispatch","submit","post"],
        ["draft","plan","outline","sketch","compose","review","discuss","describe"]),
}

def main():
    tk = AutoTokenizer.from_pretrained(MODEL)
    tc_id = tk.convert_tokens_to_ids("<tool_call>")
    print("tool_call id:", tc_id, flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def rate(prompts):
        hits, ex = 0, []
        for i in range(0, len(prompts), 8):
            batch = prompts[i:i+8]
            enc = tk(batch, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model(**enc).logits[:, -1, :]
            top = out.argmax(-1).tolist()
            hits += sum(t == tc_id for t in top)
            ex += [tk.decode([t]) for t in top]
        return hits / len(prompts), ex

    results = {}
    # ---- per-domain verb sweep ----
    for dom, (tool, tail, bodies) in TAIL.items():
        ex_v, an_v = VERBS[dom]
        results[dom] = {}
        for side, pool in (("exec", ex_v), ("analysis", an_v)):
            for v in pool:
                r, ex = rate([build(S[tool], v, tail, b) for b in bodies])
                results[dom][f"{side}:{v}"] = {"rate": r, "tops": ex[:3]}
                print(f"{dom} {side:8s} {v:10s} tool_call={r*100:5.1f}%  first_tokens={ex[:3]}", flush=True)

    # ---- affordance reversal: SAME user turn, different tool schema ----
    print("\n=== AFFORDANCE REVERSAL (identical user turn, schema swapped) ===", flush=True)
    rev = {}
    for label, (verb, tool_a, tool_b, dom) in {
        "review_writefile_vs_submitreview": ("review", "write_file", "submit_review", "D1"),
        "document_writefile_vs_submitrev":  ("document", "write_file", "submit_review", "D1"),
        "inspect_writefile_vs_readfile":    ("inspect", "write_file", "read_file", "D1"),
        "draft_sendemail_vs_savedraft":     ("draft", "send_email", "save_draft", "D5"),
        "outline_sendemail_vs_savedraft":   ("outline", "send_email", "save_draft", "D5"),
    }.items():
        tool, tail, bodies = TAIL[dom]
        ra, _ = rate([build(S[tool_a], verb, tail, b) for b in bodies])
        rb, _ = rate([build(S[tool_b], verb, tail, b) for b in bodies])
        rev[label] = {"A": ra, "B": rb}
        print(f"{label:36s} {tool_a:14s}={ra*100:5.1f}%   {tool_b:14s}={rb*100:5.1f}%", flush=True)

    # ---- irrelevant-tool control: execution verb, unusable tool ----
    print("\n=== IRRELEVANT TOOL (exec verb, get_weather schema) ===", flush=True)
    irr = {}
    for dom in ["D1", "D2", "D4"]:
        tool, tail, bodies = TAIL[dom]
        v = VERBS[dom][0][0]
        r0, _ = rate([build(S[tool], v, tail, b) for b in bodies])
        r1, _ = rate([build(S["get_weather"], v, tail, b) for b in bodies])
        irr[dom] = {"native": r0, "irrelevant": r1}
        print(f"{dom} verb={v:10s} native={r0*100:5.1f}%  get_weather={r1*100:5.1f}%", flush=True)

    json.dump({"domains": results, "reversal": rev, "irrelevant": irr},
              open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
