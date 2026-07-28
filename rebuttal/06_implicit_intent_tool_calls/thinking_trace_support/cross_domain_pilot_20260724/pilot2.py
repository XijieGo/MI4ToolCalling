#!/usr/bin/env python3
"""Pilot round 2:
 (a) fix D3/D4/D5 by giving the *object* in the task body (as D1 gives the def)
 (b) confirm the affordance double dissociation as a full 2x2 grid
 (c) test non-verb routes to no-call (system policy / user suffix / irrelevant tool)
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/root/autodl-tmp/Qwen/Qwen3-8B"
REF = Path("./datasets/train/clean/apps_python_1.txt").read_text()
A = REF.find("<tools>\n") + len("<tools>\n"); B = REF.find("</tools>", A)
PRE, POST = REF[:A], REF[B:]
U = POST.find("<|im_start|>user\n") + len("<|im_start|>user\n")
POST_HEAD, POST_TAIL = POST[:U], POST[POST.find("<|im_end|>", U):]

def build(schema, verb, tail, body, sys_extra="", user_suffix=""):
    v = verb[:1].upper() + verb[1:]
    pre = PRE + schema + POST.replace(POST[U:], "")[:0] if False else None
    head = PRE + schema + POST_HEAD
    if sys_extra:  # inject an extra policy line into the system turn, before <|im_end|>
        head = head.replace("</tool_call><|im_end|>", "</tool_call>\n" + sys_extra + "<|im_end|>")
    return head + f"{v} {tail}\n{body}\n" + (user_suffix + "\n" if user_suffix else "") + POST_TAIL

S = {
"write_file": '{"type":"function","function":{"name":"write_file","description":"Write file.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}',
"web_search": '{"type":"function","function":{"name":"web_search","description":"Search web.","parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}},"required":["query"]}}}',
"run_sql":    '{"type":"function","function":{"name":"run_sql","description":"Run query.","parameters":{"type":"object","properties":{"database":{"type":"string"},"query":{"type":"string"}},"required":["database","query"]}}}',
"send_email": '{"type":"function","function":{"name":"send_email","description":"Send email.","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}}',
"submit_review": '{"type":"function","function":{"name":"submit_review","description":"Submit review.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"comments":{"type":"string"}},"required":["file_path","comments"]}}}',
"get_weather":  '{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string"}},"required":["city"]}}}',
"calculator": '{"type":"function","function":{"name":"calculator","description":"Evaluate expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"},"precision":{"type":"integer"}},"required":["expression"]}}}',
}

D1_BODIES = [Path(p).read_text().split("docstring below:\n",1)[1].split("\n<|im_end|>")[0]
             for p in sorted(Path("./datasets/test/clean").glob("*_python_*.txt"))[:8]]

# D4 v2: the SQL query is GIVEN in the body (parallels D1's given function definition)
D4v2 = [
"Schema:\n  stadium(stadium_id, location, name, capacity, average)\n  concert(concert_id, stadium_id, year)\nSELECT name FROM stadium WHERE stadium_id NOT IN\n  (SELECT stadium_id FROM concert);",
"Schema:\n  department(dept_id, name, budget)\n  instructor(inst_id, name, dept_id, salary)\nSELECT d.name FROM department d JOIN instructor i ON i.dept_id = d.dept_id\n  GROUP BY d.name HAVING AVG(i.salary) > 80000;",
"Schema:\n  customer(cust_id, name, city)\n  orders(order_id, cust_id, total)\n  refund(refund_id, order_id, amount)\nSELECT c.name FROM customer c JOIN orders o ON o.cust_id = c.cust_id\n  GROUP BY c.name HAVING COUNT(*) > 3;",
"Schema:\n  airport(code, city, country)\n  flight(flight_id, origin, destination, minutes)\nSELECT DISTINCT a.city FROM flight f JOIN airport a ON a.code = f.destination\n  WHERE f.origin = 'ZRH' AND f.minutes < 120;",
"Schema:\n  book(book_id, title, author_id, year)\n  author(author_id, name, birth_year)\nSELECT b.title FROM book b JOIN author a ON a.author_id = b.author_id\n  WHERE a.birth_year < 1900;",
"Schema:\n  product(prod_id, name, category, price)\n  stock(store_id, prod_id, quantity)\nSELECT p.category FROM product p JOIN stock s ON s.prod_id = p.prod_id\n  GROUP BY p.category HAVING SUM(s.quantity) < 50;",
"Schema:\n  patient(pat_id, name, city)\n  visit(visit_id, pat_id, clinic_id, visit_date)\nSELECT clinic_id FROM visit WHERE visit_date >= '2024-01-01'\n  GROUP BY clinic_id HAVING COUNT(DISTINCT pat_id) > 200;",
"Schema:\n  team(team_id, name, league)\n  match(match_id, home_id, away_id, season, goals)\nSELECT t.name FROM team t LEFT JOIN match m ON m.home_id = t.team_id\n  AND m.season = 2019 GROUP BY t.name HAVING COALESCE(SUM(m.goals),0) = 0;",
]

# D5 v2: the finished email is GIVEN in the body
D5v2 = [
"To: procurement@northgate-labs.example\nSubject: Revised delivery window for PO-4471\nBody: The new window is 12-16 August. The calibration units have slipped by two weeks. Please send written acknowledgement by Friday.",
"To: facilities@brightline-group.example\nSubject: Server room cooling fault\nBody: The cooling unit on level 3 failed overnight and the room reached 34C. Please schedule an engineer visit before Thursday.",
"To: hr@calder-associates.example\nSubject: Onboarding schedule for the September cohort\nBody: We propose induction on 4 September with three sessions. Please confirm the room bookings this week.",
"To: support@meridian-cloud.example\nSubject: Billing discrepancy on invoice INV-20881\nBody: The invoice charges for 40 seats rather than 25. The seat report is attached. Please issue a corrected invoice.",
"To: editors@harbourpress.example\nSubject: Revised manuscript deadline\nBody: Chapter three needs another two weeks. We propose 30 October. The remaining chapters stay on schedule.",
"To: logistics@varden-freight.example\nSubject: Container hold at Rotterdam\nBody: Container VFR-7712 has been held since Monday. Please send the customs reference and a revised arrival estimate.",
"To: alumni@westbrook-college.example\nSubject: Guest lecture on 14 November\nBody: I accept the invitation. Please confirm the expected audience size and that a projector and microphone are available.",
"To: safety@lindmark-works.example\nSubject: Quarterly inspection findings\nBody: There were two minor findings and both were closed on site. Please send the signed report by end of month.",
]

D3v2 = [
"Entity: Alvin Ailey American Dance Theater\nClaim: The company was founded in 1958 and its signature work Revelations premiered two years later.",
"Entity: The Great Barrier Reef\nClaim: It lies off the coast of Queensland, Australia, and is the largest coral reef system in the world.",
"Entity: Marie Curie\nClaim: She won Nobel Prizes in two different sciences and was the first woman to win a Nobel Prize.",
"Entity: The Trans-Siberian Railway\nClaim: It connects Moscow to Vladivostok and is the longest railway line in the world.",
"Entity: Mount Kilimanjaro\nClaim: It is a dormant volcano in Tanzania and the highest free-standing mountain in the world.",
"Entity: The Rosetta Stone\nClaim: It carries the same decree in three scripts and was central to deciphering Egyptian hieroglyphs.",
"Entity: Ada Lovelace\nClaim: She wrote an algorithm for Charles Babbage's Analytical Engine and is regarded as an early programmer.",
"Entity: The Amazon River\nClaim: It discharges more water than any other river and runs through Peru, Colombia, and Brazil.",
]

D2 = [
"A storage tank holds 4,860 litres. Each hour a pump removes 7.5% of the volume remaining at the start of that hour. After how many whole hours does the remaining volume first fall below 1,200 litres?",
"Maria buys 14 crates of apples at $23 each and sells 60% of the apples for $1.80 each. Each crate holds 40 apples. The rest spoil. What is her total profit?",
"A train leaves at 09:12 travelling 84 km/h. A second leaves the same station at 09:40 travelling 112 km/h on the same track. How many kilometres from the station does the second catch the first?",
"A worker is paid $18.50 per hour for the first 38 hours and 1.6 times that rate beyond it. In a week she works 47 hours and 12% is withheld for tax. What is her take-home pay?",
"A rectangular garden measures 27 m by 19 m. A path 1.5 m wide is laid inside along all four edges. What area of the original garden remains uncovered by the path?",
"A bacterial culture starts at 2,400 cells and increases by 18% every 3 hours. What is the population after 15 hours, rounded to the nearest whole cell?",
"Three friends split a bill of $184.60. One pays 40%, the second 35%, and the third the rest plus an 18% tip on the whole bill. How much does the third pay?",
"A car depreciates 22% in its first year and 14% each year after. If it cost $31,500 new, what is its value after 4 years, to the nearest dollar?",
]

CFG = {
 "D2  calculator": ("calculator", "the answer to the word problem stated below:", D2,
    ["compute","calculate","evaluate","count","total"], ["outline","discuss","describe","review","study","compare","explore"]),
 "D3v2 web_search": ("web_search", "the claim shown below about the entity given:", D3v2,
    ["search","verify","check","confirm","lookup"], ["outline","describe","define","discuss","review","study","compare","explore","document"]),
 "D4v2 run_sql": ("run_sql", "the SQL query shown below against the database schema given:", D4v2,
    ["run","execute","query","fetch","list"], ["outline","describe","discuss","review","study","compare","explore","inspect","document"]),
 "D5v2 send_email": ("send_email", "the email message shown below to the recipient given:", D5v2,
    ["send","mail","forward","dispatch","submit"], ["outline","describe","discuss","review","study","compare","explore","inspect","draft"]),
}

def main():
    tk = AutoTokenizer.from_pretrained(MODEL)
    tc = tk.convert_tokens_to_ids("<tool_call>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    def rate(prompts):
        hits = 0; ex = []
        for i in range(0, len(prompts), 8):
            enc = tk(prompts[i:i+8], return_tensors="pt", padding=True, padding_side="left",
                     add_special_tokens=False).to("cuda")
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :]
            top = lg.argmax(-1).tolist()
            hits += sum(t == tc for t in top); ex += [tk.decode([t]) for t in top]
        return hits/len(prompts), ex

    out = {"domains": {}, "grid": {}, "routes": {}}
    for dom, (tool, tail, bodies, ev, av) in CFG.items():
        print(f"\n--- {dom} :: '{{VERB}} {tail}' ---", flush=True)
        out["domains"][dom] = {}
        for side, pool in (("exec", ev), ("analysis", av)):
            for v in pool:
                r, ex = rate([build(S[tool], v, tail, b) for b in bodies])
                out["domains"][dom][f"{side}:{v}"] = r
                print(f"  {side:8s} {v:10s} {r*100:5.1f}%   {ex[:2]}", flush=True)

    print("\n=== AFFORDANCE GRID: identical user turn, only <tools> differs (D1 bodies) ===", flush=True)
    tail1 = "the function body in solve.py based on the function definition and docstring below:"
    for v in ["write","review","document","describe","discuss","complete","comment","study"]:
        row = {}
        for tool in ["write_file", "submit_review", "get_weather"]:
            r, _ = rate([build(S[tool], v, tail1, b) for b in D1_BODIES])
            row[tool] = r
        out["grid"][v] = row
        print(f"  {v:10s} write_file={row['write_file']*100:5.1f}%  submit_review={row['submit_review']*100:5.1f}%  get_weather={row['get_weather']*100:5.1f}%", flush=True)

    print("\n=== NON-VERB ROUTES TO NO-CALL (D1 bodies, execution verb 'write') ===", flush=True)
    routes = {
      "baseline (write + write_file)":        dict(schema="write_file"),
      "system policy: no function calls":     dict(schema="write_file", sys_extra="Do not call any function. Answer the user directly in plain text."),
      "system policy: read-only session":     dict(schema="write_file", sys_extra="This session is read-only. Tools are unavailable."),
      "user suffix: explain only":            dict(schema="write_file", user_suffix="Explain your approach in text only; do not modify any files."),
      "user suffix: no tools":                dict(schema="write_file", user_suffix="Do not use any tool for this request."),
      "irrelevant tool (get_weather)":        dict(schema="get_weather"),
    }
    for name, kw in routes.items():
        sch = S[kw.pop("schema")]
        r, ex = rate([build(sch, "write", tail1, b, **kw) for b in D1_BODIES])
        out["routes"][name] = r
        print(f"  {name:38s} {r*100:5.1f}%   {ex[:2]}", flush=True)

    json.dump(out, open("/tmp/claude-0/-root/31c8ba56-b3a0-4ba8-9fd6-18fd261d6350/scratchpad/pilot2_results.json","w"), indent=1)

main()
