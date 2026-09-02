#!/usr/bin/env python3
"""Historical reconstruction of the implicit-intent source collection.

Rule-based only: no LLM is called anywhere in this script.  Naturalness comes
from real human-written text already in the paper's own corpora (APPS
docstrings, FEVER claims, Spider's `question` column, Vectrix email bodies).
The only synthesis is (a) picking unused source records, (b) wrapping them in
a small fixed bank of non-imperative carrier phrases, and (c) rejecting any
render that contains a banned verb anywhere in the string.

This script is retained for provenance, but its D3--D5 raw source pool was an
internal construction input and is deliberately not a prerequisite for a
standalone rerun.  Start from ``implicit_intent_oversampled_600.jsonl`` and
``run_cross_model_removal.py`` for the released experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

from release_paths import ARTIFACT_DIR as OUT_DIR
from release_paths import REPO_ROOT as PROJECT_ROOT
from release_paths import SRC_ROOT

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multidomain.common import TOOL_SCHEMAS, build_template  # noqa: E402

SEED = 42
N_PER_PATTERN = 30  # oversample; final selection keeps 10/pattern that are also baseline-positive

# ---------------------------------------------------------------------------
# Banned-verb blocklist: union of every clean/corrupt verb actually used in
# the paper's D1/D3/D4/D5 construction, plus a conservative safety net of
# directive/analysis synonyms and the imperative Spider question-openers.
# Word-boundary regex over hand-enumerated inflections (no lemmatizer).
# ---------------------------------------------------------------------------
BANNED_INFLECTIONS: dict[str, list[str]] = {
    # D1 clean/corrupt
    "complete": ["complete", "completes", "completed", "completing"],
    "build": ["build", "builds", "built", "building"],
    "write": ["write", "writes", "wrote", "written", "writing"],
    "save": ["save", "saves", "saved", "saving"],
    "add": ["add", "adds", "added", "adding"],
    "discuss": ["discuss", "discusses", "discussed", "discussing"],
    "explore": ["explore", "explores", "explored", "exploring"],
    "study": ["study", "studies", "studied", "studying"],
    "review": ["review", "reviews", "reviewed", "reviewing"],
    "inspect": ["inspect", "inspects", "inspected", "inspecting"],
    # D3 clean/corrupt
    "search": ["search", "searches", "searched", "searching"],
    "describe": ["describe", "describes", "described", "describing"],
    # D4 clean/corrupt
    "run": ["run", "runs", "ran", "running"],
    "execute": ["execute", "executes", "executed", "executing"],
    "query": ["query", "queries", "queried", "querying"],
    "fetch": ["fetch", "fetches", "fetched", "fetching"],
    "detail": ["detail", "details", "detailed", "detailing"],
    # D5 clean/corrupt
    "send": ["send", "sends", "sent", "sending"],
    "mail": ["mail", "mails", "mailed", "mailing"],
    "forward": ["forward", "forwards", "forwarded", "forwarding"],
    "dispatch": ["dispatch", "dispatches", "dispatched", "dispatching"],
    "submit": ["submit", "submits", "submitted", "submitting"],
    # safety-net synonyms: tight, kept close to the *specific* pool verbs
    # above (near-synonyms a reader would recognize as the same causal cue),
    # not generic English verbs. "found/get/make/put/call/use/open/read/
    # give/tell/count/return/show/list/ask/answer/please" were tried and
    # dropped: at N=4500+ natural-sentence pools they rejected >99% of
    # candidates because they are ordinary English, not tool-call cues.
    "implement": ["implement", "implements", "implemented", "implementing"],
    "generate": ["generate", "generates", "generated", "generating"],
    "draft": ["draft", "drafts", "drafted", "drafting"],
    "compose": ["compose", "composes", "composed", "composing"],
    "verify": ["verify", "verifies", "verified", "verifying"],
    "confirm": ["confirm", "confirms", "confirmed", "confirming"],
    "examine": ["examine", "examines", "examined", "examining"],
    "analyze": ["analyze", "analyzes", "analyzed", "analyzing"],
    "audit": ["audit", "audits", "audited", "auditing"],
    "investigate": ["investigate", "investigates", "investigated", "investigating"],
    "browse": ["browse", "browses", "browsed", "browsing"],
    "retrieve": ["retrieve", "retrieves", "retrieved", "retrieving"],
    "pull": ["pull", "pulls", "pulled", "pulling"],
    "post": ["post", "posts", "posted", "posting"],
    "deliver": ["deliver", "delivers", "delivered", "delivering"],
    "transmit": ["transmit", "transmits", "transmitted", "transmitting"],
}
LOOK_UP_RE = re.compile(r"\blook(s|ed|ing)?\s+up\b", re.IGNORECASE)
_ALL_SURFACE_FORMS = sorted({f for forms in BANNED_INFLECTIONS.values() for f in forms}, key=len, reverse=True)
BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _ALL_SURFACE_FORMS) + r")\b", re.IGNORECASE)

SAFE_OPENERS = ("what", "how", "which", "who", "when")


def has_banned_verb(text: str) -> str | None:
    m = LOOK_UP_RE.search(text)
    if m:
        return m.group(0)
    m = BANNED_RE.search(text)
    return m.group(0) if m else None


def stable_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def strip_period(s: str) -> str:
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Load the 500-pair source_ids already used, per domain, so the new pool is
# disjoint from the generalization/no-verb tables.
# ---------------------------------------------------------------------------

def used_ids_d1() -> set[str]:
    used = set()
    for split in ("train", "test"):
        for row in map(json.loads, open(PROJECT_ROOT / f"datasets/v4_multidomain_balanced/D1/{split}/clean/manifest.jsonl")):
            used.add(row["source_filename"])
    return used


def used_ids_other(domain: str) -> set[str]:
    used = set()
    path = PROJECT_ROOT / f"datasets/v4_multidomain_balanced/{domain}/selected_pairs.jsonl"
    for row in map(json.loads, open(path)):
        used.add(row["source_id"])
    return used


# ---------------------------------------------------------------------------
# Domain-specific candidate pools -> list[dict] with a "source_id" and the
# raw fields each pattern bank needs.
# ---------------------------------------------------------------------------

def pool_d1() -> list[dict]:
    used = used_ids_d1()
    out = []
    for split_dir in ("train", "test"):
        manifest_path = PROJECT_ROOT / f"datasets/{split_dir}/clean/manifest.jsonl"
        for row in map(json.loads, open(manifest_path)):
            if row.get("template_kind") != "solve.py":
                continue
            source_id = row["source_filename"]
            if source_id in used:
                continue
            fpath = PROJECT_ROOT / f"datasets/{split_dir}/clean/{row['output_filename']}"
            text = fpath.read_text(encoding="utf-8")
            marker = "<|im_start|>user\n"
            start = text.find(marker) + len(marker)
            end = text.find("<|im_end|>", start)
            user_block = text[start:end]
            lines = user_block.split("\n", 1)
            if len(lines) != 2:
                continue
            _instruction_line, body = lines
            body = body.rstrip("\n")
            m = re.search(r"\bdef\s+(\w+)\s*\(", body)
            if not m:
                continue
            func = m.group(1)
            out.append({
                "source_id": source_id,
                "func": func,
                "file": "solve.py",
                "block": body,
            })
    out.sort(key=lambda r: stable_rank(SEED, f"D1:{r['source_id']}"))
    return out


def pool_d3() -> list[dict]:
    used = used_ids_other("D3")
    out = []
    for row in map(json.loads, open(PROJECT_ROOT / "outputs/v4_multidomain_balanced/source_pool/records/D3.jsonl")):
        if row["source_id"] in used:
            continue
        body = row["body"]
        lines = body.split("\n")
        claim = lines[0].removeprefix("Search query: ").strip()
        entity = lines[1].removeprefix("Entity: ").strip() if len(lines) > 1 else row["source_metadata"].get("entity", "")
        out.append({
            "source_id": row["source_id"],
            "claim": claim,
            "claim_nop": strip_period(claim),
            "entity": entity,
        })
    out.sort(key=lambda r: stable_rank(SEED, f"D3:{r['source_id']}"))
    return out


def pool_d4() -> list[dict]:
    used = used_ids_other("D4")
    out = []
    for row in map(json.loads, open(PROJECT_ROOT / "outputs/v4_multidomain_balanced/source_pool/records/D4.jsonl")):
        if row["source_id"] in used:
            continue
        question = row["source_metadata"].get("question", "").strip()
        if not question:
            continue
        first_word = re.match(r"[A-Za-z']+", question)
        if not first_word or first_word.group(0).lower() not in SAFE_OPENERS:
            continue
        if not question.endswith("?"):
            question = question + "?"
        db_id = row["source_metadata"]["db_id"]
        body = row["body"]
        schema_lines = [ln.strip() for ln in body.split("\n") if ln.strip().startswith(tuple(row["source_metadata"]["tables"]))]
        schema0 = schema_lines[0] if schema_lines else ""
        out.append({
            "source_id": row["source_id"],
            "db_id": db_id,
            "question_full": question,
            "schema0": schema0,
        })
    out.sort(key=lambda r: stable_rank(SEED, f"D4:{r['source_id']}"))
    return out


def pool_d5() -> list[dict]:
    used = used_ids_other("D5")
    out = []
    for row in map(json.loads, open(PROJECT_ROOT / "outputs/v4_multidomain_balanced/source_pool/records/D5.jsonl")):
        if row["source_id"] in used:
            continue
        body = row["body"]
        m_subj = re.search(r"^Subject: (.*)$", body, re.MULTILINE)
        m_body = re.search(r"^Body: (.*)$", body, re.MULTILINE | re.DOTALL)
        if not m_subj or not m_body:
            continue
        subject = m_subj.group(1).strip()
        body_text = m_body.group(1).strip()
        out.append({
            "source_id": row["source_id"],
            "subject_nop": strip_period(subject),
            "body_text": body_text,
            "recipient_ref": "the recipient",
        })
    out.sort(key=lambda r: stable_rank(SEED, f"D5:{r['source_id']}"))
    return out


# ---------------------------------------------------------------------------
# Carrier stem banks, per domain per pattern.  Every stem is a fixed string
# with slot placeholders; slot values are the real record fields above.
# ---------------------------------------------------------------------------

STEMS: dict[str, dict[str, list[str]]] = {
    "D1": {
        "P1": [
            "Does `{file}` already handle `{func}` correctly, or is it still just the stub below?\n{block}",
            "Is `{func}` in `{file}` actually finished, given the docstring below?\n{block}",
            "Has anyone gotten to `{func}` yet? Here's the docstring `{file}` still has:\n{block}",
            "What's `{func}` supposed to do? This is all `{file}` has so far:\n{block}",
        ],
        "P2": [
            "I haven't gotten to `{func}` yet — `{file}` still only has the docstring below:\n{block}",
            "I'm not sure `{func}` does anything right now. This is all that's in `{file}`:\n{block}",
            "I still need to sort out `{func}`. Here's what `{file}` currently has:\n{block}",
            "I keep meaning to go back to `{func}` in `{file}`, and it's still just this:\n{block}",
        ],
        "P3": [
            "`{func}` has been sitting untouched in `{file}` since the ticket opened:\n{block}",
            "Nobody's gotten around to `{func}` yet. `{file}` still only has this:\n{block}",
            "The stub for `{func}` in `{file}` hasn't changed in days:\n{block}",
            "There's an open item for `{func}` that `{file}` doesn't handle yet:\n{block}",
        ],
        "P4": [
            "If `{func}` still isn't handling the case in this docstring, the tests stay red:\n{block}",
            "Unless `{func}` matches this docstring, `{file}` won't pass:\n{block}",
            "If `{file}` is still missing this, the pipeline stays red:\n{block}",
            "Should `{func}` end up matching the docstring below, the failing test would finally pass:\n{block}",
        ],
        "P5": [
            "`{func}` in `{file}` is still just a stub. Docstring:\n{block}",
            "`{file}`, `{func}`:\n{block}",
            "Still open — `{func}` in `{file}`:\n{block}",
            "`{file}` is missing `{func}`:\n{block}",
        ],
    },
    "D3": {
        "P1": [
            "Is it true that {claim_nop}?",
            "Does this actually check out — {claim_nop}?",
            "Quick sanity question: {claim_nop}?",
            "Is this actually accurate — {claim_nop}?",
        ],
        "P2": [
            "I'm not sure whether {claim_nop}.",
            "I don't actually know if {claim_nop}.",
            "I've been assuming {claim_nop}, without really knowing.",
            "I keep seeing it claimed that {claim_nop}, and I'm not convinced.",
        ],
        "P3": [
            "Someone keeps repeating that {claim_nop}, and it's never been settled.",
            "There's a claim going around that {claim_nop}.",
            "The slide deck states that {claim_nop}, with no citation attached.",
            "A comment on the thread insists that {claim_nop}.",
        ],
        "P4": [
            "If it turns out {claim_nop} isn't accurate, the whole slide needs a correction.",
            "Whatever the truth about {entity} turns out to be, it changes how the section reads.",
            "If {claim_nop} holds up, the citation section is fine as-is.",
            "Should {claim_nop} turn out to be wrong, the fact sheet is wrong too.",
        ],
        "P5": [
            "{claim_nop}. First I've heard of it.",
            "{entity}: {claim_nop}.",
            "Came across this about {entity}: {claim_nop}.",
            "{claim_nop} — that's the whole claim, no source attached.",
        ],
    },
    "D4": {
        "P1": [
            "{question_full}",
            "Quick one — {question_full}",
            "For `{db_id}`: {question_full}",
            "{question_full} (`{db_id}`, `{schema0}`)",
        ],
        "P2": [
            "I still don't know — {question_full}",
            "Nobody's told me yet — {question_full}",
            "This hasn't come out of `{db_id}` yet — {question_full}",
            "I keep putting this off — {question_full}",
        ],
        "P3": [
            "There's an open item on the ticket — {question_full}",
            "Someone in standup raised this and nobody had an answer — {question_full}",
            "The dashboard is missing this number — {question_full}",
            "`{db_id}` should have the answer, but it's still missing — {question_full}",
        ],
        "P4": [
            "Whatever the answer turns out to be, it decides whether this ships — {question_full}",
            "If this number is off, the report is wrong — {question_full}",
            "Depending on the answer here, the migration plan changes — {question_full}",
            "Unless someone already has this, the meeting is stuck — {question_full}",
        ],
        "P5": [
            "`{db_id}`, `{schema0}`. {question_full}",
            "{question_full} (`{db_id}`)",
            "Still open: {question_full}",
            "`{db_id}` — {question_full}",
        ],
    },
    "D5": {
        "P1": [
            "Has {recipient_ref} heard back yet about \"{subject_nop}\"? Here's what's ready:\n{body_text}",
            "Did the reply about \"{subject_nop}\" go out already? It's ready:\n{body_text}",
            "Is {recipient_ref} still waiting on this?\n{body_text}",
            "Does {recipient_ref} already have this, or is it still sitting here?\n{body_text}",
        ],
        "P2": [
            "I haven't gotten back to {recipient_ref} about \"{subject_nop}\" yet, and the reply's already sitting here:\n{body_text}",
            "I keep forgetting {recipient_ref} is waiting on this:\n{body_text}",
            "I still owe {recipient_ref} a reply on \"{subject_nop}\":\n{body_text}",
            "This has been ready for a while and it's still sitting here:\n{body_text}",
        ],
        "P3": [
            "{recipient_ref} is still waiting to hear back about \"{subject_nop}\":\n{body_text}",
            "Nobody's gotten back to {recipient_ref} yet:\n{body_text}",
            "The reply to \"{subject_nop}\" has been sitting ready for two days:\n{body_text}",
            "There's an unanswered thread with {recipient_ref}:\n{body_text}",
        ],
        "P4": [
            "If {recipient_ref} doesn't hear back today, the deadline slips:\n{body_text}",
            "Unless this reaches {recipient_ref} soon, the thread goes stale:\n{body_text}",
            "Should this stay unanswered much longer, {recipient_ref} will escalate:\n{body_text}",
            "As long as this sits here, {recipient_ref} is stuck waiting:\n{body_text}",
        ],
        "P5": [
            "{recipient_ref}, \"{subject_nop}\":\n{body_text}",
            "Still unanswered — {recipient_ref}:\n{body_text}",
            "Ready to go, {recipient_ref} waiting:\n{body_text}",
            "\"{subject_nop}\" — still open:\n{body_text}",
        ],
    },
}

PATTERNS = ("P1", "P2", "P3", "P4", "P5")


def build_domain_items(domain: str, pool: list[dict]) -> list[dict]:
    rng = random.Random(stable_rank(SEED, f"stems:{domain}"))
    items: list[dict] = []
    pool_iter = iter(pool)
    exhausted = False
    rejects = []
    for pattern in PATTERNS:
        stems = STEMS[domain][pattern]
        order = stems * ((N_PER_PATTERN // len(stems)) + 1)
        rng.shuffle(order)
        order = order[:N_PER_PATTERN]
        got = 0
        while got < N_PER_PATTERN:
            try:
                rec = next(pool_iter)
            except StopIteration:
                exhausted = True
                break
            stem = order[got]
            try:
                text = stem.format(**rec)
            except KeyError as e:
                rejects.append({"source_id": rec["source_id"], "pattern": pattern, "reason": f"missing field {e}"})
                continue
            bad = has_banned_verb(text)
            if bad:
                rejects.append({"source_id": rec["source_id"], "pattern": pattern, "reason": f"banned verb '{bad}'"})
                continue
            items.append({
                "domain": domain,
                "pattern": pattern,
                "source_id": rec["source_id"],
                "text": text,
            })
            got += 1
        if exhausted:
            break
    return items, rejects


def render_prompt(domain: str, reference_path: Path, user_text: str, schema: str) -> str:
    template = build_template(reference_path)
    return (
        template.prefix_before_schema
        + schema
        + template.after_schema_before_user
        + user_text.strip()
        + "\n"
        + template.assistant_suffix
    )


def extract_d1_schema(reference_path: Path) -> str:
    text = reference_path.read_text(encoding="utf-8")
    start = text.find("<tools>\n") + len("<tools>\n")
    end = text.find("\n</tools>", start)
    return text[start:end]


def main() -> None:
    domains = {
        "D1": (pool_d1, PROJECT_ROOT / "datasets/train/clean/apps_python_1.txt"),
        "D3": (pool_d3, PROJECT_ROOT / "datasets/v4_multidomain_balanced/D3/train/clean/d3_0001.txt"),
        "D4": (pool_d4, PROJECT_ROOT / "datasets/v4_multidomain_balanced/D4/train/clean/d4_0006.txt"),
        "D5": (pool_d5, PROJECT_ROOT / "datasets/v4_multidomain_balanced/D5/train/clean/d5_0001.txt"),
    }
    schemas = dict(TOOL_SCHEMAS)
    schemas["D1"] = extract_d1_schema(domains["D1"][1])

    all_items = []
    all_rejects = []
    summary = {}
    for domain, (pool_fn, ref_path) in domains.items():
        pool = pool_fn()
        items, rejects = build_domain_items(domain, pool)
        for i, item in enumerate(items):
            item["item_id"] = f"{domain.lower()}_implicit_{i+1:03d}"
            item["prompt"] = render_prompt(domain, ref_path, item["text"], schemas[domain])
        all_items.extend(items)
        all_rejects.extend(rejects)
        summary[domain] = {"pool_size": len(pool), "built": len(items), "rejected": len(rejects)}
        print(f"{domain}: pool={len(pool)} built={len(items)} rejected={len(rejects)}")

    out_path = OUT_DIR / "implicit_intent_200.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    rej_path = OUT_DIR / "implicit_intent_rejects.jsonl"
    with rej_path.open("w", encoding="utf-8") as f:
        for r in all_rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (OUT_DIR / "implicit_intent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTotal items: {len(all_items)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
