#!/usr/bin/env python3
"""Build D3/D4/D5 candidate pairs using only deterministic source-field rules.

No model is loaded here.  The resulting JSONL contains all verb combinations
for each rule-valid source record and is the only input to later behavioral
screening.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq

from .common import (
    DOMAINS,
    Candidate,
    assert_character_minimal_pair,
    build_template,
    render_prompt,
    sha256_text,
    stable_rank,
    write_json,
    write_jsonl,
)


FORBIDDEN_MARKERS = ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tools>", "<tools>")
WORD_RE = re.compile(r"\b[\w'-]+\b")
TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?)",
    flags=re.IGNORECASE,
)
THREAD_RE = re.compile(r"THR-\d+")
TRIVIAL_SELECT_STAR_RE = re.compile(r"\bSELECT\s+\*\s+FROM\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-reference", type=Path, required=True)
    parser.add_argument("--fever-train", type=Path, required=True)
    parser.add_argument("--spider-train", type=Path, required=True)
    parser.add_argument("--spider-tables", type=Path, required=True)
    parser.add_argument("--vectrix-emails", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d3-body-limit", type=int, default=5000)
    parser.add_argument("--d4-body-limit", type=int, default=5000)
    parser.add_argument("--d5-body-limit", type=int, default=0, help="0 means every rule-valid Vectrix email")
    parser.add_argument("--domains", nargs="+", choices=tuple(DOMAINS), default=tuple(DOMAINS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_inline(value: Any) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def has_forbidden_marker(value: str) -> bool:
    lower = value.lower()
    return any(marker.lower() in lower for marker in FORBIDDEN_MARKERS)


def first_evidence_page(evidence: Any) -> str | None:
    if not isinstance(evidence, list):
        return None
    for evidence_set in evidence:
        if not isinstance(evidence_set, list):
            continue
        for item in evidence_set:
            if isinstance(item, (list, tuple)) and len(item) >= 4 and item[2]:
                # FEVER/Wikipedia disambiguators use literal Penn Treebank
                # bracket markers (for example ``-LRB-``).  Normalize those
                # source-format artifacts deterministically so the finished
                # object remains readable without changing its meaning.
                title = clean_inline(item[2]).replace("_", " ")
                for raw, rendered in (
                    ("-LRB-", "("),
                    ("-RRB-", ")"),
                    ("-LSB-", "["),
                    ("-RSB-", "]"),
                    ("-LCB-", "{"),
                    ("-RCB-", "}"),
                ):
                    title = title.replace(raw, rendered)
                title = re.sub(r"\s+([\)\]\}])", r"\1", title)
                title = re.sub(r"([\(\[\{])\s+", r"\1", title)
                return title
    return None


def d3_records(path: Path, *, seed: int, limit: int) -> list[dict[str, Any]]:
    """Extract factual, evidence-backed finished claims from FEVER.

    Restricting this domain to SUPPORTS avoids treating an intentionally false
    claim as an action the web-search tool should genuinely verify.  We use the
    evidence-page title only as a compact entity anchor; neither the evidence
    sentence nor any LLM-authored text is inserted into the prompt.
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # A ranged-download smoke test can end halfway through a JSONL
            # record.  Full source files are expected to be valid; skipping a
            # malformed terminal fragment keeps format checks useful without
            # changing any valid source record.
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = row.get("label")
            claim = clean_inline(row.get("claim", ""))
            entity = first_evidence_page(row.get("evidence"))
            if label != "SUPPORTS" or not entity:
                continue
            if not 8 <= word_count(claim) <= 45 or has_forbidden_marker(claim):
                continue
            source_id = str(row["id"])
            records.append(
                {
                    "source_id": source_id,
                    "source_group": entity.casefold(),
                    "source_metadata": {"label": label, "entity": entity},
                    # The FEVER claim itself is the complete, source-derived
                    # web-search query.  We expose that role explicitly rather
                    # than asking a model to formulate a query.
                    "body": f"Search query: {claim}\nEntity: {entity}",
                }
            )
    if not records:
        raise ValueError("FEVER source rules produced no SUPPORTS records")
    records.sort(key=lambda row: (stable_rank(seed, f"D3:{row['source_id']}"), row["source_id"]))
    return records if limit <= 0 else records[:limit]


def sql_table_names(query: str) -> list[str]:
    names: list[str] = []
    for name in TABLE_RE.findall(query):
        normalized = name.strip('`"[]').lower()
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def load_spider_schemas(path: Path) -> dict[str, dict[str, list[str]]]:
    """Read Spider's original table/column metadata into a compact lookup."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Spider tables file must be a JSON list: {path}")
    result: dict[str, dict[str, list[str]]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        db_id = clean_inline(row.get("db_id", ""))
        table_names = row.get("table_names_original") or row.get("table_names") or []
        column_names = row.get("column_names_original") or row.get("column_names") or []
        if not db_id or not isinstance(table_names, list) or not isinstance(column_names, list):
            continue
        tables: dict[str, list[str]] = {
            clean_inline(name).casefold(): []
            for name in table_names
            if clean_inline(name)
        }
        for item in column_names:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            table_index, column_name = item
            if not isinstance(table_index, int) or not 0 <= table_index < len(table_names):
                continue
            table_name = clean_inline(table_names[table_index]).casefold()
            column = clean_inline(column_name)
            if table_name in tables and column and column not in tables[table_name]:
                tables[table_name].append(column)
        result[db_id] = tables
    if not result:
        raise ValueError(f"No schemas found in {path}")
    return result


def d4_records(path: Path, tables_path: Path, *, seed: int, limit: int) -> list[dict[str, Any]]:
    """Build an object-given SQL prompt with the query's complete local schema."""

    table = pq.read_table(path, columns=["db_id", "query", "question"])
    schemas = load_spider_schemas(tables_path)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(table.to_pylist()):
        db_id = clean_inline(row.get("db_id", ""))
        query = clean_inline(row.get("query", ""))
        question = clean_inline(row.get("question", ""))
        upper = query.upper()
        if not db_id or not query.startswith("SELECT "):
            continue
        if any(keyword in upper for keyword in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ")):
            continue
        if not 6 <= word_count(query) <= 70 or has_forbidden_marker(query):
            continue
        if TRIVIAL_SELECT_STAR_RE.search(query):
            continue
        tables = sql_table_names(query)
        if not 1 <= len(tables) <= 4:
            continue
        db_schema = schemas.get(db_id)
        if db_schema is None or any(table_name not in db_schema or not db_schema[table_name] for table_name in tables):
            continue
        source_id = f"{db_id}:{index}"
        schema_lines = [f"  {table_name}({', '.join(db_schema[table_name])})" for table_name in tables]
        body = f"Database: {db_id}\nSchema:\n" + "\n".join(schema_lines) + f"\nSQL:\n  {query}"
        records.append(
            {
                "source_id": source_id,
                "source_group": db_id,
                "source_metadata": {"db_id": db_id, "question": question, "tables": tables},
                "body": body,
            }
        )
    records.sort(key=lambda row: (stable_rank(seed, f"D4:{row['source_id']}"), row["source_id"]))
    return records if limit <= 0 else records[:limit]


def first_sentences(value: str, *, maximum: int = 3) -> str:
    text = clean_inline(value)
    chunks = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        kept.append(chunk)
        if len(kept) >= maximum:
            break
    return " ".join(kept)


def email_thread_id(message_id: str) -> str:
    match = THREAD_RE.search(message_id)
    return match.group(0) if match else message_id


def d5_records(path: Path, *, seed: int, limit: int) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["message_id", "to", "subject", "body"])
    records: list[dict[str, Any]] = []
    for row in table.to_pylist():
        message_id = clean_inline(row.get("message_id", ""))
        recipients = row.get("to") or []
        if isinstance(recipients, str):
            recipients = [recipients]
        if not isinstance(recipients, list) or not recipients:
            continue
        subject = clean_inline(row.get("subject", ""))
        body_text = first_sentences(str(row.get("body", "")))
        if not message_id or not clean_inline(recipients[0]) or not subject:
            continue
        if not (3 <= word_count(subject) <= 20 and 25 <= word_count(body_text) <= 130):
            continue
        # Keep the released prompt set free of source email addresses.  The
        # recipient is a deterministic synthetic address, not a model rewrite.
        synthetic_recipient = f"recipient-{sha256_text(message_id)[:12]}@mail.example"
        joined = "\n".join((synthetic_recipient, subject, body_text))
        if has_forbidden_marker(joined):
            continue
        source_hash = sha256_text(message_id)[:16]
        thread_hash = sha256_text(email_thread_id(message_id))[:16]
        records.append(
            {
                "source_id": f"d5-{source_hash}",
                "source_group": f"d5-thread-{thread_hash}",
                "source_metadata": {
                    "source_message_sha256": source_hash,
                    "source_thread_sha256": thread_hash,
                    "recipient_is_synthetic": True,
                    "source_to_count": len(recipients),
                },
                "body": f"To: {synthetic_recipient}\nSubject: {subject}\nBody: {body_text}",
            }
        )
    records.sort(key=lambda row: (stable_rank(seed, f"D5:{row['source_id']}"), row["source_id"]))
    return records if limit <= 0 else records[:limit]


def candidates_for_domain(
    domain: str,
    records: Iterable[dict[str, Any]],
    *,
    template: Any,
) -> Iterable[dict[str, Any]]:
    config = DOMAINS[domain]
    for record in records:
        for clean_verb in config.clean_verbs:
            for corrupt_verb in config.corrupt_verbs:
                clean_prompt = render_prompt(
                    template,
                    schema=config.schema,
                    verb=clean_verb,
                    tail=config.tail,
                    body=record["body"],
                )
                corrupt_prompt = render_prompt(
                    template,
                    schema=config.schema,
                    verb=corrupt_verb,
                    tail=config.tail,
                    body=record["body"],
                )
                assert_character_minimal_pair(clean_prompt, corrupt_prompt, clean_verb, corrupt_verb)
                candidate = Candidate(
                    candidate_id=f"{domain}:{record['source_id']}:{clean_verb}:{corrupt_verb}",
                    domain=domain,
                    source_id=record["source_id"],
                    source_group=record["source_group"],
                    source_metadata=record["source_metadata"],
                    body=record["body"],
                    clean_verb=clean_verb,
                    corrupt_verb=corrupt_verb,
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                )
                yield candidate.as_dict()


def prepare_output(root: Path, *, overwrite: bool) -> Path:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {root}; use --overwrite for this exact directory")
        for path in root.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                import shutil

                shutil.rmtree(path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    args = parse_args()
    root = prepare_output(args.output_root.resolve(), overwrite=args.overwrite)
    template = build_template(args.template_reference.resolve())
    records_by_domain: dict[str, list[dict[str, Any]]] = {}
    if "D3" in args.domains:
        records_by_domain["D3"] = d3_records(args.fever_train.resolve(), seed=args.seed, limit=args.d3_body_limit)
    if "D4" in args.domains:
        records_by_domain["D4"] = d4_records(
            args.spider_train.resolve(),
            args.spider_tables.resolve(),
            seed=args.seed,
            limit=args.d4_body_limit,
        )
    if "D5" in args.domains:
        records_by_domain["D5"] = d5_records(args.vectrix_emails.resolve(), seed=args.seed, limit=args.d5_body_limit)
    summaries: dict[str, Any] = {}
    for domain, records in records_by_domain.items():
        records_path = root / "records" / f"{domain}.jsonl"
        candidate_path = root / "candidates" / f"{domain}.jsonl"
        record_count = write_jsonl(records_path, records)
        candidate_count = write_jsonl(candidate_path, candidates_for_domain(domain, records, template=template))
        combo_counts = Counter(
            f"{clean}/{corrupt}"
            for clean in DOMAINS[domain].clean_verbs
            for corrupt in DOMAINS[domain].corrupt_verbs
        )
        summaries[domain] = {
            "records": record_count,
            "candidates": candidate_count,
            "candidate_verb_pairs": dict(combo_counts),
            "record_path": str(records_path),
            "candidate_path": str(candidate_path),
        }
    write_json(
        root / "build_manifest.json",
        {
            "seed": args.seed,
            "template": template.__dict__,
            "domains": {key: DOMAINS[key].__dict__ for key in args.domains},
            "limits": {"D3": args.d3_body_limit, "D4": args.d4_body_limit, "D5": args.d5_body_limit},
            "sources": {
                "D3": str(args.fever_train.resolve()),
                "D4": {"train": str(args.spider_train.resolve()), "tables": str(args.spider_tables.resolve())},
                "D5": str(args.vectrix_emails.resolve()),
            },
            "summary": summaries,
        },
    )
    for domain, summary in summaries.items():
        print(f"{domain}: {summary['records']} rule-valid bodies, {summary['candidates']} candidate pairs")


if __name__ == "__main__":
    main()
