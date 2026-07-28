#!/usr/bin/env python3
"""Render a deterministic verb grid over already rule-valid domain bodies.

This is used when a cross-scale screen identifies that a domain needs a
different, semantically coherent verb/tail interface.  It never alters source
text and never uses a model; it only splices records into the shared scaffold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import (
    DOMAINS,
    Candidate,
    assert_character_minimal_pair,
    build_template,
    render_prompt,
    sha256_text,
    write_json,
    write_jsonl,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--template-reference", type=Path, required=True)
    parser.add_argument("--domain", choices=tuple(DOMAINS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-verbs", nargs="+", required=True)
    parser.add_argument("--corrupt-verbs", nargs="+", required=True)
    parser.add_argument("--tail", type=str, required=True)
    parser.add_argument(
        "--tool-description",
        type=str,
        default=None,
        help=(
            "Optional replacement for the function description in the domain tool schema. "
            "The same schema is rendered for both sides of every minimal pair."
        ),
    )
    parser.add_argument("--body-limit", type=int, default=0, help="0 means every input record")
    parser.add_argument("--variant-label", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.records.resolve()))
    if args.body_limit > 0:
        records = records[: args.body_limit]
    if not records:
        raise ValueError("No records supplied")
    template = build_template(args.template_reference.resolve())
    config = DOMAINS[args.domain]
    schema = config.schema
    if args.tool_description is not None:
        parsed_schema = json.loads(schema)
        parsed_schema["function"]["description"] = args.tool_description
        schema = json.dumps(parsed_schema, ensure_ascii=False, separators=(",", ":"))
    variant_hash = sha256_text(f"{args.variant_label}:{args.tail}")[:12]

    def rows() -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record in records:
            for clean_verb in args.clean_verbs:
                for corrupt_verb in args.corrupt_verbs:
                    clean_prompt = render_prompt(
                        template, schema=schema, verb=clean_verb, tail=args.tail, body=record["body"]
                    )
                    corrupt_prompt = render_prompt(
                        template, schema=schema, verb=corrupt_verb, tail=args.tail, body=record["body"]
                    )
                    assert_character_minimal_pair(clean_prompt, corrupt_prompt, clean_verb, corrupt_verb)
                    result.append(
                        Candidate(
                            candidate_id=(
                                f"{args.domain}:{args.variant_label}:{variant_hash}:{record['source_id']}:"
                                f"{clean_verb}:{corrupt_verb}"
                            ),
                            domain=args.domain,
                            source_id=record["source_id"],
                            source_group=record["source_group"],
                            source_metadata=record["source_metadata"],
                            body=record["body"],
                            clean_verb=clean_verb,
                            corrupt_verb=corrupt_verb,
                            clean_prompt=clean_prompt,
                            corrupt_prompt=corrupt_prompt,
                        ).as_dict()
                    )
        return result

    rendered = rows()
    write_jsonl(args.output.resolve(), rendered)
    write_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "domain": args.domain,
            "variant_label": args.variant_label,
            "tail": args.tail,
            "tool_description": args.tool_description,
            "clean_verbs": args.clean_verbs,
            "corrupt_verbs": args.corrupt_verbs,
            "source_records": len(records),
            "candidate_pairs": len(rendered),
            "template": template.__dict__,
        },
    )
    print(f"{args.domain}: rendered {len(rendered)} pairs over {len(records)} records")


if __name__ == "__main__":
    main()
