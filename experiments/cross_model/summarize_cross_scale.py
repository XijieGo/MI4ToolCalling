#!/usr/bin/env python3
"""Render the cross-model table from completed experiment artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from result_metadata import transcoder_metadata, transcoder_notes
REPO_ROOT = Path(__file__).resolve().parents[2]


def render_table6(report: dict, destination: Path) -> None:
    rows = report['rows']
    if len(rows) != 7 or len({row['model_key'] for row in rows}) != 7:
        raise ValueError('A verified seven-model core table is required')
    fields = list(rows[0]['display'])
    lines = ['# Cross-model measured Table 6', '',
             'L is the zero-based decoder input residual (pre hook).', '',
             '| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] + ['---:']*(len(fields)-1)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(row['display'][field] for field in fields) + ' |')
    lines.extend(['', 'Domains averages conditional induction and suppression across three domains. Tau2 averages its two conditional native arms. All displayed scores use gain 1.', '',
                  f"Verified run: `{report['run_dir']}`. Detailed values and source paths: `core_cross_model_table.json`.", ''])
    for row in rows:
        metadata = row.get('K_metadata', {
            'reconstruction_flagged_window_layers': row.get('K_unreliable_window_layers', []),
        })
        for note in transcoder_notes(transcoder_metadata(metadata)):
            lines.append(f"- {row['display']['Model']}: {note}")
    destination.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=REPO_ROOT/'results')
    parser.add_argument('--output-file', type=Path, default=REPO_ROOT/'results/cross_scale_table6_summary.md')
    args = parser.parse_args()
    source = args.results_dir/'core_cross_model_table.json'
    if not source.is_file():
        raise RuntimeError('Finish the rerun and generate summarize_core_table.py\'s verified core table first')
    report = json.loads(source.read_text())
    run = Path(report['run_dir'])
    state = json.loads((run/'status.json').read_text())
    if not (run/'completed').exists() or not all(task['state'] == 'completed' for task in state['tasks']):
        raise RuntimeError('The measured run is incomplete')
    render_table6(report, args.output_file)
    print(f'Measured Table 6 written to {args.output_file}')


if __name__ == '__main__':
    main()
