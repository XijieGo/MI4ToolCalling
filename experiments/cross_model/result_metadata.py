"""Metric provenance and reconstruction diagnostics for generated summaries."""
from __future__ import annotations


def transcoder_metadata(summary: dict) -> dict:
    metadata = dict(summary)
    status = metadata.get("status", "")
    if (metadata.get("K_corrupt_over_K_clean") is not None
            and status not in {"reference", "from_feature_summary"} and not status.startswith("measured")):
        metadata.update(status="reference", source="Reference formation summary")
    # Read earlier result schemas while writing the current diagnostic names.
    for previous, current in (
        ("unreliable_release_layers", "reconstruction_flagged_release_layers"),
        ("unreliable_window_layers", "reconstruction_flagged_window_layers"),
    ):
        if previous in metadata:
            metadata.setdefault(current, metadata.pop(previous))
    if "quality_flag_criterion" in metadata:
        metadata["quality_flag_criterion"] = (
            "Training clean or corrupt reconstruction VE < 0 or undefined; "
            "K aggregates all supplied fixed-window layers"
        )
    return metadata


def transcoder_notes(metadata: dict) -> list[str]:
    notes = []
    if metadata.get("status") in {"reference", "from_feature_summary"}:
        notes.append(f"K source: {metadata.get('source', 'reference formation summary')}.")
    measured = metadata.get("measured_window_layers", [])
    missing = metadata.get("missing_window_layers", [])
    if missing:
        window = sorted(set(measured) | set(missing))
        notes.append(f"K checkpoint layers: {measured}; fixed formation window: {window}.")
    flagged = metadata.get("reconstruction_flagged_window_layers", [])
    if flagged:
        notes.append(
            f"Training reconstruction diagnostic: VE < 0 or undefined at layers {flagged}. "
            "K aggregates all supplied fixed-window layers."
        )
    return notes
