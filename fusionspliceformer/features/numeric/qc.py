"""Quality-control checks and summaries for raw numeric feature tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .schema import (
    ACCEPTOR_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    DONOR_FEATURE_NAMES,
    FORBIDDEN_OUTPUT_COLUMNS,
)


def validate_conservation_coverage(
    frame: pd.DataFrame,
    minimum_coverage: float = 0.99,
    *, enforce: bool = True,
) -> dict[str, float]:
    """Require compatible tracks while allowing rare, explicitly marked gaps."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    required = {"phylop_available", "phastcons_available"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Conservation availability columns are missing: {missing}")
    coverages = {
        "phylop_coverage": float(frame["phylop_available"].fillna(0).mean()),
        "phastcons_coverage": float(frame["phastcons_available"].fillna(0).mean()),
    }
    below = {name: value for name, value in coverages.items() if value < minimum_coverage}
    if below and enforce:
        detail = ", ".join(f"{name}={value:.6f}" for name, value in below.items())
        raise RuntimeError(
            "Conservation coverage is below the required threshold "
            f"{minimum_coverage:.2%}: {detail}. Check BigWig paths, chromosome naming, "
            "and GRCh38 coordinate compatibility."
        )
    return coverages


def validate_feature_frame(
    frame: pd.DataFrame,
    expected_rows: int | None = None,
    minimum_conservation_coverage: float = 0.99,
    *, defer_coverage: bool = False,
) -> dict:
    forbidden = sorted(FORBIDDEN_OUTPUT_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"Label/evidence columns leaked into feature output: {forbidden}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Row count changed: expected {expected_rows}, observed {len(frame)}")
    if "event_group_id" in frame and frame["event_group_id"].duplicated().any():
        raise ValueError("Feature output has duplicate event_group_id values")
    missing_shared = sorted(set(DONOR_FEATURE_NAMES) - set(frame.columns))
    if missing_shared:
        raise ValueError(f"Feature output is missing shared features: {missing_shared}")
    for feature in ALL_FEATURE_NAMES:
        if feature in frame.columns:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
            values = frame[feature].dropna()
            if not values.map(math.isfinite).all():
                raise ValueError(f"Feature contains non-finite non-NaN values: {feature}")
    region_counts = frame["site_type"].value_counts(dropna=False).to_dict()
    missingness = {
        feature: float(frame[feature].isna().mean())
        for feature in ALL_FEATURE_NAMES
        if feature in frame.columns
    }
    donor = frame.loc[frame["site_type"] == "donor"]
    conservation = validate_conservation_coverage(
        frame,
        minimum_coverage=minimum_conservation_coverage,
        enforce=not defer_coverage,
    )
    return {
        "status": "PASS",
        "row_count": int(len(frame)),
        "region_counts": {str(key): int(value) for key, value in region_counts.items()},
        "feature_count_acceptor": len(ACCEPTOR_FEATURE_NAMES),
        "feature_count_donor": len(DONOR_FEATURE_NAMES),
        "missing_fraction": missingness,
        **conservation,
        "minimum_conservation_coverage": float(minimum_conservation_coverage),
        "coverage_check_deferred": defer_coverage,
        "donor_acceptor_specific_columns_all_na": bool(
            donor[list(set(ACCEPTOR_FEATURE_NAMES) - set(DONOR_FEATURE_NAMES))].isna().all().all()
            if len(donor)
            else True
        ),
        "label_or_evidence_columns_present": forbidden,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
