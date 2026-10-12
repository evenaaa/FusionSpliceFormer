"""Distance, GC composition, and transcript-structure aggregation."""

from __future__ import annotations

import gzip
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .sequence import IndexedFasta, gc_fraction, valid_fraction


ATTRIBUTE_RE = re.compile(r'(\S+)\s+"([^"]*)"')


def _stable(identifier: object) -> str:
    value = "" if pd.isna(identifier) else str(identifier)
    return value.split(".", 1)[0]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def read_exon_lengths(gtf_paths: list[Path]) -> dict[tuple[str, str], int]:
    """Read exact-version exon lengths with unambiguous stable-ID fallbacks."""
    exact_lengths: dict[tuple[str, str], int] = {}
    stable_candidates: dict[tuple[str, str], set[int]] = {}
    for path in gtf_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with _open_text(path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9 or fields[2] != "exon":
                    continue
                attrs = dict(ATTRIBUTE_RE.findall(fields[8]))
                transcript_id = attrs.get("transcript_id", "")
                exon_id = attrs.get("exon_id", "")
                if not transcript_id or not exon_id:
                    continue
                length = int(fields[4]) - int(fields[3]) + 1
                key = (transcript_id, exon_id)
                previous = exact_lengths.get(key)
                if previous is not None and previous != length:
                    raise ValueError(f"Conflicting exon length for {key}: {previous} versus {length}")
                exact_lengths[key] = length
                stable_candidates.setdefault((_stable(transcript_id), _stable(exon_id)), set()).add(length)
    lengths = dict(exact_lengths)
    for stable_key, candidates in stable_candidates.items():
        if len(candidates) == 1:
            lengths[(f"@stable:{stable_key[0]}", f"@stable:{stable_key[1]}")] = next(iter(candidates))
    return lengths


def build_structural_context(
    memberships_path: Path,
    catalog_path: Path,
    gtf_paths: list[Path],
) -> pd.DataFrame:
    memberships = pd.read_csv(memberships_path, sep="\t", low_memory=False)
    required = {
        "physical_site_id", "transcript_id", "transcript_id_stable", "exon_id",
        "intron_ordinal", "intron_start_1based", "intron_end_1based",
    }
    missing = required - set(memberships.columns)
    if missing:
        raise ValueError(f"Membership table is missing columns: {sorted(missing)}")
    memberships["transcript_exact"] = memberships["transcript_id"].astype(str)
    memberships["transcript_stable"] = memberships["transcript_id_stable"].fillna(
        memberships["transcript_exact"].map(_stable)
    )
    memberships["exon_exact"] = memberships["exon_id"].astype(str)
    memberships["exon_stable"] = memberships["exon_id"].map(_stable)
    memberships = memberships.drop_duplicates(
        ["physical_site_id", "transcript_exact", "exon_exact", "intron_ordinal"]
    ).copy()
    memberships["intron_ordinal"] = pd.to_numeric(memberships["intron_ordinal"], errors="coerce")
    memberships["intron_length"] = (
        pd.to_numeric(memberships["intron_end_1based"], errors="coerce")
        - pd.to_numeric(memberships["intron_start_1based"], errors="coerce")
        + 1
    )
    transcript_totals = memberships.groupby("transcript_exact")["intron_ordinal"].max()
    memberships["transcript_intron_count"] = memberships["transcript_exact"].map(transcript_totals)
    memberships["relative_intron_rank"] = memberships["intron_ordinal"] / memberships["transcript_intron_count"]
    memberships["is_first_intron"] = (memberships["intron_ordinal"] == 1).astype(float)
    memberships["is_last_intron"] = (
        memberships["intron_ordinal"] == memberships["transcript_intron_count"]
    ).astype(float)
    exon_lengths = read_exon_lengths(gtf_paths)
    def lookup_exon_length(row) -> float:
        exact = exon_lengths.get((row.transcript_exact, row.exon_exact))
        if exact is not None:
            return float(exact)
        return float(
            exon_lengths.get(
                (f"@stable:{row.transcript_stable}", f"@stable:{row.exon_stable}"),
                np.nan,
            )
        )

    memberships["exon_length"] = [lookup_exon_length(row) for row in memberships.itertuples()]
    structural = memberships.groupby("physical_site_id", as_index=False).agg(
        exon_length_median=("exon_length", "median"),
        intron_length_median=("intron_length", "median"),
        relative_intron_rank_median=("relative_intron_rank", "median"),
        first_intron_fraction=("is_first_intron", "mean"),
        last_intron_fraction=("is_last_intron", "mean"),
        structural_transcript_count=("transcript_exact", "nunique"),
    )
    catalog = pd.read_csv(
        catalog_path,
        sep="\t",
        usecols=["physical_site_id", "transcript_count", "membership_count"],
    )
    result = catalog.merge(structural, on="physical_site_id", how="left", validate="one_to_one")
    result["transcript_context_available"] = result[
        ["exon_length_median", "intron_length_median", "relative_intron_rank_median"]
    ].notna().all(axis=1).astype(float)
    return result


def event_context_features(
    fasta: IndexedFasta,
    *,
    chrom: str,
    variant_pos_1based: int,
    anchor_1based: int,
    strand: str,
    site_type: str,
    signed_distance: int,
    structural_context: dict | None,
    variant_gc_length: int = 51,
    site_gc_length: int = 101,
) -> dict[str, float]:
    variant_radius = variant_gc_length // 2
    site_radius = site_gc_length // 2
    variant_sequence = fasta.fetch_padded(
        chrom, int(variant_pos_1based) - variant_radius, variant_gc_length
    )
    site_sequence = fasta.transcript_window(
        chrom, int(anchor_1based), strand, -site_radius, site_radius
    )
    if site_type == "donor":
        is_exonic = int(signed_distance) <= 0
    elif site_type == "acceptor":
        is_exonic = int(signed_distance) >= 0
    else:
        raise ValueError(f"Unsupported site_type: {site_type}")
    context = structural_context or {}
    exon_length = float(context.get("exon_length_median", float("nan")))
    intron_length = float(context.get("intron_length_median", float("nan")))
    transcript_count = float(context.get("transcript_count", float("nan")))
    membership_count = float(context.get("membership_count", float("nan")))

    def log1p_or_nan(value: float) -> float:
        return math.log1p(value) if math.isfinite(value) and value >= 0 else float("nan")

    return {
        "signed_distance": float(signed_distance),
        "log1p_abs_distance": math.log1p(abs(int(signed_distance))),
        "is_exonic": float(is_exonic),
        "gc_variant_51": gc_fraction(variant_sequence),
        "gc_site_101": gc_fraction(site_sequence),
        "variant_sequence_valid_fraction": valid_fraction(variant_sequence),
        "site_sequence_valid_fraction": valid_fraction(site_sequence),
        "log1p_exon_length_median": log1p_or_nan(exon_length),
        "log1p_intron_length_median": log1p_or_nan(intron_length),
        "relative_intron_rank_median": float(context.get("relative_intron_rank_median", float("nan"))),
        "first_intron_fraction": float(context.get("first_intron_fraction", float("nan"))),
        "last_intron_fraction": float(context.get("last_intron_fraction", float("nan"))),
        "log1p_transcript_count": log1p_or_nan(transcript_count),
        "log1p_membership_count": log1p_or_nan(membership_count),
        "transcript_context_available": float(context.get("transcript_context_available", 0.0)),
    }
