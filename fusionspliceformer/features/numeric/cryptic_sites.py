"""Variant-overlapping candidate splice-motif scanning."""

from __future__ import annotations

import math

from .pwm import InformationPwm
from .sequence import IndexedFasta, mutate_transcript_window, orient_allele


def candidate_anchor_offsets(variant_offset: int, pwm: InformationPwm) -> list[int]:
    """Return candidate anchors whose fixed motif window overlaps the SNV.

    Anchor zero is the annotated physical site and is excluded: it is scored by the
    canonical feature family, not relabelled as a cryptic candidate.
    """
    first = int(variant_offset) - pwm.end_offset
    last = int(variant_offset) - pwm.start_offset
    return [anchor for anchor in range(first, last + 1) if anchor != 0]


def cryptic_information_features(
    fasta: IndexedFasta,
    pwm: InformationPwm,
    *,
    chrom: str,
    anchor_1based: int,
    strand: str,
    variant_offset: int,
    alt: str,
    canonical_alt_score: float,
) -> dict[str, float]:
    candidates = candidate_anchor_offsets(int(variant_offset), pwm)
    if not candidates:
        return {
            "best_variant_site_ic_ref": float("nan"),
            "best_variant_site_ic_alt": float("nan"),
            "delta_best_variant_site_ic": float("nan"),
            "cryptic_competition_alt": float("nan"),
        }
    region_start = min(anchor + pwm.start_offset for anchor in candidates)
    region_end = max(anchor + pwm.end_offset for anchor in candidates)
    reference_region = fasta.transcript_window(
        chrom, anchor_1based, strand, region_start, region_end
    )
    alternative_region = mutate_transcript_window(
        reference_region,
        region_start,
        int(variant_offset),
        orient_allele(alt, strand),
    )
    reference_scores: list[float] = []
    alternative_scores: list[float] = []
    for candidate_anchor in candidates:
        left = candidate_anchor + pwm.start_offset - region_start
        right = left + pwm.length
        ref_score = pwm.score(reference_region[left:right])
        alt_score = pwm.score(alternative_region[left:right])
        if math.isfinite(ref_score):
            reference_scores.append(ref_score)
        if math.isfinite(alt_score):
            alternative_scores.append(alt_score)
    best_ref = max(reference_scores, default=float("nan"))
    best_alt = max(alternative_scores, default=float("nan"))
    return {
        "best_variant_site_ic_ref": best_ref,
        "best_variant_site_ic_alt": best_alt,
        "delta_best_variant_site_ic": best_alt - best_ref,
        "cryptic_competition_alt": best_alt - canonical_alt_score,
    }
