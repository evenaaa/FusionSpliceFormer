"""Annotated splice-site information-content features."""

from __future__ import annotations

import math

from .pwm import InformationPwm
from .sequence import IndexedFasta, mutate_transcript_window, orient_allele


def canonical_information_features(
    fasta: IndexedFasta,
    pwm: InformationPwm,
    *,
    chrom: str,
    anchor_1based: int,
    strand: str,
    variant_offset: int,
    ref: str,
    alt: str,
) -> dict[str, float]:
    """Score the annotated motif on REF and ALT using one fixed PWM."""
    genomic_ref = fasta.base(chrom, anchor_1based + variant_offset * (1 if strand == "+" else -1))
    if genomic_ref != str(ref).upper():
        raise ValueError(
            f"GRCh38 REF mismatch at {chrom}:{anchor_1based + variant_offset * (1 if strand == '+' else -1)}; "
            f"manifest={ref}, FASTA={genomic_ref}"
        )
    reference_sequence = fasta.transcript_window(
        chrom, anchor_1based, strand, pwm.start_offset, pwm.end_offset
    )
    alternative_sequence = mutate_transcript_window(
        reference_sequence,
        pwm.start_offset,
        int(variant_offset),
        orient_allele(alt, strand),
    )
    reference_score = pwm.score(reference_sequence)
    alternative_score = pwm.score(alternative_sequence)
    delta = alternative_score - reference_score
    return {
        "canonical_ic_ref": reference_score,
        "canonical_ic_alt": alternative_score,
        "delta_canonical_ic": delta,
    }
