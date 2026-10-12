"""Acceptor-specific AG-exclusion-zone and pyrimidine-tract features."""

from __future__ import annotations

from .sequence import IndexedFasta, mutate_transcript_window, orient_allele


PYRIMIDINES = frozenset("CT")
PURINES = frozenset("AG")


def _motif_starts(sequence: str, motifs: set[str]) -> set[int]:
    lengths = {len(motif) for motif in motifs}
    if len(lengths) != 1:
        raise ValueError("All motifs must have equal length")
    length = lengths.pop()
    return {
        index
        for index in range(0, len(sequence) - length + 1)
        if sequence[index : index + length] in motifs
    }


def _pyrimidine_fraction(sequence: str) -> float:
    valid = [base for base in sequence if base in "ACGT"]
    if not valid:
        return float("nan")
    return sum(base in PYRIMIDINES for base in valid) / len(valid)


def _longest_pyrimidine_run(sequence: str) -> int:
    longest = current = 0
    for base in sequence:
        if base in PYRIMIDINES:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def acceptor_sequence_features(
    fasta: IndexedFasta,
    *,
    chrom: str,
    anchor_1based: int,
    strand: str,
    variant_offset: int,
    ref: str,
    alt: str,
    agez_offsets: tuple[int, int] = (-50, -3),
) -> dict[str, float]:
    start, end = agez_offsets
    reference = fasta.transcript_window(chrom, anchor_1based, strand, start, end)
    oriented_alt = orient_allele(alt, strand)
    oriented_ref = orient_allele(ref, strand)
    alternative = mutate_transcript_window(reference, start, int(variant_offset), oriented_alt)
    ref_ag = _motif_starts(reference, {"AG"})
    alt_ag = _motif_starts(alternative, {"AG"})
    ref_yag = _motif_starts(reference, {"CAG", "TAG"})
    alt_yag = _motif_starts(alternative, {"CAG", "TAG"})
    ref_fraction = _pyrimidine_fraction(reference)
    alt_fraction = _pyrimidine_fraction(alternative)
    ref_run = _longest_pyrimidine_run(reference)
    alt_run = _longest_pyrimidine_run(alternative)
    return {
        "creates_ag_in_agez": float(bool(alt_ag - ref_ag)),
        "destroys_ag_in_agez": float(bool(ref_ag - alt_ag)),
        "creates_yag_in_agez": float(bool(alt_yag - ref_yag)),
        "destroys_yag_in_agez": float(bool(ref_yag - alt_yag)),
        "changes_acceptor_minus3_y_to_r": float(
            int(variant_offset) == -3 and oriented_ref in PYRIMIDINES and oriented_alt in PURINES
        ),
        "ppt_pyrimidine_fraction_ref": ref_fraction,
        "ppt_pyrimidine_fraction_alt": alt_fraction,
        "delta_ppt_pyrimidine_fraction": alt_fraction - ref_fraction,
        "ppt_longest_run_ref": float(ref_run),
        "ppt_longest_run_alt": float(alt_run),
        "delta_ppt_longest_run": float(alt_run - ref_run),
    }
