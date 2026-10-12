"""Numerical feature definitions and machine-readable documentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


IDENTITY_COLUMNS = (
    "event_group_id",
    "event_key",
    "variant_key",
    "physical_site_id",
    "site_type",
    "strand",
)

REQUIRED_EVENT_COLUMNS = (
    "event_group_id",
    "event_key",
    "variant_key",
    "grch38_chrom",
    "grch38_pos",
    "grch38_ref",
    "grch38_alt",
    "physical_site_id",
    "strand",
    "site_type",
    "transcript_oriented_distance",
    "absolute_distance",
)

FORBIDDEN_OUTPUT_COLUMNS = {
    "target_label",
    "target_label_name",
    "final_unified_label",
    "functional_resolution",
    "clinical_resolution",
    "label_basis",
    "cohort_role",
    "source_file_ids",
    "evidence_tiers",
    "evidence_role",
    "role_use_status",
    "role_label_hint",
    "role_reason",
    "target_name",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    regions: tuple[str, ...]
    description: str
    missing: str = "NaN plus an explicit availability indicator where applicable"


SHARED_FEATURES = (
    FeatureSpec("canonical_ic_ref", "information_content", ("acceptor", "donor"), "Reference annotated-site PWM information content."),
    FeatureSpec("canonical_ic_alt", "information_content", ("acceptor", "donor"), "Alternative annotated-site PWM information content."),
    FeatureSpec("delta_canonical_ic", "information_content", ("acceptor", "donor"), "ALT minus REF annotated-site information content."),
    FeatureSpec("best_variant_site_ic_ref", "cryptic", ("acceptor", "donor"), "Best reference PWM score among non-annotated motif windows overlapping the SNV."),
    FeatureSpec("best_variant_site_ic_alt", "cryptic", ("acceptor", "donor"), "Best alternative PWM score among non-annotated motif windows overlapping the SNV."),
    FeatureSpec("delta_best_variant_site_ic", "cryptic", ("acceptor", "donor"), "ALT minus REF best variant-overlapping motif score."),
    FeatureSpec("cryptic_competition_alt", "cryptic", ("acceptor", "donor"), "Best alternative variant-overlapping score minus alternative annotated-site score."),
    FeatureSpec("signed_distance", "distance", ("acceptor", "donor"), "Variant position minus site anchor in transcript orientation."),
    FeatureSpec("log1p_abs_distance", "distance", ("acceptor", "donor"), "Natural log of one plus absolute transcript-oriented distance."),
    FeatureSpec("is_exonic", "distance", ("acceptor", "donor"), "One when the variant lies on the exonic side of the site anchor."),
    FeatureSpec("gc_variant_51", "sequence_context", ("acceptor", "donor"), "GC fraction in the 51-nt variant-centred window."),
    FeatureSpec("gc_site_101", "sequence_context", ("acceptor", "donor"), "GC fraction in the 101-nt site-centred window."),
    FeatureSpec("variant_sequence_valid_fraction", "sequence_context", ("acceptor", "donor"), "ACGT fraction in the variant-centred window."),
    FeatureSpec("site_sequence_valid_fraction", "sequence_context", ("acceptor", "donor"), "ACGT fraction in the site-centred window."),
    FeatureSpec("phylop_variant", "conservation", ("acceptor", "donor"), "phyloP score at the variant base."),
    FeatureSpec("phylop_available", "missingness", ("acceptor", "donor"), "One when a finite phyloP value is available."),
    FeatureSpec("phastcons_variant_51_mean", "conservation", ("acceptor", "donor"), "Mean finite phastCons score in a 51-nt variant-centred window."),
    FeatureSpec("phastcons_variant_51_valid_fraction", "conservation", ("acceptor", "donor"), "Fraction of positions with finite phastCons values."),
    FeatureSpec("phastcons_available", "missingness", ("acceptor", "donor"), "One when at least one finite phastCons value is available."),
    FeatureSpec("log1p_exon_length_median", "transcript_context", ("acceptor", "donor"), "Log-transformed median adjacent exon length across unique transcripts."),
    FeatureSpec("log1p_intron_length_median", "transcript_context", ("acceptor", "donor"), "Log-transformed median adjacent intron length across unique transcripts."),
    FeatureSpec("relative_intron_rank_median", "transcript_context", ("acceptor", "donor"), "Median intron ordinal divided by transcript intron count."),
    FeatureSpec("first_intron_fraction", "transcript_context", ("acceptor", "donor"), "Fraction of unique transcript memberships in which this is the first intron."),
    FeatureSpec("last_intron_fraction", "transcript_context", ("acceptor", "donor"), "Fraction of unique transcript memberships in which this is the last intron."),
    FeatureSpec("log1p_transcript_count", "annotation_support", ("acceptor", "donor"), "Log-transformed unique transcript count in the site catalog."),
    FeatureSpec("log1p_membership_count", "annotation_support", ("acceptor", "donor"), "Log-transformed annotation membership count in the site catalog."),
    FeatureSpec("transcript_context_available", "missingness", ("acceptor", "donor"), "One when structural transcript aggregation is available."),
)

ACCEPTOR_FEATURES = (
    FeatureSpec("creates_ag_in_agez", "acceptor_agez", ("acceptor",), "SNV creates at least one AG in integer offsets -50..-3."),
    FeatureSpec("destroys_ag_in_agez", "acceptor_agez", ("acceptor",), "SNV destroys at least one AG in integer offsets -50..-3."),
    FeatureSpec("creates_yag_in_agez", "acceptor_agez", ("acceptor",), "SNV creates at least one CAG or TAG in the AGEZ."),
    FeatureSpec("destroys_yag_in_agez", "acceptor_agez", ("acceptor",), "SNV destroys at least one CAG or TAG in the AGEZ."),
    FeatureSpec("changes_acceptor_minus3_y_to_r", "acceptor_agez", ("acceptor",), "At offset -3, the SNV changes C/T to A/G."),
    FeatureSpec("ppt_pyrimidine_fraction_ref", "acceptor_ppt", ("acceptor",), "Reference C/T fraction in integer offsets -50..-3."),
    FeatureSpec("ppt_pyrimidine_fraction_alt", "acceptor_ppt", ("acceptor",), "Alternative C/T fraction in integer offsets -50..-3."),
    FeatureSpec("delta_ppt_pyrimidine_fraction", "acceptor_ppt", ("acceptor",), "ALT minus REF AGEZ pyrimidine fraction."),
    FeatureSpec("ppt_longest_run_ref", "acceptor_ppt", ("acceptor",), "Longest uninterrupted reference C/T run in the AGEZ."),
    FeatureSpec("ppt_longest_run_alt", "acceptor_ppt", ("acceptor",), "Longest uninterrupted alternative C/T run in the AGEZ."),
    FeatureSpec("delta_ppt_longest_run", "acceptor_ppt", ("acceptor",), "ALT minus REF longest AGEZ pyrimidine run."),
)

ALL_SPECS = SHARED_FEATURES + ACCEPTOR_FEATURES
ALL_FEATURE_NAMES = tuple(spec.name for spec in ALL_SPECS)
DONOR_FEATURE_NAMES = tuple(spec.name for spec in SHARED_FEATURES)
ACCEPTOR_FEATURE_NAMES = ALL_FEATURE_NAMES
ACCEPTOR_ONLY_FEATURE_NAMES = tuple(spec.name for spec in ACCEPTOR_FEATURES)


def write_schema(path: Path, schema_version: str) -> None:
    payload = {
        "schema_version": schema_version,
        "delta_convention": "ALT_MINUS_REF",
        "identity_columns": list(IDENTITY_COLUMNS),
        "donor_feature_order": list(DONOR_FEATURE_NAMES),
        "acceptor_feature_order": list(ACCEPTOR_FEATURE_NAMES),
        "features": [asdict(spec) for spec in ALL_SPECS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
