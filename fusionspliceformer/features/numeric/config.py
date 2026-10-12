"""Configuration and default project paths for numeric feature generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path.cwd()


@dataclass(frozen=True)
class FeaturePaths:
    project_root: Path = PROJECT_ROOT
    fasta: Path = PROJECT_ROOT / "dataset/annodata/fa/GRCh38.primary_assembly.genome.fa"
    site_catalog: Path = PROJECT_ROOT / "dataset/interim/splice_site_catalog/curated_splice_sites.grch38.tsv.gz"
    memberships: Path = PROJECT_ROOT / "dataset/interim/splice_site_catalog/splice_site_memberships.grch38.tsv.gz"
    mane_gtf: Path = PROJECT_ROOT / "dataset/annodata/gtf/MANE.GRCh38.v1.4.ensembl_genomic.gtf"
    gencode_gtf: Path = PROJECT_ROOT / "dataset/annodata/gtf/gencode.v47.basic.annotation.gtf"
    asset_dir: Path = PROJECT_ROOT / "dataset/interim/feature_assets"
    phylop_bigwig: Path = PROJECT_ROOT / "dataset/annodata/conservation/hg38.phyloP100way.bw"
    phastcons_bigwig: Path = PROJECT_ROOT / "dataset/annodata/conservation/hg38.phastCons100way.bw"

    def resolved(self) -> "FeaturePaths":
        values = {}
        for key, value in asdict(self).items():
            values[key] = None if value is None else Path(value).resolve()
        return FeaturePaths(**values)


@dataclass(frozen=True)
class FeatureSettings:
    schema_version: str = "numeric_features_v1.0.0"
    reference_build: str = "GRCh38"
    pwm_pseudocount: float = 0.5
    donor_offsets: tuple[int, int] = (-2, 6)
    acceptor_offsets: tuple[int, int] = (-25, 1)
    agez_offsets: tuple[int, int] = (-50, -3)
    variant_gc_length: int = 51
    site_gc_length: int = 101
    phastcons_length: int = 51
    minimum_conservation_coverage: float = 0.99
    accepted_bases: tuple[str, ...] = field(default=("A", "C", "G", "T"))

    def motif_offsets(self, site_type: str) -> tuple[int, int]:
        if site_type == "donor":
            return self.donor_offsets
        if site_type == "acceptor":
            return self.acceptor_offsets
        raise ValueError(f"Unsupported site_type: {site_type!r}")

    def validate(self) -> None:
        if self.donor_offsets[1] - self.donor_offsets[0] + 1 != 9:
            raise ValueError("Donor PWM window must contain 9 nt")
        if self.acceptor_offsets[1] - self.acceptor_offsets[0] + 1 != 27:
            raise ValueError("Acceptor PWM window must contain 27 nt")
        if self.agez_offsets != (-50, -3):
            raise ValueError("AGEZ offsets must be integers from -50 to -3")
        for name in ("variant_gc_length", "site_gc_length", "phastcons_length"):
            if getattr(self, name) < 1 or getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.pwm_pseudocount <= 0:
            raise ValueError("PWM pseudocount must be positive")
        if not 0 < self.minimum_conservation_coverage <= 1:
            raise ValueError("minimum_conservation_coverage must be in (0, 1]")
