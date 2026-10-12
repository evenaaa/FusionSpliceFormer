"""GRCh38 0-based, half-open input windows centered on splice junctions."""

from __future__ import annotations

from dataclasses import dataclass


POLICY_VERSION = 1
COORDINATE_SYSTEM = "GRCh38_0_BASED_HALF_OPEN"

SITE_UPSTREAM_BP = 5000
SITE_DOWNSTREAM_BP = 5000
SITE_SEQUENCE_LENGTH = SITE_UPSTREAM_BP + SITE_DOWNSTREAM_BP

VALID_STRANDS = {"+", "-"}
VALID_SITE_TYPES = {"acceptor", "donor"}


@dataclass(frozen=True, order=True)
class GenomicWindow:
    """A genomic interval in 0-based, half-open coordinates."""

    chrom: str
    start0: int
    end0: int
    window_type: str

    def __post_init__(self):
        if not self.chrom:
            raise ValueError("Window chromosome cannot be empty")
        if self.end0 <= self.start0:
            raise ValueError(
                f"Window must have positive length: {self.chrom}:{self.start0}-{self.end0}"
            )

    @property
    def length(self) -> int:
        return self.end0 - self.start0

    @property
    def start_1based(self) -> int:
        """Start coordinate for ``IndexedFasta.fetch_padded``."""
        return self.start0 + 1


@dataclass(frozen=True)
class PhysicalSpliceSite:
    chrom: str
    anchor_1based: int
    strand: str
    site_type: str

    @property
    def junction_interbase_0based(self) -> int:
        # Donor(+) and acceptor(-) boundaries lie after their adjacent exonic
        # anchor base; the other two combinations lie before it.
        if (self.strand == "+" and self.site_type == "donor") or (
            self.strand == "-" and self.site_type == "acceptor"
        ):
            return self.anchor_1based
        return self.anchor_1based - 1


def parse_physical_site_id(value: str) -> PhysicalSpliceSite:
    parts = str(value or "").strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid physical_site_id: {value!r}")
    chrom, anchor_text, strand, site_type = parts
    try:
        anchor = int(anchor_text)
    except ValueError as exc:
        raise ValueError(f"Invalid splice-site anchor: {value!r}") from exc
    if anchor < 1:
        raise ValueError(f"Splice-site anchor must be 1-based positive: {value!r}")
    if strand not in VALID_STRANDS:
        raise ValueError(f"Unsupported splice-site strand: {value!r}")
    if site_type not in VALID_SITE_TYPES:
        raise ValueError(f"Unsupported splice-site type: {value!r}")
    return PhysicalSpliceSite(chrom, anchor, strand, site_type)


def site_window(site: PhysicalSpliceSite | str) -> GenomicWindow:
    if not isinstance(site, PhysicalSpliceSite):
        site = parse_physical_site_id(site)
    center0 = site.junction_interbase_0based
    window = GenomicWindow(
        site.chrom,
        center0 - SITE_UPSTREAM_BP,
        center0 + SITE_DOWNSTREAM_BP,
        "site",
    )
    if window.length != SITE_SEQUENCE_LENGTH:
        raise AssertionError("Site-window length policy drifted")
    return window


def event_site_window(row: dict[str, object]) -> GenomicWindow:
    site = parse_physical_site_id(str(row.get("physical_site_id", "")))
    chrom = str(row.get("grch38_chrom", ""))
    if not chrom:
        raise ValueError("Event is missing grch38_chrom")
    if site.chrom != chrom:
        raise ValueError(
            f"Variant/site chromosome mismatch for {row.get('event_key', '')}: "
            f"{chrom} != {site.chrom}"
        )
    row_strand = str(row.get("strand", "") or "")
    row_site_type = str(row.get("site_type", "") or "")
    if row_strand and row_strand != site.strand:
        raise ValueError("Event strand disagrees with physical_site_id")
    if row_site_type and row_site_type != site.site_type:
        raise ValueError("Event site_type disagrees with physical_site_id")
    position = int(row['grch38_pos'])
    if position < 1:
        raise ValueError('Variant position must be a positive 1-based integer')
    return site_window(site)


def policy_manifest() -> dict[str, object]:
    return {
        "policy_version": POLICY_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "site_upstream_bp": SITE_UPSTREAM_BP,
        "site_downstream_bp": SITE_DOWNSTREAM_BP,
        "site_sequence_length": SITE_SEQUENCE_LENGTH,
        "site_anchor": "junction_interbase_0based",
    }
