"""Unified label-independent event-level numeric feature pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .acceptor_features import acceptor_sequence_features
from .config import FeaturePaths, FeatureSettings
from .conservation import ConservationTracks
from .context_features import event_context_features
from .cryptic_sites import cryptic_information_features
from .information_content import canonical_information_features
from .pwm import load_pwm
from .schema import ACCEPTOR_ONLY_FEATURE_NAMES, ALL_FEATURE_NAMES, IDENTITY_COLUMNS, REQUIRED_EVENT_COLUMNS
from .sequence import IndexedFasta


class NumericFeaturePipeline:
    def __init__(
        self,
        paths: FeaturePaths | None = None,
        settings: FeatureSettings | None = None,
    ):
        self.paths = (paths or FeaturePaths()).resolved()
        self.settings = settings or FeatureSettings()
        self.settings.validate()
        self.catalog = self._load_catalog(self.paths.site_catalog)
        self.structural_context = self._load_structural_context(
            self.paths.asset_dir / "structural_context.grch38.tsv.gz"
        )
        self.pwms = {
            site_type: load_pwm(self.paths.asset_dir / f"{site_type}_pwm.grch38.json")
            for site_type in ("donor", "acceptor")
        }

    @staticmethod
    def _load_catalog(path: Path) -> dict[str, dict]:
        columns = [
            "physical_site_id", "chrom", "site_anchor_pos_1based", "strand", "site_type",
            "transcript_count", "membership_count",
        ]
        frame = pd.read_csv(path, sep="\t", usecols=columns)
        if frame["physical_site_id"].duplicated().any():
            raise ValueError("Splice-site catalog contains duplicate physical_site_id values")
        return frame.set_index("physical_site_id").to_dict("index")

    @staticmethod
    def _load_structural_context(path: Path) -> dict[str, dict]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Structural-context asset is missing: {path}. Run prepare_references.py; see README.md (Prepare references)."
            )
        frame = pd.read_csv(path, sep="\t")
        if frame["physical_site_id"].duplicated().any():
            raise ValueError("Structural-context asset contains duplicate physical_site_id values")
        return frame.set_index("physical_site_id").to_dict("index")

    def _validate_event(self, row: dict, site: dict) -> None:
        site_id = row["physical_site_id"]
        for event_key, site_key in (
            ("grch38_chrom", "chrom"), ("strand", "strand"), ("site_type", "site_type")
        ):
            if str(row[event_key]) != str(site[site_key]):
                raise ValueError(
                    f"Event/catalog mismatch for {site_id}: {event_key}={row[event_key]!r}, "
                    f"catalog={site[site_key]!r}"
                )
        sign = 1 if row["strand"] == "+" else -1
        expected_position = int(site["site_anchor_pos_1based"]) + sign * int(row["transcript_oriented_distance"])
        if expected_position != int(row["grch38_pos"]):
            raise ValueError(
                f"Distance/coordinate mismatch for {row['event_key']}: expected {expected_position}, "
                f"observed {row['grch38_pos']}"
            )
        if abs(int(row["transcript_oriented_distance"])) != int(row["absolute_distance"]):
            raise ValueError(f"absolute_distance mismatch for {row['event_key']}")

    def build(self, events: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        missing = sorted(set(REQUIRED_EVENT_COLUMNS) - set(events.columns))
        if missing:
            raise ValueError(f"Event table is missing required columns: {missing}")
        if events["event_group_id"].duplicated().any():
            raise ValueError("Input contains duplicate event_group_id values")
        output_rows: list[dict] = []
        with IndexedFasta(self.paths.fasta) as fasta, ConservationTracks(
            self.paths.phylop_bigwig, self.paths.phastcons_bigwig
        ) as conservation:
            for row in events.to_dict("records"):
                site_id = str(row["physical_site_id"])
                if site_id not in self.catalog:
                    raise KeyError(f"Physical site absent from catalog: {site_id}")
                site = self.catalog[site_id]
                self._validate_event(row, site)
                site_type = str(row["site_type"])
                pwm = self.pwms[site_type]
                anchor = int(site["site_anchor_pos_1based"])
                distance = int(row["transcript_oriented_distance"])
                result = {column: row.get(column, "") for column in IDENTITY_COLUMNS}
                canonical = canonical_information_features(
                    fasta,
                    pwm,
                    chrom=str(row["grch38_chrom"]),
                    anchor_1based=anchor,
                    strand=str(row["strand"]),
                    variant_offset=distance,
                    ref=str(row["grch38_ref"]),
                    alt=str(row["grch38_alt"]),
                )
                result.update(canonical)
                result.update(
                    cryptic_information_features(
                        fasta,
                        pwm,
                        chrom=str(row["grch38_chrom"]),
                        anchor_1based=anchor,
                        strand=str(row["strand"]),
                        variant_offset=distance,
                        alt=str(row["grch38_alt"]),
                        canonical_alt_score=canonical["canonical_ic_alt"],
                    )
                )
                structural = dict(self.structural_context.get(site_id, {}))
                structural.setdefault("transcript_count", site.get("transcript_count"))
                structural.setdefault("membership_count", site.get("membership_count"))
                result.update(
                    event_context_features(
                        fasta,
                        chrom=str(row["grch38_chrom"]),
                        variant_pos_1based=int(row["grch38_pos"]),
                        anchor_1based=anchor,
                        strand=str(row["strand"]),
                        site_type=site_type,
                        signed_distance=distance,
                        structural_context=structural,
                        variant_gc_length=self.settings.variant_gc_length,
                        site_gc_length=self.settings.site_gc_length,
                    )
                )
                result.update(
                    conservation.features(
                        str(row["grch38_chrom"]),
                        int(row["grch38_pos"]),
                        self.settings.phastcons_length,
                    )
                )
                if site_type == "acceptor":
                    result.update(
                        acceptor_sequence_features(
                            fasta,
                            chrom=str(row["grch38_chrom"]),
                            anchor_1based=anchor,
                            strand=str(row["strand"]),
                            variant_offset=distance,
                            ref=str(row["grch38_ref"]),
                            alt=str(row["grch38_alt"]),
                            agez_offsets=self.settings.agez_offsets,
                        )
                    )
                else:
                    for name in ACCEPTOR_ONLY_FEATURE_NAMES:
                        result[name] = float("nan")
                output_rows.append(result)
                if progress_callback:
                    progress_callback(len(output_rows))
        output = pd.DataFrame(output_rows)
        ordered_identity = [column for column in IDENTITY_COLUMNS if column in output.columns]
        output = output[ordered_identity + list(ALL_FEATURE_NAMES)]
        return output
