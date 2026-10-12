"""Independent splice-site position-weight matrix construction and scoring."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import FeatureSettings
from .sequence import IndexedFasta


BASES = ("A", "C", "G", "T")
BASE_INDEX = {base: index for index, base in enumerate(BASES)}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InformationPwm:
    site_type: str
    start_offset: int
    end_offset: int
    counts: np.ndarray
    probabilities: np.ndarray
    information: np.ndarray
    pseudocount: float
    site_count: int

    @property
    def length(self) -> int:
        return self.end_offset - self.start_offset + 1

    def score(self, sequence: str) -> float:
        sequence = sequence.upper()
        if len(sequence) != self.length or any(base not in BASE_INDEX for base in sequence):
            return float("nan")
        return float(sum(self.information[index, BASE_INDEX[base]] for index, base in enumerate(sequence)))

    def to_dict(self) -> dict:
        return {
            "site_type": self.site_type,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "base_order": list(BASES),
            "counts": self.counts.tolist(),
            "probabilities": self.probabilities.tolist(),
            "information_bits": self.information.tolist(),
            "pseudocount": self.pseudocount,
            "site_count": self.site_count,
            "formula": "log2(p(base, position) / 0.25)",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InformationPwm":
        if payload.get("base_order") != list(BASES):
            raise ValueError("PWM base order must be A,C,G,T")
        return cls(
            site_type=payload["site_type"],
            start_offset=int(payload["start_offset"]),
            end_offset=int(payload["end_offset"]),
            counts=np.asarray(payload["counts"], dtype=float),
            probabilities=np.asarray(payload["probabilities"], dtype=float),
            information=np.asarray(payload["information_bits"], dtype=float),
            pseudocount=float(payload["pseudocount"]),
            site_count=int(payload["site_count"]),
        )


def pwm_from_sequences(
    sequences: Iterable[str], site_type: str, offsets: tuple[int, int], pseudocount: float
) -> InformationPwm:
    start, end = offsets
    length = end - start + 1
    counts = np.full((length, 4), pseudocount, dtype=float)
    site_count = 0
    for sequence in sequences:
        sequence = sequence.upper()
        if len(sequence) != length or any(base not in BASE_INDEX for base in sequence):
            continue
        for index, base in enumerate(sequence):
            counts[index, BASE_INDEX[base]] += 1.0
        site_count += 1
    if site_count == 0:
        raise ValueError(f"No valid {site_type} sequences were supplied")
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    information = np.log2(probabilities / 0.25)
    return InformationPwm(site_type, start, end, counts, probabilities, information, pseudocount, site_count)


def build_pwms(
    catalog_path: Path,
    fasta_path: Path,
    settings: FeatureSettings,
) -> tuple[dict[str, InformationPwm], dict]:
    settings.validate()
    sequences: dict[str, list[str]] = {"donor": [], "acceptor": []}
    rejected: dict[str, int] = {}
    seen: set[str] = set()
    with IndexedFasta(fasta_path) as fasta, gzip.open(catalog_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            site_id = row["physical_site_id"]
            if site_id in seen:
                rejected["duplicate_physical_site"] = rejected.get("duplicate_physical_site", 0) + 1
                continue
            seen.add(site_id)
            if row.get("motif_status") != "CANONICAL":
                rejected["noncanonical_motif"] = rejected.get("noncanonical_motif", 0) + 1
                continue
            if str(row.get("default_mapping_eligible", "")).lower() != "true":
                rejected["not_default_mapping_eligible"] = rejected.get("not_default_mapping_eligible", 0) + 1
                continue
            site_type = row["site_type"]
            start, end = settings.motif_offsets(site_type)
            sequence = fasta.transcript_window(
                row["chrom"], int(row["site_anchor_pos_1based"]), row["strand"], start, end
            )
            if any(base not in BASE_INDEX for base in sequence):
                rejected[f"invalid_sequence_{site_type}"] = rejected.get(f"invalid_sequence_{site_type}", 0) + 1
                continue
            sequences[site_type].append(sequence)
    pwms = {
        site_type: pwm_from_sequences(
            sequence_list,
            site_type,
            settings.motif_offsets(site_type),
            settings.pwm_pseudocount,
        )
        for site_type, sequence_list in sequences.items()
    }
    metadata = {
        "schema_version": settings.schema_version,
        "reference_build": settings.reference_build,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "fasta_path": str(fasta_path.resolve()),
        "fasta_sha256": sha256_file(fasta_path),
        "unique_physical_sites_seen": len(seen),
        "retained_site_counts": {key: value.site_count for key, value in pwms.items()},
        "rejected_counts": rejected,
        "filters": ["unique physical_site_id", "motif_status=CANONICAL", "default_mapping_eligible=true", "ACGT-only window"],
        "label_independent": True,
    }
    return pwms, metadata


def save_pwm(path: Path, pwm: InformationPwm) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pwm.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_pwm(path: Path) -> InformationPwm:
    return InformationPwm.from_dict(json.loads(path.read_text(encoding="utf-8")))
