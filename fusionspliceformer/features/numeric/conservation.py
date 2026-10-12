"""Optional, provenance-explicit phyloP and phastCons BigWig features.

The reader uses :mod:`pybigtools`, whose pre-built Windows wheels avoid the
native build failure encountered with pyBigWig on this project platform.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class ConservationTracks:
    def __init__(self, phylop_path: Path | None = None, phastcons_path: Path | None = None):
        self.phylop_path = Path(phylop_path) if phylop_path else None
        self.phastcons_path = Path(phastcons_path) if phastcons_path else None
        if self.phylop_path is None or self.phastcons_path is None:
            raise ValueError(
                "phyloP and phastCons BigWig tracks are required by the numeric-feature schema."
            )
        self._phylop = self._open(self.phylop_path)
        self._phastcons = self._open(self.phastcons_path)

    @staticmethod
    def _open(path: Path | None):
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Configured conservation track does not exist: {path}")
        try:
            import pybigtools
        except ImportError as exc:
            raise ImportError(
                "pybigtools is required when a conservation BigWig is configured. "
                "Install it with: python -m pip install pybigtools"
            ) from exc
        return pybigtools.open(str(path))

    def close(self) -> None:
        for handle in (self._phylop, self._phastcons):
            if handle is not None and hasattr(handle, "close"):
                handle.close()

    def __enter__(self) -> "ConservationTracks":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @staticmethod
    def _values(handle, chrom: str, start0: int, end0: int) -> list[float]:
        if handle is None:
            return []
        try:
            # ``fillna=None`` explicitly preserves missing bases as NaN and
            # avoids version-dependent pybigtools default behaviour.
            values = handle.values(chrom, max(0, start0), max(0, end0), fillna=None)
        except (RuntimeError, KeyError, ValueError, OSError):
            return []
        return [float(value) for value in values if value is not None and math.isfinite(float(value))]

    def features(self, chrom: str, variant_pos_1based: int, phastcons_length: int = 51) -> dict[str, float]:
        variant0 = int(variant_pos_1based) - 1
        phylop_values = self._values(self._phylop, chrom, variant0, variant0 + 1)
        radius = phastcons_length // 2
        phastcons_values = self._values(
            self._phastcons, chrom, variant0 - radius, variant0 + radius + 1
        )
        phylop = phylop_values[0] if phylop_values else float("nan")
        phastcons = float(np.mean(phastcons_values)) if phastcons_values else float("nan")
        return {
            "phylop_variant": phylop,
            "phylop_available": float(bool(phylop_values)),
            "phastcons_variant_51_mean": phastcons,
            "phastcons_variant_51_valid_fraction": len(phastcons_values) / phastcons_length,
            "phastcons_available": float(bool(phastcons_values)),
        }
