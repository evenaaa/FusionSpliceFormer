"""Indexed FASTA access and transcript-oriented SNV sequence construction."""

from __future__ import annotations

from pathlib import Path


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
VALID_BASES = frozenset("ACGT")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1].upper()


def orient_allele(allele: str, strand: str) -> str:
    allele = str(allele).upper()
    if len(allele) != 1 or allele not in VALID_BASES:
        raise ValueError(f"Only A/C/G/T SNV alleles are supported: {allele!r}")
    if strand == "+":
        return allele
    if strand == "-":
        return reverse_complement(allele)
    raise ValueError(f"Unsupported strand: {strand!r}")


class IndexedFasta:
    """Minimal random-access reader for an uncompressed FASTA plus .fai."""

    def __init__(self, fasta_path: Path | str):
        self.fasta_path = Path(fasta_path)
        self.fai_path = Path(str(self.fasta_path) + ".fai")
        if not self.fasta_path.is_file() or not self.fai_path.is_file():
            raise FileNotFoundError(
                f"Both uncompressed FASTA and index are required: {self.fasta_path}, {self.fai_path}"
            )
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with self.fai_path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) >= 5:
                    self.index[fields[0]] = tuple(map(int, fields[1:5]))
        self._handle = self.fasta_path.open("rb")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "IndexedFasta":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def chromosome_length(self, chrom: str) -> int:
        if chrom not in self.index:
            raise KeyError(f"Chromosome is absent from FASTA index: {chrom}")
        return self.index[chrom][0]

    def fetch(self, chrom: str, start_1based: int, length: int) -> str:
        if chrom not in self.index or length <= 0:
            return ""
        chrom_length, offset, line_bases, line_width = self.index[chrom]
        if start_1based < 1 or start_1based + length - 1 > chrom_length:
            return ""
        zero_based = start_1based - 1
        byte_offset = offset + (zero_based // line_bases) * line_width + (zero_based % line_bases)
        self._handle.seek(byte_offset)
        pieces: list[bytes] = []
        remaining = length
        while remaining:
            chunk = self._handle.read(max(remaining + 32, 128))
            if not chunk:
                break
            sequence = chunk.replace(b"\n", b"").replace(b"\r", b"")
            take = min(remaining, len(sequence))
            pieces.append(sequence[:take])
            remaining -= take
        result = b"".join(pieces).decode("ascii").upper()
        return result if len(result) == length else ""

    def fetch_padded(self, chrom: str, start_1based: int, length: int) -> str:
        if length <= 0:
            return ""
        chrom_length = self.chromosome_length(chrom)
        requested_end = start_1based + length - 1
        left_pad = max(0, 1 - start_1based)
        right_pad = max(0, requested_end - chrom_length)
        valid_start = max(1, start_1based)
        valid_end = min(chrom_length, requested_end)
        middle = self.fetch(chrom, valid_start, max(0, valid_end - valid_start + 1))
        return "N" * left_pad + middle + "N" * right_pad

    def base(self, chrom: str, pos_1based: int) -> str:
        return self.fetch(chrom, pos_1based, 1)

    def transcript_window(
        self,
        chrom: str,
        anchor_1based: int,
        strand: str,
        start_offset: int,
        end_offset: int,
    ) -> str:
        if end_offset < start_offset:
            raise ValueError("end_offset must be >= start_offset")
        length = end_offset - start_offset + 1
        if strand == "+":
            return self.fetch_padded(chrom, anchor_1based + start_offset, length)
        if strand == "-":
            genomic_start = anchor_1based - end_offset
            return reverse_complement(self.fetch_padded(chrom, genomic_start, length))
        raise ValueError(f"Unsupported strand: {strand!r}")


def mutate_transcript_window(
    reference_sequence: str,
    window_start_offset: int,
    variant_offset: int,
    oriented_alt: str,
) -> str:
    index = variant_offset - window_start_offset
    if not 0 <= index < len(reference_sequence):
        return reference_sequence
    return reference_sequence[:index] + oriented_alt + reference_sequence[index + 1 :]


def valid_fraction(sequence: str) -> float:
    if not sequence:
        return float("nan")
    return sum(base in VALID_BASES for base in sequence.upper()) / len(sequence)


def gc_fraction(sequence: str) -> float:
    sequence = sequence.upper()
    valid = [base for base in sequence if base in VALID_BASES]
    if not valid:
        return float("nan")
    return sum(base in {"G", "C"} for base in valid) / len(valid)
