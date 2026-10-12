"""Build a transcript-aware, physically deduplicated GRCh38 splice-site catalog.

Reference preparation uses MANE v1.4 as the primary annotation and retains GENCODE v47 Basic
protein-coding sites as a controlled fallback.  It does not read any variant
labels and does not map variants to sites.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


PRIMARY_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
CHROM_RANK = {chrom: index for index, chrom in enumerate(PRIMARY_CHROMS, start=1)}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

MEMBERSHIP_FIELDS = [
    "physical_site_id",
    "chrom",
    "site_anchor_pos_1based",
    "junction_interbase_0based",
    "strand",
    "site_type",
    "gene_id",
    "gene_id_stable",
    "gene_symbol",
    "transcript_id",
    "transcript_id_stable",
    "refseq_transcript_ids",
    "exon_id",
    "intron_ordinal",
    "intron_start_1based",
    "intron_end_1based",
    "annotation_source",
    "annotation_class",
    "source_gtf",
]

CATALOG_FIELDS = [
    "physical_site_id",
    "chrom",
    "site_anchor_pos_1based",
    "junction_interbase_0based",
    "strand",
    "site_type",
    "expected_dinucleotide",
    "reference_dinucleotide",
    "motif_status",
    "catalog_tier",
    "annotation_priority",
    "default_mapping_eligible",
    "default_mapping_reason",
    "gene_ids",
    "gene_ids_stable",
    "gene_symbols",
    "transcript_ids",
    "transcript_ids_stable",
    "refseq_transcript_ids",
    "exon_ids",
    "mane_select_transcript_ids",
    "mane_plus_clinical_transcript_ids",
    "gencode_basic_transcript_ids",
    "annotation_sources",
    "source_gtf_files",
    "gene_count",
    "transcript_count",
    "membership_count",
]


def resolve_project_root(project_root=None):
    if project_root is not None:
        root = Path(project_root).resolve()
    else:
        root = Path.cwd().resolve()
        if root.name.lower() == "data_preparation":
            root = root.parent
    if not (root / "dataset").is_dir():
        raise FileNotFoundError(
            f"Cannot find the dataset directory under project root: {root}"
        )
    return root


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_identifier(value):
    value = (value or "").strip()
    return value.split(".", 1)[0] if value else ""


def normalize_chrom(value):
    value = (value or "").strip()
    if not value:
        return None
    if not value.startswith("chr"):
        value = "chr" + value
    return value if value in CHROM_RANK else None


def parse_gtf_attributes(text):
    attributes = {}
    for piece in text.rstrip().rstrip(";").split(";"):
        piece = piece.strip()
        if not piece:
            continue
        key, separator, value = piece.partition(" ")
        if not separator:
            continue
        value = value.strip().strip('"')
        attributes.setdefault(key, []).append(value)
    return attributes


def first_attribute(attributes, key, default=""):
    values = attributes.get(key)
    return values[0] if values else default


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


@contextmanager
def open_text(path):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("rt", encoding="utf-8", newline="")
    try:
        yield handle
    finally:
        handle.close()


@contextmanager
def deterministic_gzip_writer(path):
    """Atomically write UTF-8 text in a reproducible gzip container."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    raw_handle = temporary.open("wb")
    gzip_handle = gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0
    )
    text_handle = io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="")
    try:
        yield text_handle
        text_handle.flush()
        text_handle.close()
        raw_handle.close()
        os.replace(temporary, path)
    except Exception:
        try:
            text_handle.close()
        except Exception:
            pass
        try:
            raw_handle.close()
        except Exception:
            pass
        if temporary.exists():
            temporary.unlink()
        raise


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class IndexedFasta:
    """Small faidx-compatible reader for uncompressed FASTA files."""

    def __init__(self, fasta_path):
        self.fasta_path = Path(fasta_path)
        self.fai_path = Path(str(self.fasta_path) + ".fai")
        if not self.fasta_path.is_file() or not self.fai_path.is_file():
            raise FileNotFoundError(
                f"FASTA and FAI are both required: {self.fasta_path}, {self.fai_path}"
            )
        self.index = {}
        with self.fai_path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 5:
                    continue
                name, length, offset, line_bases, line_width = fields[:5]
                self.index[name] = tuple(
                    int(value) for value in (length, offset, line_bases, line_width)
                )
        self.handle = self.fasta_path.open("rb")

    def close(self):
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def fetch(self, chrom, start_1based, length):
        if chrom not in self.index or length <= 0:
            return ""
        chrom_length, offset, line_bases, line_width = self.index[chrom]
        if start_1based < 1 or start_1based + length - 1 > chrom_length:
            return ""
        zero_based = start_1based - 1
        byte_offset = (
            offset
            + (zero_based // line_bases) * line_width
            + (zero_based % line_bases)
        )
        self.handle.seek(byte_offset)
        pieces = []
        remaining = length
        while remaining > 0:
            chunk = self.handle.read(max(remaining + 16, 64))
            if not chunk:
                break
            sequence = chunk.replace(b"\n", b"").replace(b"\r", b"")
            take = min(remaining, len(sequence))
            pieces.append(sequence[:take])
            remaining -= take
        return b"".join(pieces).decode("ascii").upper()


def _annotation_class(annotation_source, attributes):
    tags = set(attributes.get("tag", []))
    transcript_type = first_attribute(attributes, "transcript_type")
    if annotation_source == "MANE_v1.4":
        if "MANE_Select" in tags:
            return "MANE_Select", None
        if "MANE_Plus_Clinical" in tags:
            return "MANE_Plus_Clinical", None
        return None, "MANE_TRANSCRIPT_WITHOUT_ELIGIBLE_TAG"
    if "basic" not in tags:
        return None, "GENCODE_TRANSCRIPT_WITHOUT_BASIC_TAG"
    if transcript_type != "protein_coding":
        return None, "GENCODE_NON_PROTEIN_CODING_TRANSCRIPT"
    return "GENCODE_Basic", None


def iter_eligible_transcripts(
    gtf_path,
    annotation_source,
    qc,
    max_transcripts=None,
):
    """Yield eligible transcript records without loading the whole GTF."""

    gtf_path = Path(gtf_path)
    current = None
    yielded = 0

    def emit_current():
        nonlocal current, yielded
        if current is None:
            return None
        record = current
        current = None
        yielded += 1
        qc["eligible_transcripts"] += 1
        return record

    with open_text(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                qc["malformed_gtf_rows"] += 1
                continue
            feature = fields[2]
            if feature == "transcript":
                record = emit_current()
                if record is not None:
                    yield record
                    if max_transcripts is not None and yielded >= max_transcripts:
                        return

                qc["transcript_rows"] += 1
                attributes = parse_gtf_attributes(fields[8])
                transcript_id = first_attribute(attributes, "transcript_id")
                chrom = normalize_chrom(fields[0])
                strand = fields[6]
                annotation_class, rejection = _annotation_class(
                    annotation_source, attributes
                )
                if not transcript_id:
                    rejection = "MISSING_TRANSCRIPT_ID"
                elif chrom is None:
                    rejection = "NON_PRIMARY_CHROM"
                elif strand not in {"+", "-"}:
                    rejection = "INVALID_STRAND"
                if rejection:
                    qc[f"rejected_transcript::{rejection}"] += 1
                    current = None
                    continue

                refseq_ids = sorted(
                    {
                        value.split(":", 1)[1]
                        for value in attributes.get("db_xref", [])
                        if value.startswith("RefSeq:") and ":" in value
                    }
                )
                current = {
                    "chrom": chrom,
                    "strand": strand,
                    "gene_id": first_attribute(attributes, "gene_id"),
                    "gene_id_stable": stable_identifier(
                        first_attribute(attributes, "gene_id")
                    ),
                    "gene_symbol": first_attribute(attributes, "gene_name"),
                    "transcript_id": transcript_id,
                    "transcript_id_stable": stable_identifier(transcript_id),
                    "refseq_transcript_ids": refseq_ids,
                    "transcript_type": first_attribute(
                        attributes, "transcript_type"
                    ),
                    "annotation_source": annotation_source,
                    "annotation_class": annotation_class,
                    "source_gtf": gtf_path.name,
                    "exons": [],
                }
                continue

            if feature != "exon" or current is None:
                continue
            attributes = parse_gtf_attributes(fields[8])
            transcript_id = first_attribute(attributes, "transcript_id")
            if transcript_id != current["transcript_id"]:
                qc["exon_transcript_mismatch"] += 1
                continue
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                qc["invalid_exon_coordinate"] += 1
                continue
            if start < 1 or end < start:
                qc["invalid_exon_coordinate"] += 1
                continue
            current["exons"].append(
                {
                    "start": start,
                    "end": end,
                    "exon_id": first_attribute(attributes, "exon_id"),
                    "exon_number": first_attribute(attributes, "exon_number"),
                }
            )

    record = emit_current()
    if record is not None:
        yield record


def _site_coordinates(strand, site_type, exon):
    if site_type == "donor":
        anchor = exon["end"] if strand == "+" else exon["start"]
    else:
        anchor = exon["start"] if strand == "+" else exon["end"]
    if (strand == "+" and site_type == "donor") or (
        strand == "-" and site_type == "acceptor"
    ):
        interbase = anchor
    else:
        interbase = anchor - 1
    return anchor, interbase


def transcript_to_memberships(transcript, qc):
    unique_exons = {}
    for exon in transcript["exons"]:
        key = (exon["start"], exon["end"], exon["exon_id"])
        unique_exons[key] = exon
    exons = sorted(
        unique_exons.values(),
        key=lambda exon: (exon["start"], exon["end"]),
        reverse=transcript["strand"] == "-",
    )
    if len(exons) < 2:
        qc["rejected_transcript::FEWER_THAN_TWO_EXONS"] += 1
        return

    qc["multiexon_transcripts"] += 1
    for index, (upstream_exon, downstream_exon) in enumerate(
        zip(exons, exons[1:]), start=1
    ):
        if transcript["strand"] == "+":
            intron_start = upstream_exon["end"] + 1
            intron_end = downstream_exon["start"] - 1
        else:
            intron_start = downstream_exon["end"] + 1
            intron_end = upstream_exon["start"] - 1
        if intron_end < intron_start:
            qc["rejected_junction::OVERLAPPING_OR_ADJACENT_EXONS"] += 1
            continue

        for site_type, exon in (
            ("donor", upstream_exon),
            ("acceptor", downstream_exon),
        ):
            anchor, interbase = _site_coordinates(
                transcript["strand"], site_type, exon
            )
            physical_site_id = ":".join(
                [
                    transcript["chrom"],
                    str(anchor),
                    transcript["strand"],
                    site_type,
                ]
            )
            yield {
                "physical_site_id": physical_site_id,
                "chrom_rank": CHROM_RANK[transcript["chrom"]],
                "chrom": transcript["chrom"],
                "site_anchor_pos_1based": anchor,
                "junction_interbase_0based": interbase,
                "strand": transcript["strand"],
                "site_type": site_type,
                "gene_id": transcript["gene_id"],
                "gene_id_stable": transcript["gene_id_stable"],
                "gene_symbol": transcript["gene_symbol"],
                "transcript_id": transcript["transcript_id"],
                "transcript_id_stable": transcript["transcript_id_stable"],
                "refseq_transcript_ids": ";".join(
                    transcript["refseq_transcript_ids"]
                ),
                "exon_id": exon["exon_id"],
                "intron_ordinal": index,
                "intron_start_1based": intron_start,
                "intron_end_1based": intron_end,
                "annotation_source": transcript["annotation_source"],
                "annotation_class": transcript["annotation_class"],
                "source_gtf": transcript["source_gtf"],
            }


def _create_database(path):
    path = Path(path)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE memberships (
            physical_site_id TEXT NOT NULL,
            chrom_rank INTEGER NOT NULL,
            chrom TEXT NOT NULL,
            site_anchor_pos_1based INTEGER NOT NULL,
            junction_interbase_0based INTEGER NOT NULL,
            strand TEXT NOT NULL,
            site_type TEXT NOT NULL,
            gene_id TEXT,
            gene_id_stable TEXT,
            gene_symbol TEXT,
            transcript_id TEXT NOT NULL,
            transcript_id_stable TEXT,
            refseq_transcript_ids TEXT,
            exon_id TEXT,
            intron_ordinal INTEGER NOT NULL,
            intron_start_1based INTEGER NOT NULL,
            intron_end_1based INTEGER NOT NULL,
            annotation_source TEXT NOT NULL,
            annotation_class TEXT NOT NULL,
            source_gtf TEXT NOT NULL,
            UNIQUE (
                physical_site_id, transcript_id, annotation_source,
                intron_ordinal, site_type
            )
        )
        """
    )
    return connection


def _insert_memberships(
    connection,
    gtf_path,
    annotation_source,
    qc,
    mane_gene_ids,
    max_transcripts=None,
    progress_every=25000,
):
    columns = ["chrom_rank"] + MEMBERSHIP_FIELDS
    placeholders = ",".join(["?"] * len(columns))
    statement = (
        "INSERT OR IGNORE INTO memberships ("
        + ",".join(columns)
        + ") VALUES ("
        + placeholders
        + ")"
    )
    batch = []
    processed = 0
    inserted_before = connection.total_changes
    for transcript in iter_eligible_transcripts(
        gtf_path,
        annotation_source,
        qc,
        max_transcripts=max_transcripts,
    ):
        processed += 1
        if annotation_source == "MANE_v1.4" and transcript["gene_id_stable"]:
            mane_gene_ids.add(transcript["gene_id_stable"])
        qc[f"eligible_class::{transcript['annotation_class']}"] += 1
        for membership in transcript_to_memberships(transcript, qc) or ():
            batch.append(tuple(membership[column] for column in columns))
            qc[f"membership_class::{membership['annotation_class']}"] += 1
            qc[f"membership_site_type::{membership['site_type']}"] += 1
            if len(batch) >= 10000:
                connection.executemany(statement, batch)
                batch.clear()
        if progress_every and processed % progress_every == 0:
            print(
                f"  {annotation_source}: processed {processed:,} eligible transcripts"
            )
    if batch:
        connection.executemany(statement, batch)
    connection.commit()
    inserted = connection.total_changes - inserted_before
    qc["inserted_memberships"] += inserted
    print(
        f"  {annotation_source}: completed {processed:,} eligible transcripts; "
        f"inserted {inserted:,} memberships"
    )


def _export_memberships(connection, output_path):
    query_columns = MEMBERSHIP_FIELDS
    query = (
        "SELECT "
        + ",".join(query_columns)
        + " FROM memberships ORDER BY chrom_rank, site_anchor_pos_1based, "
        "strand, site_type, transcript_id, annotation_source, intron_ordinal"
    )
    rows_written = 0
    with deterministic_gzip_writer(output_path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(query_columns)
        for row in connection.execute(query):
            writer.writerow(row)
            rows_written += 1
    return rows_written


def _split_values(value):
    return {item for item in (value or "").split(";") if item}


def _new_site_group(row):
    return {
        "physical_site_id": row["physical_site_id"],
        "chrom": row["chrom"],
        "site_anchor_pos_1based": row["site_anchor_pos_1based"],
        "junction_interbase_0based": row["junction_interbase_0based"],
        "strand": row["strand"],
        "site_type": row["site_type"],
        "gene_ids": set(),
        "gene_ids_stable": set(),
        "gene_symbols": set(),
        "transcript_ids": set(),
        "transcript_ids_stable": set(),
        "refseq_transcript_ids": set(),
        "exon_ids": set(),
        "mane_select_transcript_ids": set(),
        "mane_plus_clinical_transcript_ids": set(),
        "gencode_basic_transcript_ids": set(),
        "annotation_sources": set(),
        "annotation_classes": set(),
        "source_gtf_files": set(),
        "membership_count": 0,
    }


def _add_to_site_group(group, row):
    for field in (
        "gene_ids",
        "gene_ids_stable",
        "gene_symbols",
        "transcript_ids",
        "transcript_ids_stable",
        "exon_ids",
        "annotation_sources",
        "source_gtf_files",
    ):
        source_field = {
            "gene_ids": "gene_id",
            "gene_ids_stable": "gene_id_stable",
            "gene_symbols": "gene_symbol",
            "transcript_ids": "transcript_id",
            "transcript_ids_stable": "transcript_id_stable",
            "exon_ids": "exon_id",
            "annotation_sources": "annotation_source",
            "source_gtf_files": "source_gtf",
        }[field]
        if row[source_field]:
            group[field].add(row[source_field])
    group["refseq_transcript_ids"].update(
        _split_values(row["refseq_transcript_ids"])
    )
    annotation_class = row["annotation_class"]
    group["annotation_classes"].add(annotation_class)
    transcript_id = row["transcript_id"]
    if annotation_class == "MANE_Select":
        group["mane_select_transcript_ids"].add(transcript_id)
    elif annotation_class == "MANE_Plus_Clinical":
        group["mane_plus_clinical_transcript_ids"].add(transcript_id)
    elif annotation_class == "GENCODE_Basic":
        group["gencode_basic_transcript_ids"].add(transcript_id)
    group["membership_count"] += 1


def _reference_dinucleotide(fasta, chrom, anchor, strand, site_type):
    if strand == "+" and site_type == "donor":
        sequence = fasta.fetch(chrom, anchor + 1, 2)
    elif strand == "+" and site_type == "acceptor":
        sequence = fasta.fetch(chrom, anchor - 2, 2)
    elif strand == "-" and site_type == "donor":
        sequence = reverse_complement(fasta.fetch(chrom, anchor - 2, 2))
    else:
        sequence = reverse_complement(fasta.fetch(chrom, anchor + 1, 2))
    return sequence


def _finalize_site_group(group, fasta, mane_gene_ids):
    classes = group["annotation_classes"]
    if "MANE_Select" in classes:
        catalog_tier = "MANE_SELECT"
        annotation_priority = 1
    elif "MANE_Plus_Clinical" in classes:
        catalog_tier = "MANE_PLUS_CLINICAL"
        annotation_priority = 2
    else:
        catalog_tier = "GENCODE_BASIC"
        annotation_priority = 3

    non_mane_genes = group["gene_ids_stable"] - mane_gene_ids
    if catalog_tier.startswith("MANE_"):
        default_eligible = True
        default_reason = "MANE_PRIMARY"
    elif non_mane_genes:
        default_eligible = True
        default_reason = "GENCODE_FALLBACK_FOR_GENE_WITHOUT_MANE"
    else:
        default_eligible = False
        default_reason = "GENCODE_ALTERNATIVE_REQUIRES_SOURCE_TRANSCRIPT_MATCH"

    expected = "GT" if group["site_type"] == "donor" else "AG"
    observed = _reference_dinucleotide(
        fasta,
        group["chrom"],
        group["site_anchor_pos_1based"],
        group["strand"],
        group["site_type"],
    )
    if len(observed) != 2 or any(base not in "ACGT" for base in observed):
        motif_status = "UNRESOLVED"
    elif observed == expected:
        motif_status = "CANONICAL"
    else:
        motif_status = "NONCANONICAL"

    def joined(name):
        return ";".join(sorted(group[name]))

    return {
        "physical_site_id": group["physical_site_id"],
        "chrom": group["chrom"],
        "site_anchor_pos_1based": group["site_anchor_pos_1based"],
        "junction_interbase_0based": group["junction_interbase_0based"],
        "strand": group["strand"],
        "site_type": group["site_type"],
        "expected_dinucleotide": expected,
        "reference_dinucleotide": observed,
        "motif_status": motif_status,
        "catalog_tier": catalog_tier,
        "annotation_priority": annotation_priority,
        "default_mapping_eligible": str(default_eligible).lower(),
        "default_mapping_reason": default_reason,
        "gene_ids": joined("gene_ids"),
        "gene_ids_stable": joined("gene_ids_stable"),
        "gene_symbols": joined("gene_symbols"),
        "transcript_ids": joined("transcript_ids"),
        "transcript_ids_stable": joined("transcript_ids_stable"),
        "refseq_transcript_ids": joined("refseq_transcript_ids"),
        "exon_ids": joined("exon_ids"),
        "mane_select_transcript_ids": joined("mane_select_transcript_ids"),
        "mane_plus_clinical_transcript_ids": joined(
            "mane_plus_clinical_transcript_ids"
        ),
        "gencode_basic_transcript_ids": joined(
            "gencode_basic_transcript_ids"
        ),
        "annotation_sources": joined("annotation_sources"),
        "source_gtf_files": joined("source_gtf_files"),
        "gene_count": len(group["gene_ids_stable"]),
        "transcript_count": len(group["transcript_ids"]),
        "membership_count": group["membership_count"],
    }


def _export_catalog(connection, output_path, fasta_path, mane_gene_ids):
    connection.row_factory = sqlite3.Row
    query = (
        "SELECT * FROM memberships ORDER BY chrom_rank, "
        "site_anchor_pos_1based, strand, site_type, transcript_id, "
        "annotation_source, intron_ordinal"
    )
    qc = Counter()
    rows_written = 0
    current = None
    with IndexedFasta(fasta_path) as fasta, deterministic_gzip_writer(
        output_path
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CATALOG_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in connection.execute(query):
            if current is None or row["physical_site_id"] != current["physical_site_id"]:
                if current is not None:
                    result = _finalize_site_group(current, fasta, mane_gene_ids)
                    writer.writerow(result)
                    rows_written += 1
                    qc[f"tier::{result['catalog_tier']}"] += 1
                    qc[f"site_type::{result['site_type']}"] += 1
                    qc[f"motif::{result['motif_status']}"] += 1
                    qc[
                        f"default_mapping::{result['default_mapping_eligible']}"
                    ] += 1
                current = _new_site_group(row)
            _add_to_site_group(current, row)
        if current is not None:
            result = _finalize_site_group(current, fasta, mane_gene_ids)
            writer.writerow(result)
            rows_written += 1
            qc[f"tier::{result['catalog_tier']}"] += 1
            qc[f"site_type::{result['site_type']}"] += 1
            qc[f"motif::{result['motif_status']}"] += 1
            qc[f"default_mapping::{result['default_mapping_eligible']}"] += 1
    return rows_written, qc


def _completed_summary_is_reusable(summary_path):
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if summary.get("status") != "complete":
        return None
    for output in summary.get("outputs", {}).values():
        path = Path(output["path"])
        if not path.is_file() or file_sha256(path) != output.get("sha256"):
            return None
    summary["reused"] = True
    return summary


def build_curated_splice_site_catalog(
    project_root=None,
    reuse_existing=True,
    output_root=None,
    qc_root=None,
    progress_every=25000,
):
    """Build and return the splice-site catalog summary dictionary."""

    root = resolve_project_root(project_root)
    gtf_root = root / "dataset" / "annodata" / "gtf"
    fasta_path = (
        root
        / "dataset"
        / "annodata"
        / "fa"
        / "GRCh38.primary_assembly.genome.fa"
    )
    mane_gtf = gtf_root / "MANE.GRCh38.v1.4.ensembl_genomic.gtf"
    gencode_gtf = gtf_root / "gencode.v47.basic.annotation.gtf"
    output_root = Path(output_root or root / "dataset" / "interim" / "splice_site_catalog")
    qc_root = Path(qc_root or root / "dataset" / "interim" / "splice_site_catalog_qc")
    output_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)

    membership_path = output_root / "splice_site_memberships.grch38.tsv.gz"
    catalog_path = output_root / "curated_splice_sites.grch38.tsv.gz"
    summary_path = qc_root / "splice_site_catalog.summary.json"
    database_path = output_root / "_splice_site_catalog.build.sqlite3"

    for required in (mane_gtf, gencode_gtf, fasta_path, Path(str(fasta_path) + ".fai")):
        if not required.is_file():
            raise FileNotFoundError(f"Required input is missing: {required}")

    if reuse_existing:
        reusable = _completed_summary_is_reusable(summary_path)
        if reusable is not None:
            print("Reusing the complete, checksum-validated splice-site catalog.")
            return reusable

    started = time.perf_counter()
    source_qc = {
        "MANE_v1.4": Counter(),
        "GENCODE_v47_Basic": Counter(),
    }
    mane_gene_ids = set()
    connection = _create_database(database_path)
    try:
        print("Building transcript-site memberships from MANE v1.4...")
        _insert_memberships(
            connection,
            mane_gtf,
            "MANE_v1.4",
            source_qc["MANE_v1.4"],
            mane_gene_ids,
            max_transcripts=None,
            progress_every=progress_every,
        )
        print("Building controlled GENCODE v47 Basic fallback memberships...")
        _insert_memberships(
            connection,
            gencode_gtf,
            "GENCODE_v47_Basic",
            source_qc["GENCODE_v47_Basic"],
            mane_gene_ids,
            max_transcripts=None,
            progress_every=progress_every,
        )
        connection.execute(
            "CREATE INDEX membership_order_idx ON memberships "
            "(chrom_rank, site_anchor_pos_1based, strand, site_type)"
        )
        connection.commit()

        print("Writing transcript-site membership table...")
        membership_rows = _export_memberships(connection, membership_path)
        print("Collapsing memberships into physical splice sites...")
        catalog_rows, catalog_qc = _export_catalog(
            connection, catalog_path, fasta_path, mane_gene_ids
        )
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "schema_version": 1,
        "elapsed_seconds": elapsed,
        "scope": {
            "assembly": "GRCh38",
            "chromosomes": list(PRIMARY_CHROMS),
            "mane_policy": "MANE Select and MANE Plus Clinical",
            "gencode_policy": "GENCODE v47 Basic protein-coding transcripts only",
            "physical_deduplication_key": [
                "chrom",
                "site_anchor_pos_1based",
                "strand",
                "site_type",
            ],
            "site_anchor_definition": (
                "The adjacent exonic base: last exonic base for a donor and "
                "first exonic base for an acceptor, in transcript orientation."
            ),
            "distance_convention_for_next_phase": (
                "distance=(variant_pos-site_anchor_pos)*strand_sign, where "
                "strand_sign is +1 on '+' and -1 on '-'."
            ),
            "motif_policy": (
                "Reference dinucleotides are annotated but noncanonical sites "
                "are retained."
            ),
        },
        "inputs": {
            "mane_gtf": {
                "path": str(mane_gtf),
                "size_bytes": mane_gtf.stat().st_size,
                "sha256": file_sha256(mane_gtf),
            },
            "gencode_gtf": {
                "path": str(gencode_gtf),
                "size_bytes": gencode_gtf.stat().st_size,
                "sha256": file_sha256(gencode_gtf),
            },
            "grch38_fasta": {
                "path": str(fasta_path),
                "size_bytes": fasta_path.stat().st_size,
                "fai_path": str(Path(str(fasta_path) + ".fai")),
            },
        },
        "source_qc": {
            source: dict(sorted(counter.items()))
            for source, counter in source_qc.items()
        },
        "catalog_qc": dict(sorted(catalog_qc.items())),
        "mane_gene_count": len(mane_gene_ids),
        "outputs": {
            "memberships": {
                "path": str(membership_path),
                "rows": membership_rows,
                "sha256": file_sha256(membership_path),
            },
            "physical_sites": {
                "path": str(catalog_path),
                "rows": catalog_rows,
                "sha256": file_sha256(catalog_path),
            },
        },
        "reused": False,
    }
    write_json_atomic(summary_path, summary)
    print(
        f"Completed: {membership_rows:,} memberships -> {catalog_rows:,} "
        f"physical sites in {elapsed / 60:.1f} minutes"
    )
    return summary


def load_catalog_summary(project_root=None):
    root = resolve_project_root(project_root)
    summary_path = (
        root
        / "dataset"
        / "interim"
        / "splice_site_catalog_qc"
        / "splice_site_catalog.summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Catalog summary has not been created: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validate_catalog_outputs(summary):
    """Validate checksums, physical-site uniqueness, and core field invariants."""

    result = {
        "checksum_failures": 0,
        "physical_rows": 0,
        "duplicate_physical_site_ids": 0,
        "invalid_site_types": 0,
        "invalid_strands": 0,
        "invalid_anchor_positions": 0,
        "unresolved_motifs": 0,
    }
    for output in summary["outputs"].values():
        path = Path(output["path"])
        if not path.is_file() or file_sha256(path) != output["sha256"]:
            result["checksum_failures"] += 1

    catalog_path = Path(summary["outputs"]["physical_sites"]["path"])
    previous_id = None
    with gzip.open(catalog_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            result["physical_rows"] += 1
            if row["physical_site_id"] == previous_id:
                result["duplicate_physical_site_ids"] += 1
            previous_id = row["physical_site_id"]
            result["invalid_site_types"] += row["site_type"] not in {
                "acceptor",
                "donor",
            }
            result["invalid_strands"] += row["strand"] not in {"+", "-"}
            try:
                result["invalid_anchor_positions"] += (
                    int(row["site_anchor_pos_1based"]) < 1
                )
            except ValueError:
                result["invalid_anchor_positions"] += 1
            result["unresolved_motifs"] += row["motif_status"] == "UNRESOLVED"
    result["row_count_matches_summary"] = (
        result["physical_rows"] == summary["outputs"]["physical_sites"]["rows"]
    )
    result["valid"] = (
        result["checksum_failures"] == 0
        and result["duplicate_physical_site_ids"] == 0
        and result["invalid_site_types"] == 0
        and result["invalid_strands"] == 0
        and result["invalid_anchor_positions"] == 0
        and result["row_count_matches_summary"]
    )
    return result

