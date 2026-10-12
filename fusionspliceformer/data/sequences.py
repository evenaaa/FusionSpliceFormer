from pathlib import Path
import csv
import gzip
from fusionspliceformer.data import windows as window_policy
from fusionspliceformer.features.numeric.config import FeaturePaths
from fusionspliceformer.features.numeric.sequence import IndexedFasta, reverse_complement
from fusionspliceformer.data.catalog import deterministic_gzip_writer, file_sha256, write_json_atomic
SCHEMA_VERSION = "sequence_inputs_v1.0.0"


OUTPUT_FIELDS = [
    "event_group_id", "event_key", "variant_key", "physical_site_id",
    "site_type", "strand",
    "site_window_start0", "site_window_end0",
    "variant_in_site_window", "site_ref_sequence", "site_alt_sequence",
]


def _mutate_genomic(sequence, window, pos_1based, ref, alt):
    index = int(pos_1based) - 1 - window.start0
    if not 0 <= index < len(sequence):
        return sequence, False
    observed = sequence[index].upper()
    ref = str(ref).upper()
    alt = str(alt).upper()
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
        raise ValueError("Sequence input builder supports A/C/G/T SNVs only")
    if observed not in {ref, "N"}:
        raise ValueError(
            f"FASTA REF mismatch at {window.chrom}:{pos_1based}: {observed} != {ref}"
        )
    return sequence[:index] + alt + sequence[index + 1 :], True


def build_event_sequences(row, fasta):
    window = window_policy.event_site_window(row)
    for suffix, observed in (("start0", window.start0), ("end0", window.end0)):
        field = f"site_window_{suffix}"
        if str(row.get(field, "")).strip() and int(row[field]) != observed:
            raise ValueError(f"Input {field} does not match the sequence window")
    site_ref_genomic = fasta.fetch_padded(
        window.chrom, window.start_1based, window.length
    ).upper()
    site_alt_genomic, in_site = _mutate_genomic(
        site_ref_genomic, window, row["grch38_pos"], row["grch38_ref"], row["grch38_alt"]
    )
    if not in_site:
        raise ValueError('Variant lies outside the target-site 10-kb window')
    if row["strand"] == "-":
        site_ref_genomic = reverse_complement(site_ref_genomic)
        site_alt_genomic = reverse_complement(site_alt_genomic)
    if len(site_ref_genomic) != 10_000 or len(site_alt_genomic) != 10_000:
        raise AssertionError("Site sequence is not exactly 10,000 bp")
    return {
        "event_group_id": row["event_group_id"],
        "event_key": row["event_key"],
        "variant_key": row["variant_key"],
        "physical_site_id": row["physical_site_id"],
        "site_type": row["site_type"],
        "strand": row["strand"],
        "site_window_start0": window.start0,
        "site_window_end0": window.end0,
        "variant_in_site_window": str(in_site).lower(),
        "site_ref_sequence": site_ref_genomic,
        "site_alt_sequence": site_alt_genomic,
    }


def build_sequence_file(input_path, output_path, fasta_path=None, limit=None, *, progress=False, total=None,
                        fasta_digest=None):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    fasta_path = Path(fasta_path or FeaturePaths().fasta).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    from fusionspliceformer.runtime import Progress
    reporter=Progress('Preparing REF/ALT sequences',total) if progress and total is not None else None
    with IndexedFasta(fasta_path) as fasta, gzip.open(
        input_path, "rt", encoding="utf-8", newline=""
    ) as source, deterministic_gzip_writer(output_path) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        from fusionspliceformer.features.numeric.schema import FORBIDDEN_OUTPUT_COLUMNS
        if FORBIDDEN_OUTPUT_COLUMNS.intersection(reader.fieldnames or []):
            raise ValueError("Remove label/evidence fields from prediction input")
        writer = csv.DictWriter(
            destination, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in reader:
            if limit is not None and rows >= limit:
                break
            writer.writerow(build_event_sequences(row, fasta))
            rows += 1
            if reporter: reporter.update(rows)
    manifest = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "row_limit": limit,
        "rows": rows,
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "fasta_path": str(fasta_path),
        "fasta_sha256": fasta_digest if fasta_digest is not None else file_sha256(fasta_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "window_policy": window_policy.policy_manifest(),
        "label_columns_copied": False,
    }
    write_json_atomic(
        output_path.with_name(output_path.name.replace(".tsv.gz", ".manifest.json")),
        manifest,
    )
    return manifest
