from pathlib import Path
import pandas as pd
from fusionspliceformer.features.numeric.config import FeaturePaths, FeatureSettings
from fusionspliceformer.features.numeric.pipeline import NumericFeaturePipeline
from fusionspliceformer.features.numeric.pwm import sha256_file
from fusionspliceformer.features.numeric.qc import validate_feature_frame, write_json
from fusionspliceformer.features.numeric.schema import write_schema
def build_features(
    input_path: Path,
    output_path: Path,
    paths: FeaturePaths,
    settings: FeatureSettings,
    *,
    limit: int | None = None,
    progress: bool = False,
    pipeline=None,
    defer_coverage=False,
) -> dict:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    events = pd.read_csv(input_path, sep="\t", nrows=limit, low_memory=False)
    from fusionspliceformer.features.numeric.schema import FORBIDDEN_OUTPUT_COLUMNS
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(events.columns):
        raise ValueError("Remove label/evidence fields from prediction input")
    pipeline = pipeline if pipeline is not None else NumericFeaturePipeline(paths, settings)
    from fusionspliceformer.runtime import Progress
    reporter=Progress('Preparing numeric features',len(events)) if progress else None
    features = pipeline.build(events, progress_callback=reporter.update if reporter else None)
    qc = validate_feature_frame(
        features,
        expected_rows=len(events),
        minimum_conservation_coverage=settings.minimum_conservation_coverage,
        defer_coverage=defer_coverage,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        output_path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    schema_path = output_path.parent / "numeric_feature_schema.json"
    write_schema(schema_path, settings.schema_version)
    qc_path = output_path.with_name(output_path.name.replace(".tsv.gz", ".qc.json"))
    write_json(qc_path, qc)
    manifest = {
        "status": "PASS",
        "schema_version": settings.schema_version,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_limit": limit,
        "label_columns_copied": False,
        "delta_convention": "ALT_MINUS_REF",
        "qc_path": str(qc_path),
        "qc": qc,
    }
    manifest_path = output_path.with_name(output_path.name.replace(".tsv.gz", ".manifest.json"))
    write_json(manifest_path, manifest)
    return manifest
