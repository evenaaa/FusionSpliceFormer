from fusionspliceformer.features.numeric.config import FeaturePaths, FeatureSettings
from fusionspliceformer.features.numeric.context_features import build_structural_context
from fusionspliceformer.features.numeric.pwm import build_pwms, save_pwm, sha256_file
from fusionspliceformer.features.numeric.qc import write_json
from fusionspliceformer.features.numeric.schema import write_schema
def build_assets(paths: FeaturePaths, settings: FeatureSettings, force: bool = False) -> dict:
    paths = paths.resolved()
    settings.validate()
    paths.asset_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        paths.asset_dir / "donor_pwm.grch38.json",
        paths.asset_dir / "acceptor_pwm.grch38.json",
        paths.asset_dir / "structural_context.grch38.tsv.gz",
        paths.asset_dir / "feature_asset_manifest.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Feature assets already exist; use --force only when intentionally rebuilding: "
            + ", ".join(map(str, existing))
        )
    pwms, pwm_metadata = build_pwms(paths.site_catalog, paths.fasta, settings)
    for site_type, pwm in pwms.items():
        save_pwm(paths.asset_dir / f"{site_type}_pwm.grch38.json", pwm)
    structural = build_structural_context(
        paths.memberships, paths.site_catalog, [paths.mane_gtf, paths.gencode_gtf]
    )
    structural_path = paths.asset_dir / "structural_context.grch38.tsv.gz"
    structural.to_csv(
        structural_path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    schema_path = paths.asset_dir / "numeric_feature_schema.json"
    write_schema(schema_path, settings.schema_version)
    manifest = {
        "status": "PASS",
        "schema_version": settings.schema_version,
        "reference_build": settings.reference_build,
        "label_independent": True,
        "pwm": pwm_metadata,
        "structural_context": {
            "path": str(structural_path),
            "sha256": sha256_file(structural_path),
            "row_count": int(len(structural)),
            "context_available_count": int(structural["transcript_context_available"].sum()),
            "memberships_sha256": sha256_file(paths.memberships),
            "mane_gtf_sha256": sha256_file(paths.mane_gtf),
            "gencode_gtf_sha256": sha256_file(paths.gencode_gtf),
        },
        "outputs": {str(path.name): sha256_file(path) for path in targets[:3]},
    }
    write_json(paths.asset_dir / "feature_asset_manifest.json", manifest)
    return manifest

