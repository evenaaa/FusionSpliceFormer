"""Prepare reference assets and event caches for prediction."""
from dataclasses import fields
from pathlib import Path
from .io import atomic_json, sha256
from .runtime import Progress
import json
from .features.numeric.config import FeaturePaths, FeatureSettings
from .features.numeric.schema import FORBIDDEN_OUTPUT_COLUMNS
from .data.assets import build_assets
from .data.catalog import build_curated_splice_site_catalog
from .data.numeric import build_features
from .data.sequences import build_sequence_file
from .data.cache import InputFiles, build_prediction_cache
from .input import read_events

REGIONS = ('acceptor', 'donor')


def reference_paths(root):
    root = Path(root).resolve()
    defaults = FeaturePaths()
    return FeaturePaths(**{f.name: root / Path(getattr(defaults, f.name)).relative_to(defaults.project_root)
                          for f in fields(FeaturePaths)})


def references(root):
    root = Path(root).resolve()
    build_curated_splice_site_catalog(project_root=root)
    return build_assets(reference_paths(root), FeatureSettings())


def prepare_events(events, work, reference_root, *, preprocessor_root=None, resume=False, progress=False,
                   numeric_pipeline=None, fasta_digest=None, annotation_lookup=None, defer_coverage=False):
    """Build unlabelled inference inputs using the supplied fitted preprocessors."""
    events, work = Path(events).resolve(), Path(work).resolve()
    if preprocessor_root is None:
        raise ValueError('Provide the fitted preprocessors supplied with the model weights')
    target = work / 'prediction'
    if target.exists() and not resume:
        raise FileExistsError(f'Choose a new work directory or inspect existing outputs: {target}')
    frame = read_events(events)
    if not frame.event_key.is_unique or not frame.site_type.isin(REGIONS).all():
        raise ValueError('Invalid event IDs or site types')
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(frame.columns):
        raise ValueError('Remove label/evidence fields from inference input: '+
                         ', '.join(sorted(FORBIDDEN_OUTPUT_COLUMNS.intersection(frame.columns))))
    paths = reference_paths(reference_root)
    # Builders use a common compressed event format, even for plain user TSVs.
    target.mkdir(parents=True, exist_ok=resume)
    events = target/'events.tsv.gz'
    if not events.exists():
        frame.to_csv(events, sep='\t', index=False, compression={'method':'gzip','mtime':0})
    else:
        # A preparation interrupted during input writing is safely restarted.
        try:
            old=read_events(events)
            same=old.astype(str).equals(frame.astype(str))
        except Exception:
            same=False
        if not same:
            frame.to_csv(events, sep='\t', index=False, compression={'method':'gzip','mtime':0})

    def stage(name, folder, build):
        marker=target/f'{name}.done.json'
        if resume and marker.is_file():
            saved=json.loads(marker.read_text(encoding='utf-8'))
            if not all((folder/n).is_file() and sha256(folder/n)==digest for n,digest in saved.items()):
                raise ValueError(f'Saved preparation stage failed integrity checks: {name}')
            if progress: print(f'Preparing {name}: complete (reused)',flush=True)
            return
        build()
        atomic_json(marker,{str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})

    stage('numeric',target/'numeric',lambda: build_features(events,target/'numeric/data.tsv.gz',paths,
                                                          FeatureSettings(),progress=progress,pipeline=numeric_pipeline,
                                                          defer_coverage=defer_coverage))
    stage('sequence',target/'sequence',lambda: build_sequence_file(events,target/'sequence/data.tsv.gz',
                                                                 paths.fasta,progress=progress,total=len(frame),fasta_digest=fasta_digest))
    for region in REGIONS:
        if not frame.site_type.eq(region).any():
            continue
        cache = work/'cache'/'prediction'/region
        pp = Path(preprocessor_root)/region/'preprocessor.json'
        def build_cache():
            if cache.exists():
                # Cache construction publishes its directory atomically. A crash
                # just before the stage receipt can leave a valid complete cache.
                from .runtime import cache_identity
                cache_identity(cache.parent)
                return
            reporter=Progress(f'Preparing {region} cache',int(frame.site_type.eq(region).sum())) if progress else None
            build_prediction_cache(region=region,
                files=InputFiles(events,target/'numeric/data.tsv.gz',target/'sequence/data.tsv.gz'),
                cache_dir=cache, preprocessor_path=pp, site_catalog=paths.site_catalog,
                progress_callback=reporter.update if reporter else None,annotation_lookup=annotation_lookup)
        stage(f'{region}_cache',cache,build_cache)
    atomic_json(target/'complete.json', {'status':'PASS','events':len(frame)})
    return work/'cache'/'prediction'
