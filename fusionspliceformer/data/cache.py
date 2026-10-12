"""Prepare and read unlabelled model inputs using fitted numerical preprocessors."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
import numpy as np
import pandas as pd
from fusionspliceformer.features.numeric.schema import ACCEPTOR_FEATURE_NAMES, DONOR_FEATURE_NAMES, FORBIDDEN_OUTPUT_COLUMNS
from fusionspliceformer.released_model import MODEL_VERSION
CACHE_SCHEMA_VERSION = 'prediction_cache_v1.1.0'
PREPROCESSOR_SCHEMA_VERSION = 'numeric_preprocessor_v1.0.0'
REGIONS = ('acceptor', 'donor')
BINARY_FEATURE_NAMES = frozenset({'is_exonic', 'phylop_available', 'phastcons_available', 'transcript_context_available', 'creates_ag_in_agez', 'destroys_ag_in_agez', 'creates_yag_in_agez', 'destroys_yag_in_agez', 'changes_acceptor_minus3_y_to_r'})

@dataclass(frozen=True)
class InputFiles:
    queue: Path
    numeric: Path
    sequences: Path

def _sha256_file(path: Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _normalise_region(region: str) -> str:
    region = str(region).strip().lower()
    if region not in REGIONS:
        raise ValueError(f'region must be one of {REGIONS}, got {region!r}')
    return region

def feature_names_for_region(region: str) -> tuple[str, ...]:
    region = _normalise_region(region)
    return tuple(ACCEPTOR_FEATURE_NAMES) if region == 'acceptor' else tuple(DONOR_FEATURE_NAMES)

class FittedNumericPreprocessor:
    """Apply the supplied region-specific imputation and scaling parameters.

    Missing continuous values are replaced by the fitted median.  Continuous
    features are then z-scored using the fitted mean and standard deviation.  Binary
    indicators remain 0/1 so explicit availability flags retain their intended
    meaning.
    """

    def __init__(self, region: str, feature_names: Iterable[str] | None=None, *, model_version: str=MODEL_VERSION):
        self.region = _normalise_region(region)
        self.model_version = str(model_version)
        self.feature_names = tuple(feature_names or feature_names_for_region(self.region))
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError('Numeric feature names must be unique')
        self.binary_features = tuple((name for name in self.feature_names if name in BINARY_FEATURE_NAMES))
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.medians is not None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise KeyError(f'Numeric input is missing features: {missing}')
        matrix = frame.loc[:, self.feature_names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError('Numeric preprocessor has not been fitted')
        matrix = self._matrix(frame)
        imputed = np.where(np.isnan(matrix), self.medians, matrix)
        transformed = (imputed - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise FloatingPointError('Numeric transform produced NaN or Inf')
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError('Cannot serialise an unfitted preprocessor')
        return {'schema_version': PREPROCESSOR_SCHEMA_VERSION, 'model_version': self.model_version, 'region': self.region, 'feature_names': list(self.feature_names), 'binary_features_unscaled': list(self.binary_features), 'medians': self.medians.tolist(), 'means': self.means.tolist(), 'scales': self.scales.tolist()}

    def save(self, path: Path) -> None:
        _write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path) -> 'FittedNumericPreprocessor':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema_version') != PREPROCESSOR_SCHEMA_VERSION:
            raise ValueError('Unsupported numeric preprocessor schema')
        if payload.get('model_version') != MODEL_VERSION:
            raise ValueError('Preprocessor does not match the supported model version')
        result = cls(payload['region'], payload['feature_names'], model_version=payload.get('model_version', MODEL_VERSION))
        result.medians = np.asarray(payload['medians'], dtype=np.float64)
        result.means = np.asarray(payload['means'], dtype=np.float64)
        result.scales = np.asarray(payload['scales'], dtype=np.float64)
        expected = (len(result.feature_names),)
        for (name, value) in (('medians', result.medians), ('means', result.means), ('scales', result.scales)):
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f'Invalid {name} in preprocessor artifact')
        if (result.scales <= 0).any():
            raise ValueError('Preprocessor scales must be positive')
        return result

def _read_region_table(path: Path, region: str, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t', low_memory=False,
                        converters={'ID': lambda value: value or '.'})
    if 'site_type' not in frame:
        raise KeyError(f'Missing site_type in {path}')
    frame = frame.loc[frame['site_type'].astype(str).str.lower().eq(region)].reset_index(drop=True)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError('limit must be positive or None')
        frame = frame.head(int(limit)).copy()
    return frame
_SEQUENCE_LUT = np.full(256, 4, dtype=np.uint8)
for (_base_index, _base) in enumerate('ACGT'):
    _SEQUENCE_LUT[ord(_base)] = _base_index
    _SEQUENCE_LUT[ord(_base.lower())] = _base_index

def encode_dna(sequence: str, expected_length: int) -> np.ndarray:
    sequence = str(sequence)
    if len(sequence) != expected_length:
        raise ValueError(f'Expected sequence length {expected_length}, got {len(sequence)}')
    raw = np.frombuffer(sequence.encode('ascii', errors='strict'), dtype=np.uint8)
    encoded = _SEQUENCE_LUT[raw]
    invalid = ~np.isin(raw, np.frombuffer(b'ACGTNacgtn', dtype=np.uint8))
    if invalid.any():
        bad = chr(int(raw[np.flatnonzero(invalid)[0]]))
        raise ValueError(f'Unexpected DNA character {bad!r}')
    return encoded

def _prepare_cache_target(cache_dir: Path, overwrite: bool) -> Path:
    cache_dir = Path(cache_dir).resolve()
    if cache_dir.exists():
        if not overwrite:
            raise FileExistsError(f'Cache already exists: {cache_dir}; pass overwrite=True explicitly')
        if cache_dir.is_file():
            raise NotADirectoryError(cache_dir)
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    return cache_dir

def build_prediction_cache(*, region: str, files: InputFiles, cache_dir: Path, preprocessor: FittedNumericPreprocessor | None=None, preprocessor_path: Path | None=None, limit: int | None=None, overwrite: bool=False, model_version: str=MODEL_VERSION, progress_callback=None, site_catalog: Path | None=None, annotation_lookup=None) -> dict:
    """Build the numerical and paired 10-kb sequence inputs for one site type."""
    region = _normalise_region(region)
    files = InputFiles(*(Path(value).resolve() for value in asdict(files).values()))
    for path in (files.queue, files.numeric, files.sequences):
        if not path.is_file():
            raise FileNotFoundError(path)
    if preprocessor is not None and preprocessor_path is not None:
        raise ValueError('Provide preprocessor or preprocessor_path, not both')
    numeric = _read_region_table(files.numeric, region, limit)
    queue = _read_region_table(files.queue, region, limit)
    if numeric.empty:
        raise ValueError(f'No {region} prediction events')
    required_identity = {'event_key', 'variant_key', 'physical_site_id', 'site_type'}
    for (name, frame) in (('numeric', numeric), ('queue', queue)):
        missing = required_identity - set(frame)
        if missing:
            raise KeyError(f'{name} input is missing identity columns: {sorted(missing)}')
    numeric_keys = numeric['event_key'].astype(str).tolist()
    queue_keys = queue['event_key'].astype(str).tolist()
    if numeric_keys != queue_keys:
        raise ValueError('Queue and numeric event order differ')
    forbidden_numeric = FORBIDDEN_OUTPUT_COLUMNS & set(numeric.columns)
    if forbidden_numeric:
        raise PermissionError(f'Numeric features contain forbidden label/evidence fields: {sorted(forbidden_numeric)}')
    leaked = FORBIDDEN_OUTPUT_COLUMNS & set(queue.columns)
    if leaked:
        raise PermissionError(f'Prediction inputs contain forbidden fields: {sorted(leaked)}')
    if preprocessor_path is not None:
        preprocessor = FittedNumericPreprocessor.load(preprocessor_path)
    if preprocessor is None or not preprocessor.fitted:
        raise ValueError('A fitted numerical preprocessor is required')
    if preprocessor.model_version != model_version:
        raise ValueError('Preprocessor belongs to a different model version')
    if preprocessor.region != region:
        raise ValueError(f'Preprocessor region {preprocessor.region} does not match {region}')
    transformed_numeric = preprocessor.transform(numeric)
    cache_dir = _prepare_cache_target(cache_dir, overwrite)
    staging = Path(tempfile.mkdtemp(prefix=f'.prediction_{region}_', dir=cache_dir.parent))
    try:
        row_count = len(numeric)
        np.save(staging / 'numeric.npy', transformed_numeric, allow_pickle=False)
        sequence_arrays = {'site_refseq': np.lib.format.open_memmap(staging / 'site_refseq.npy', mode='w+', dtype=np.uint8, shape=(row_count, 10000)), 'site_altseq': np.lib.format.open_memmap(staging / 'site_altseq.npy', mode='w+', dtype=np.uint8, shape=(row_count, 10000))}
        written = 0
        with gzip.open(files.sequences, 'rt', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            required_sequence = {'event_key', 'site_type', 'site_ref_sequence', 'site_alt_sequence'}
            missing = required_sequence - set(reader.fieldnames or ())
            if missing:
                raise KeyError(f'Sequence input is missing: {sorted(missing)}')
            for row in reader:
                if str(row['site_type']).strip().lower() != region:
                    continue
                if written >= row_count:
                    break
                if str(row['event_key']) != numeric_keys[written]:
                    raise ValueError(f'Sequence and numeric event order differ at region row {written}')
                sequence_arrays['site_refseq'][written] = encode_dna(row['site_ref_sequence'], 10000)
                sequence_arrays['site_altseq'][written] = encode_dna(row['site_alt_sequence'], 10000)
                written += 1
                if progress_callback: progress_callback(written)
        if written != row_count:
            raise ValueError(f'Sequence rows for {region} ended at {written}, expected {row_count}')
        for array in sequence_arrays.values():
            array.flush()
            mmap_handle = getattr(array, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
        del array
        del sequence_arrays
        metadata_columns = [name for name in ('event_group_id', 'event_key', 'variant_key', 'physical_site_id', 'site_type', 'strand') if name in numeric]
        metadata = numeric.loc[:, metadata_columns].copy()
        metadata['input_id'] = queue['ID'].fillna('.').astype(str).to_numpy() if 'ID' in queue else '.'
        if site_catalog is not None:
            from fusionspliceformer.output import site_annotations
            annotations = (annotation_lookup(metadata.physical_site_id) if annotation_lookup is not None
                           else site_annotations(site_catalog, metadata.physical_site_id))
            metadata['gene'] = [annotations[site][0] for site in metadata.physical_site_id]
            metadata['transcripts'] = [annotations[site][1] for site in metadata.physical_site_id]
        metadata.to_csv(staging / 'metadata.tsv.gz', sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        output_names = ['numeric.npy', 'site_refseq.npy', 'site_altseq.npy', 'metadata.tsv.gz']
        manifest = {'status': 'PASS', 'schema_version': CACHE_SCHEMA_VERSION, 'model_version': model_version, 'purpose': 'inference', 'region': region, 'row_limit': limit, 'row_count': row_count, 'labels_present': False, 'label_free': True, 'feature_names': list(preprocessor.feature_names), 'numeric_feature_count': len(preprocessor.feature_names), 'sequence_shapes': {'site_refseq': [row_count, 10000], 'site_altseq': [row_count, 10000]}, 'event_key_sha256': _sha256_values(numeric_keys), 'source_files': {'queue': str(files.queue), 'queue_sha256': _sha256_file(files.queue), 'numeric': str(files.numeric), 'numeric_sha256': _sha256_file(files.numeric), 'sequences': str(files.sequences), 'sequences_sha256': _sha256_file(files.sequences)}, 'output_sha256': {name: _sha256_file(staging / name) for name in output_names}, 'preprocessor': preprocessor.to_dict()}
        _write_json(staging / 'manifest.json', manifest)
        os.replace(staging, cache_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

class PredictionDataset:
    """PyTorch-compatible lazy reader over one model-input cache."""

    def __init__(self, cache_dir: Path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('PyTorch is required to instantiate PredictionDataset') from exc
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = json.loads((self.cache_dir / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('status') != 'PASS':
            raise ValueError('Cache manifest is not PASS')
        if self.manifest.get('label_free') is not True:
            raise PermissionError('Cache must contain label-free prediction inputs')
        self.numeric = np.load(self.cache_dir / 'numeric.npy', mmap_mode='r')
        self.site_refseq = np.load(self.cache_dir / 'site_refseq.npy', mmap_mode='r')
        self.site_altseq = np.load(self.cache_dir / 'site_altseq.npy', mmap_mode='r')
        expected = int(self.manifest['row_count'])
        arrays = (self.numeric, self.site_refseq, self.site_altseq)
        if any((len(value) != expected for value in arrays)):
            raise ValueError('Cache arrays do not match manifest row_count')

    def __len__(self) -> int:
        return int(self.manifest['row_count'])

    def close(self):
        """Release mapped files before deleting a completed chunk on Windows."""
        for name in ('numeric', 'site_refseq', 'site_altseq'):
            handle = getattr(getattr(self, name, None), '_mmap', None)
            if handle is not None:
                handle.close()

    def __getitem__(self, index: int) -> dict:
        import torch
        item = {'numeric': torch.from_numpy(np.array(self.numeric[index], copy=True)).float(), 'site_refseq': torch.from_numpy(np.array(self.site_refseq[index], copy=True)), 'site_altseq': torch.from_numpy(np.array(self.site_altseq[index], copy=True))}
        return item
