"""Progress, restart-safe workspaces and transactional prediction output."""
from contextlib import contextmanager
from pathlib import Path
import errno
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np

from .io import atomic_json, sha256


class Progress:
    def __init__(self, label, total, initial=0, interval=5):
        self.label, self.total, self.initial = label, int(total), int(initial)
        self.started = time.perf_counter()
        self.last = self.started
        self.interval = interval
        self.update(initial, force=True)

    def update(self, completed, force=False):
        now = time.perf_counter()
        if not force and completed != self.total and now-self.last < self.interval:
            return
        elapsed = now-self.started
        advanced = completed-self.initial
        eta = f'{elapsed*(self.total-completed)/advanced:.1f}s' if advanced > 0 else '--'
        if completed == self.total:
            eta = '0.0s'
        percent = 100*completed/self.total if self.total else 100
        print(f'{self.label}: {completed:,}/{self.total:,} ({percent:.1f}%) | '
              f'elapsed {elapsed:.1f}s | ETA {eta}', flush=True)
        self.last = now


@contextmanager
def work_lock(parent):
    """Protect reference preparation in its resource directory."""
    parent = Path(os.path.abspath(parent)).resolve()
    with _file_lock(parent/'.prediction.lock', 'Another process is preparing this resource directory.'):
        yield


@contextmanager
def output_lock(output):
    """Lock a destination independently of the chosen temporary workspace."""
    output = Path(os.path.abspath(output)).resolve()
    lock = output.with_name('.'+output.name+'.prediction.lock')
    with _file_lock(lock, f'Another prediction is writing this output: {output}'):
        yield


@contextmanager
def _file_lock(path, message):
    # Keep the file after unlocking: deleting it can split concurrent callers
    # across different file handles/inodes and defeat mutual exclusion.
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(message) from error
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(message) from error
        locked = True
        yield
    finally:
        if locked:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def file_identity(path, *, content=True):
    path = Path(path).resolve()
    stat = path.stat()
    identity = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if content:
        identity['sha256'] = sha256(path)
    return identity


def cache_identity(root):
    root = Path(root).resolve()
    files = []
    for region in ('acceptor', 'donor'):
        folder = root/region
        if not folder.exists():
            continue
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS':
            raise ValueError(f'Incomplete input cache: {folder}')
        files.append(file_identity(folder/'manifest.json'))
        for name, expected in manifest['output_sha256'].items():
            path = folder/name
            if path.resolve().parent != folder.resolve() or sha256(path) != expected:
                raise ValueError(f'Input cache checksum mismatch: {path}')
            files.append(file_identity(path, content=False))
    if not files:
        raise ValueError(f'No completed input cache: {root}')
    return files


def run_signature(args, device):
    from .prepare import reference_paths
    source = args.cache_root.resolve()
    signature = {'version': 1, 'input': str(source), 'output': str(Path(os.path.abspath(args.output)).resolve()),
                 'batch_size': args.batch_size, 'device': device, 'amp': args.amp,
                 'details': getattr(args, 'details', False),
                 'chunk_size': getattr(args, 'chunk_size', 1024),
                 'models': [file_identity(p) for p in args.checkpoints]}
    import torch
    signature['torch'] = torch.__version__
    signature['numpy'] = np.__version__
    if device == 'cuda':
        signature['gpu'] = torch.cuda.get_device_name(0)
        signature['cuda'] = torch.version.cuda
    package = Path(__file__).parent
    signature['code'] = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob('*.py'))}
    if source.is_file():
        signature['input_file'] = file_identity(source)
        paths = reference_paths(args.reference_root)
        # Do not rehash multi-GB genome/BigWig files on every restart. Pin their
        # resolved path, size and nanosecond mtime; hash small derived assets.
        resources = [paths.fasta, Path(str(paths.fasta)+'.fai'), paths.phylop_bigwig,
                     paths.phastcons_bigwig, paths.site_catalog]
        resources += sorted(paths.asset_dir.glob('*'))
        signature['references'] = [file_identity(p, content=p.stat().st_size < 10_000_000)
                                   for p in resources if p.is_file()]
        signature['preprocessors'] = [file_identity(args.preprocessor_root/r/'preprocessor.json')
                                      for r in ('acceptor', 'donor')
                                      if (args.preprocessor_root/r/'preprocessor.json').is_file()]
    else:
        signature['cache'] = cache_identity(source)
    return signature


def open_run(parent, signature, *, verbose=True):
    parent = Path(os.path.abspath(parent)).resolve()
    key = json.dumps([signature['input'], signature['output']], ensure_ascii=False).encode()
    root = parent/('prediction_'+hashlib.sha256(key).hexdigest()[:16])
    if Path(signature['output']).is_relative_to(root):
        raise ValueError('Output must be outside its temporary run directory')
    marker = root/'run.json'
    if root.exists():
        if root.is_symlink() or not marker.is_file():
            raise ValueError(f'Unrecognized work directory; left untouched: {root}')
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous != {'owner': 'FusionSpliceFormer prediction', 'signature': signature}:
            raise ValueError('Previous run has different inputs, weights, settings or code. '
                             'Use a new -O path or a different -W; old progress is preserved.')
        print(f'Resuming saved run{": " + str(root) if verbose else "..."}', flush=True)
    else:
        root.mkdir(parents=True)
        atomic_json(marker, {'owner': 'FusionSpliceFormer prediction', 'signature': signature})
        if verbose: print(f'Work directory: {root}', flush=True)
    return root


def cleanup_run(root, parent, signature):
    """Delete only the exact run created/verified by this invocation."""
    root, parent = Path(root), Path(os.path.abspath(parent)).resolve()
    if root.is_symlink() or root.resolve().parent != parent or not root.name.startswith('prediction_'):
        raise ValueError('Refusing to clean an unowned work directory')
    marker = json.loads((root/'run.json').read_text(encoding='utf-8'))
    if marker != {'owner': 'FusionSpliceFormer prediction', 'signature': signature}:
        raise ValueError('Work directory ownership changed; cleanup refused')
    shutil.rmtree(root.resolve())


class ScoreStore:
    """Atomic score blocks. Incomplete .tmp files are never treated as committed."""
    def __init__(self, folder, total):
        self.folder, self.total = Path(folder), total
        self.folder.mkdir(parents=True, exist_ok=True)
        self.values = []
        for file in sorted(self.folder.glob('*.npz')):
            with np.load(file, allow_pickle=False) as saved:
                start, values = int(saved['start']), saved['scores']
            if start != len(self.values) or values.ndim != 1 or not len(values):
                raise ValueError(f'Invalid saved score block: {file}')
            if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError(f'Invalid saved probabilities: {file}')
            self.values.extend(values.tolist())
        if len(self.values) > total:
            raise ValueError('Saved prediction count exceeds input count')

    def save(self, values):
        if not values:
            return
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError('Invalid probabilities; progress was not committed')
        start = len(self.values)
        if start+len(scores) > self.total:
            raise ValueError('Too many predicted scores')
        destination = self.folder/f'{start:012d}.npz'
        fd, temporary = tempfile.mkstemp(prefix='.scores_', suffix='.tmp', dir=self.folder)
        try:
            with os.fdopen(fd, 'wb') as handle:
                np.savez(handle, start=np.asarray(start), scores=scores)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.values.extend(scores.tolist())


def publish_tsv(report, output, work, formatter):
    """Publish a closed, complete TSV atomically, without overwriting a file."""
    _publish(lambda handle: report.to_csv(handle, sep='\t', index=False,
                                         float_format=formatter, na_rep='.'), output, work)


def publish_rows(rows, columns, output, work, formatter):
    """Stream result rows without accumulating the final report in memory."""
    import csv
    import math
    count = 0
    def write(handle):
        nonlocal count
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            row = {key: row[key] for key in columns}
            if 'SCORE' in row and isinstance(row['SCORE'], (float, np.floating)):
                row['SCORE'] = formatter(row['SCORE']) if math.isfinite(row['SCORE']) else '.'
            writer.writerow(row)
            count += 1
    _publish(write, output, work)
    return count


def _publish(write, output, work):
    output, work = Path(output).resolve(), Path(work).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.final_', suffix='.tmp', dir=work)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
            write(handle)
            handle.flush(); os.fsync(handle.fileno())
        try:
            if os.name == 'nt':
                os.rename(temporary, output)  # Atomic and no-overwrite on Windows.
            else:
                os.link(temporary, output)
        except OSError as error:
            if error.errno != errno.EXDEV and getattr(error, 'winerror', None) != 17:
                raise
            # Atomic publication across volumes requires a short-lived complete
            # staging file on the output volume. Never write to the final path.
            fd2, name2 = tempfile.mkstemp(prefix='.fusion_', suffix='.tmp', dir=output.parent)
            staged = Path(name2)
            try:
                with os.fdopen(fd2, 'wb') as destination, temporary.open('rb') as source:
                    shutil.copyfileobj(source, destination)
                    destination.flush(); os.fsync(destination.fileno())
                if os.name == 'nt':
                    os.rename(staged, output)  # Windows rename refuses overwrite.
                else:
                    os.link(staged, output)
            finally:
                staged.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
