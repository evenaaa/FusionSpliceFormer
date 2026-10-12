"""Offline setup from one download folder; never overwrites existing resources."""
from pathlib import Path
import argparse
import gzip
import hashlib
import os
import tempfile
from .io import atomic_json, sha256

FILES = {
    'GRCh38.primary_assembly.genome.fa': 'fa',
    'MANE.GRCh38.v1.4.ensembl_genomic.gtf': 'gtf',
    'gencode.v47.basic.annotation.gtf': 'gtf',
    'hg38.phyloP100way.bw': 'conservation',
    'hg38.phastCons100way.bw': 'conservation',
}


def install_file(source, destination):
    """Compare decoded content on rerun; copy/decompress to a short temporary path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.suffix == '.gz' else open
    if destination.exists():
        digest = hashlib.sha256()
        with opener(source, 'rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != sha256(destination):
            raise ValueError(f'Existing reference differs; left untouched: {destination}')
        return
    fd, name = tempfile.mkstemp(prefix='.resource_', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as out, opener(source, 'rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        # Windows rename refuses overwrite; POSIX link provides the same guard.
        if os.name == 'nt':
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def fasta_index(fasta):
    """Create standard faidx records for fixed-width, uncompressed FASTA."""
    rows, seen = [], set()
    name = None
    length = offset = bases = width = 0
    previous = None
    with fasta.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line or line.startswith(b'>'):
                if name is not None:
                    if not length:
                        raise ValueError(f'Empty FASTA sequence: {name}')
                    rows.append(f'{name}\t{length}\t{offset}\t{bases}\t{width}\n')
                if not line:
                    break
                name = line[1:].split()[0].decode('ascii')
                if name in seen:
                    raise ValueError(f'Duplicate FASTA sequence: {name}')
                seen.add(name)
                length, offset, bases, width, previous = 0, stream.tell(), 0, 0, None
                continue
            if name is None:
                raise ValueError('FASTA must start with a header')
            sequence = line.rstrip(b'\r\n')
            if not sequence or any(c in sequence for c in (b' ', b'\t')):
                raise ValueError('Blank/whitespace FASTA sequence line')
            if previous is not None and previous != (bases, width):
                raise ValueError('FASTA contains a short or irregular non-final sequence line')
            if not bases:
                bases, width = len(sequence), len(line)
            if len(sequence) > bases:
                raise ValueError('FASTA line exceeds the first sequence line width')
            previous = (len(sequence), len(line))
            length += len(sequence)
    if not rows:
        raise ValueError('Empty FASTA')
    return ''.join(rows)


def setup(downloads, root):
    from .prepare import references
    from .runtime import work_lock
    downloads, root = Path(downloads).resolve(), Path(root).resolve()
    sources = {}
    for name in FILES:
        matches = [p for p in (downloads/name, downloads/(name+'.gz')) if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f'Provide exactly one of {name} or {name}.gz in {downloads}')
        sources[name] = matches[0]
    with work_lock(root):
        for number, (name, subdir) in enumerate(FILES.items(), 1):
            print(f'Reference {number}/{len(FILES)}: {name} (copy/decompress or verify)', flush=True)
            install_file(sources[name], root/'dataset/annodata'/subdir/name)
        fasta = root/'dataset/annodata/fa/GRCh38.primary_assembly.genome.fa'
        print('Indexing reference FASTA...', flush=True)
        expected = fasta_index(fasta)
        index = Path(str(fasta)+'.fai')
        if index.exists():
            if index.read_text(encoding='utf-8') != expected:
                raise ValueError(f'FASTA index differs; left untouched: {index}')
        else:
            fd, filename = tempfile.mkstemp(prefix='.index_', dir=fasta.parent)
            temporary = Path(filename)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                    stream.write(expected)
                    stream.flush()
                    os.fsync(stream.fileno())
                if os.name == 'nt':
                    os.rename(temporary, index)
                else:
                    os.link(temporary, index)
            finally:
                temporary.unlink(missing_ok=True)
        marker = root/'setup_complete.json'
        if marker.exists():
            import json
            receipt = json.loads(marker.read_text(encoding='utf-8'))
            if not all((root/p).is_file() and sha256(root/p) == digest
                       for p, digest in receipt['outputs'].items()):
                raise ValueError('Prepared resources changed; use a new -R directory.')
            print('Reference resources ready (verified existing setup).', flush=True)
            return
        asset_dir = root/'dataset/interim/feature_assets'
        if asset_dir.exists() and any(asset_dir.iterdir()):
            raise ValueError('Existing derived assets found without a setup receipt. '
                             'They were left untouched; use them for prediction or choose a new -R directory.')
        print('Building splice-site catalog and numerical feature assets...', flush=True)
        references(root)
        outputs = [p for p in (root/'dataset/interim').rglob('*') if p.is_file() and p.suffix != '.sqlite3']
        atomic_json(marker, {'status': 'PASS', 'outputs': {p.relative_to(root).as_posix(): sha256(p) for p in outputs}})
        print(f'Reference resources ready: {root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Prepare hg38 resources from a folder of downloaded files (offline).')
    parser.add_argument('-I', '--downloads', type=Path, required=True)
    parser.add_argument('-R', '--reference-root', type=Path, default=Path('refs'))
    args = parser.parse_args()
    try:
        setup(args.downloads, args.reference_root)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
