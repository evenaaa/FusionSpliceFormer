"""Disk-indexed VCF prediction with bounded event chunks and resumable results."""
import json
import math
from pathlib import Path
import shutil
import sqlite3

import pandas as pd

from .input import SIMPLE_COLUMNS
from .output import OUTPUT_COLUMNS, SUMMARY_COLUMNS, joined_identifiers
from .vcf import SiteIndex, chromosome_name, variants
from .features.numeric.sequence import IndexedFasta


class VcfStore:
    """Keep global identities, input order and committed scores on disk."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS alleles (
                id INTEGER PRIMARY KEY, chrom TEXT, pos INTEGER, ref TEXT, alt TEXT,
                status TEXT, UNIQUE(chrom,pos,ref,alt));
            CREATE TABLE IF NOT EXISTS occurrences (
                seq INTEGER PRIMARY KEY, allele INTEGER REFERENCES alleles(id), identifier TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, allele INTEGER REFERENCES alleles(id),
                site_pos INTEGER, strand TEXT, site_type TEXT, distance INTEGER,
                score REAL, report TEXT, UNIQUE(allele,site_pos,strand,site_type));
            CREATE INDEX IF NOT EXISTS ranked_targets ON events
                (allele,score DESC,distance,site_pos,strand,site_type);
            CREATE TABLE IF NOT EXISTS annotations (site TEXT PRIMARY KEY, gene TEXT, transcripts TEXT);
        ''')

    def close(self):
        self.db.close()

    def get(self, key, default=0):
        row = self.db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, int(value)))

    def counts(self):
        return (self.get('records'), self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],
                self.db.execute("SELECT COUNT(*) FROM occurrences o JOIN alleles a ON a.id=o.allele WHERE a.status!='OK'").fetchone()[0])

    def index_vcf(self, path, catalog, fasta, progress=None, commit_every=1000):
        """Stream input into SQLite; restart scans past already committed alleles.

        An occurrence and all of its targets commit together, including for
        multi-ALT records. No complete event list or VCF is retained in RAM.
        """
        if self.get('indexed'):
            return self.counts()
        if commit_every < 1:
            raise ValueError('commit_every must be positive')
        previous = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM occurrences').fetchone()[0]
        index = SiteIndex(catalog)
        try:
            with IndexedFasta(fasta) as reference:
                for seq, (record, source_chrom, pos, identifier, ref, alt) in enumerate(variants(path), 1):
                    if seq <= previous:
                        continue
                    chrom = chromosome_name(source_chrom)
                    key = (chrom or source_chrom, pos, ref, alt)
                    found = self.db.execute('SELECT id FROM alleles WHERE chrom=? AND pos=? AND ref=? AND alt=?', key).fetchone()
                    if found:
                        allele = found[0]
                    else:
                        status = 'OK'
                        if chrom is None or chrom not in reference.index:
                            status = 'UNSUPPORTED_CONTIG'
                        elif pos > reference.index[chrom][0]:
                            status = 'OUT_OF_RANGE'
                        elif len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
                            status = 'UNSUPPORTED_ALLELE'
                        elif reference.fetch_padded(chrom, pos, 1).upper() != ref:
                            status = 'REF_MISMATCH'
                        allele = self.db.execute('INSERT INTO alleles(chrom,pos,ref,alt,status) VALUES (?,?,?,?,?)', (*key, status)).lastrowid
                        count = 0
                        if status == 'OK':
                            for site in index.targets(chrom, pos):
                                self.db.execute('INSERT INTO events(allele,site_pos,strand,site_type,distance) VALUES (?,?,?,?,?)',
                                                (allele, site.anchor_1based, site.strand, site.site_type, abs(pos-site.anchor_1based)))
                                count += 1
                            if not count:
                                self.db.execute("UPDATE alleles SET status='NO_TARGET' WHERE id=?", (allele,))
                    self.db.execute('INSERT INTO occurrences VALUES (?,?,?)', (seq, allele, identifier))
                    self.set('records', record)
                    if seq % commit_every == 0:
                        self.db.commit()
                        if progress: progress(record)
            self.set('indexed', 1)
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return self.counts()

    def next_chunk(self, size):
        if size < 1:
            raise ValueError('Chunk size must be positive')
        cursor = self.db.execute('''SELECT e.id,a.chrom,a.pos,a.ref,a.alt,e.site_pos,e.strand,e.site_type
            FROM events e JOIN alleles a ON a.id=e.allele WHERE e.id>? ORDER BY e.id LIMIT ?''',
            (self.get('completed'), size))
        rows = cursor.fetchall()
        return [r[0] for r in rows], pd.DataFrame([r[1:] for r in rows], columns=SIMPLE_COLUMNS)

    def commit_chunk(self, ids, frame, report, coverage=None):
        """Scores and completion cursor are one transaction; no rounded scores."""
        if not ids or ids != list(range(self.get('completed')+1, ids[-1]+1)):
            raise ValueError('Unexpected chunk event order')
        expected = {tuple(row): eid for row, eid in zip(frame.itertuples(index=False, name=None), ids)}
        if len(expected) != len(ids) or len(report) != len(ids):
            raise ValueError('Chunk prediction row count differs')
        updates = []
        for row in report.to_dict('records'):
            key = (row['CHROM'], int(row['POS']), row['REF'], row['ALT'], int(row['SITE_POS']), row['STRAND'], row['SITE_TYPE'])
            if key not in expected:
                raise ValueError('Chunk prediction identities differ')
            score = float(row['SCORE'])
            if row['STATUS'] != 'OK' or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid committed prediction')
            updates.append((score, json.dumps(row, ensure_ascii=False, allow_nan=False), expected.pop(key)))
        if expected:
            raise ValueError('Chunk prediction identities differ')
        with self.db:
            self.db.executemany('UPDATE events SET score=?,report=? WHERE id=?', updates)
            self.set('completed', ids[-1])
            if coverage is not None:
                if coverage['row_count'] != len(ids):
                    raise ValueError('Chunk conservation row count differs')
                for name in ('phylop_coverage', 'phastcons_coverage'):
                    value = float(coverage[name])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('Invalid conservation coverage')
                    self.set(name, self.get(name)+round(value*len(ids)))
                self.set('coverage_rows', self.get('coverage_rows')+len(ids))

    def validate_coverage(self, minimum):
        """Apply the existing threshold across the whole unique-event set."""
        total = self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        if self.get('coverage_rows') != total:
            raise ValueError('Incomplete conservation coverage accounting')
        if total:
            for name in ('phylop_coverage', 'phastcons_coverage'):
                value = self.get(name)/total
                if value < minimum:
                    raise RuntimeError(f'Conservation coverage is below the required threshold {minimum:.2%}: '
                                       f'{name}={value:.6f}. Check BigWig paths, chromosome naming, and GRCh38 coordinates.')

    def prepare_annotations(self, catalog):
        if self.get('annotations_ready'):
            return
        columns = ['physical_site_id', 'gene_symbols', 'gene_ids', 'refseq_transcript_ids', 'transcript_ids']
        with self.db:
            self.db.execute('DELETE FROM annotations')
            for chunk in pd.read_csv(catalog, sep='\t', dtype=str, keep_default_na=False, usecols=columns, chunksize=10000):
                rows = []
                for r in chunk.itertuples(index=False):
                    gene = joined_identifiers(r.gene_symbols)
                    transcript = joined_identifiers(r.refseq_transcript_ids)
                    rows.append((r.physical_site_id, joined_identifiers(r.gene_ids) if gene == '.' else gene,
                                 joined_identifiers(r.transcript_ids) if transcript == '.' else transcript))
                self.db.executemany('INSERT INTO annotations VALUES (?,?,?)', rows)
            self.set('annotations_ready', 1)

    def annotations(self, site_ids):
        result = {}
        for site in set(site_ids):
            row = self.db.execute('SELECT gene,transcripts FROM annotations WHERE site=?', (site,)).fetchone()
            if row is None:
                raise ValueError(f'Target annotation missing: {site}')
            result[site] = row
        return result

    def output_rows(self, details=False):
        if not self.get('indexed') or self.db.execute('SELECT COUNT(*) FROM events WHERE report IS NULL').fetchone()[0]:
            raise ValueError('VCF predictions are incomplete; final output not published')
        if details:
            cursor = self.db.execute('''SELECT a.chrom,a.pos,a.ref,a.alt,a.status,o.identifier,e.report
                FROM occurrences o JOIN alleles a ON a.id=o.allele
                LEFT JOIN events e ON e.allele=a.id ORDER BY o.seq,e.id''')
        else:
            cursor = self.db.execute('''SELECT a.chrom,a.pos,a.ref,a.alt,a.status,'.',
                (SELECT e.report FROM events e WHERE e.allele=a.id
                 ORDER BY e.score DESC,e.distance,e.site_pos,e.strand,e.site_type LIMIT 1)
                FROM alleles a ORDER BY a.id''')
        for chrom, pos, ref, alt, status, identifier, payload in cursor:
            if payload is None:
                row = dict.fromkeys(OUTPUT_COLUMNS, '.')
                row.update(CHROM=chrom, POS=pos, REF=ref, ALT=alt, STATUS=status, PREDICTION='Not_scored')
            else:
                row = json.loads(payload)
            row['ID'] = identifier
            yield row


def remove_chunk(folder, work):
    """Remove only the fixed, owned scratch subdirectory after score commit."""
    folder, work = Path(folder), Path(work).resolve()
    if folder.is_symlink() or folder.resolve() != work/'active_chunk':
        raise ValueError('Refusing to remove an unowned chunk directory')
    if folder.exists():
        shutil.rmtree(folder)


def run_vcf(args, device, work, paths):
    from .console import Activity, ChunkProgress
    from .io import atomic_json, sha256
    from .predict import predict_ensemble, load_ensemble, display_score
    from .prepare import prepare_events
    from .runtime import publish_rows
    from .features.numeric.pipeline import NumericFeaturePipeline
    from .features.numeric.config import FeatureSettings

    work = Path(work)
    store = VcfStore(work/'vcf.sqlite')
    bundle = pipeline = fasta_digest = None
    print(f'FusionSpliceFormer | {"GPU" if device == "cuda" else "CPU"} | {len(args.checkpoints)}-model ensemble', flush=True)
    try:
        with Activity('Preparing VCF', enabled=True) as activity:
            def mapping_progress(records):
                activity.label = f'Preparing VCF: {records:,} records indexed'
            records, total, unscored = store.index_vcf(args.cache_root, paths.site_catalog, paths.fasta, mapping_progress)
            activity.final_text = f'VCF: {records:,} records | {total:,} prediction events'
        if unscored:
            print(f'Notice: {unscored:,} unscored allele occurrences; retained as Not_scored.', flush=True)
        folder = work/'active_chunk'
        marker = folder/'chunk.json'
        with ChunkProgress(total, store.get('completed')) as progress:
            while True:
                ids, frame = store.next_chunk(args.chunk_size)
                if not ids:
                    # A crash after SQL commit may leave the last scratch chunk.
                    remove_chunk(folder, work)
                    break
                identity = {'first': ids[0], 'last': ids[-1], 'count': len(ids)}
                if folder.exists():
                    if folder.is_symlink():
                        raise ValueError('Chunk workspace must not be a symbolic link')
                    prior = json.loads(marker.read_text()) if marker.exists() else None
                    if prior != identity:
                        if prior is not None and prior['last'] > store.get('completed'):
                            raise ValueError('Saved chunk does not match remaining events')
                        remove_chunk(folder, work)
                folder.mkdir(exist_ok=True)
                atomic_json(marker, identity)
                source = folder/'events.tsv'
                # Rewriting input is safe; chunk progress identities use content hashes.
                frame.to_csv(source, sep='\t', index=False)
                progress.update(store.get('completed'), 'preparing')
                if pipeline is None:
                    pipeline = NumericFeaturePipeline(paths, FeatureSettings())
                    fasta_digest = sha256(paths.fasta)
                    store.prepare_annotations(paths.site_catalog)
                cache = prepare_events(source, folder, args.reference_root,
                    preprocessor_root=args.preprocessor_root, resume=True, progress=args.verbose,
                    numeric_pipeline=pipeline, fasta_digest=fasta_digest, annotation_lookup=store.annotations,
                    defer_coverage=True)
                if bundle is None:
                    bundle = load_ensemble(args.checkpoints, device, args.verbose)
                offset = store.get('completed')
                progress.update(offset, 'predicting')
                report = predict_ensemble(args.checkpoints, cache, folder/'unused.tsv', args.batch_size, device,
                    mixed_precision=args.amp, state_dir=folder/'scores', progress=False, verbose=args.verbose,
                    model_bundle=bundle, publish=False,
                    on_batch=lambda count: progress.update(offset+count, 'predicting'))
                coverage = json.loads((folder/'prediction/numeric/data.qc.json').read_text())
                store.commit_chunk(ids, frame, report, coverage)
                remove_chunk(folder, work)
            columns = OUTPUT_COLUMNS if args.details else SUMMARY_COLUMNS
            store.validate_coverage(FeatureSettings().minimum_conservation_coverage)
            progress.update(total, 'writing output')
            count = publish_rows(store.output_rows(args.details), columns, args.output, work, display_score)
            progress.update(total, 'complete')
        return count
    finally:
        store.close()
