"""Normalize variant and target-site input records."""
import pandas as pd
from .features.numeric.schema import FORBIDDEN_OUTPUT_COLUMNS

SIMPLE_COLUMNS = ('chrom', 'pos', 'ref', 'alt', 'site_pos', 'strand', 'site_type')


def read_events(path):
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    forbidden = FORBIDDEN_OUTPUT_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError('Remove label/evidence fields from inference input: ' + ', '.join(sorted(forbidden)))
    if frame.empty:
        raise ValueError('Input contains no events')
    if set(SIMPLE_COLUMNS).issubset(frame.columns):
        if any(c.startswith('grch38_') or c == 'physical_site_id' for c in frame.columns):
            raise ValueError('Use one consistent set of input column names')
        frame = frame.copy()
        for col in SIMPLE_COLUMNS:
            frame[col] = frame[col].str.strip()
        if not frame.chrom.str.fullmatch(r'chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M)').all():
            raise ValueError('Use chr-prefixed chromosomes (chr1 to chr22, chrX, chrY or chrM)')
        for col in ('pos', 'site_pos'):
            if not frame[col].str.fullmatch(r'[1-9][0-9]*').all():
                raise ValueError(f'{col} must contain positive 1-based integers')
            frame[col] = frame[col].astype('int64')
        for col in ('ref', 'alt'):
            frame[col] = frame[col].str.upper()
            if not frame[col].isin(list('ACGT')).all():
                raise ValueError('Only single-nucleotide A/C/G/T variants are supported')
        if frame.ref.eq(frame.alt).any():
            raise ValueError('REF and ALT must differ')
        if not frame.strand.isin(['+', '-']).all():
            raise ValueError('strand must be + or -')
        if not frame.site_type.isin(['acceptor', 'donor']).all():
            raise ValueError('site_type must be acceptor or donor')
        frame['variant_key'] = frame.chrom + ':' + frame.pos.astype(str) + ':' + frame.ref + ':' + frame.alt
        frame['physical_site_id'] = frame.chrom + ':' + frame.site_pos.astype(str) + ':' + frame.strand + ':' + frame.site_type
        frame['event_key'] = frame.variant_key + '|' + frame.physical_site_id
        frame['event_group_id'] = frame.event_key
        frame['transcript_oriented_distance'] = (frame.pos - frame.site_pos) * frame.strand.map({'+': 1, '-': -1})
        frame['absolute_distance'] = frame.transcript_oriented_distance.abs()
        for col in ('chrom', 'pos', 'ref', 'alt'):
            frame['grch38_' + col] = frame[col]
    required = {'event_group_id', 'event_key', 'variant_key', 'physical_site_id', 'strand', 'site_type',
                'grch38_chrom', 'grch38_pos', 'grch38_ref', 'grch38_alt',
                'transcript_oriented_distance', 'absolute_distance'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('Expected TSV columns: ' + ', '.join(SIMPLE_COLUMNS) + '; see README.md (Input)')
    if not frame.event_key.is_unique or not frame.event_group_id.is_unique:
        raise ValueError('Duplicate variant/site events or event IDs')
    if not frame.site_type.isin(['acceptor', 'donor']).all():
        raise ValueError('site_type must be acceptor or donor')
    return frame
