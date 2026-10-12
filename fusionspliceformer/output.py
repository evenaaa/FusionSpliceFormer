"""Public prediction columns and target-site annotations."""
import math
import pandas as pd

OUTPUT_COLUMNS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'GENE', 'TRANSCRIPTS',
                  'SITE_POS', 'STRAND', 'SITE_TYPE', 'DISTANCE_BP',
                  'SCORE', 'PREDICTION', 'STATUS']
COMPACT_COLUMNS = ['CHROM', 'POS', 'REF', 'ALT', 'GENE', 'TRANSCRIPTS', 'SITE_POS',
                   'STRAND', 'SITE_TYPE', 'SCORE', 'PREDICTION']
SUMMARY_COLUMNS = COMPACT_COLUMNS.copy()


def select_output(report, *, details=False, summarize=False):
    """Keep event rows, or summarize VCF alleles by their highest-scoring site."""
    result = report.copy()
    result.loc[result.STATUS.ne('OK'), 'PREDICTION'] = 'Not_scored'
    if summarize and not details:
        return summarize_variants(result)
    return result.loc[:, OUTPUT_COLUMNS if details else COMPACT_COLUMNS]


def summarize_variants(report):
    """One row per unique variant allele, in first-appearance order.

    Choose on unrounded scores. Exact ties use distance, site coordinate,
    strand and type, in that order. All annotations come from the chosen row;
    transcripts from different targets are never combined.
    """
    selected, priorities = {}, {}
    for row in report.to_dict('records'):
        key = (row['CHROM'], int(row['POS']), row['REF'], row['ALT'])
        if row['STATUS'] == 'OK':
            score = float(row['SCORE'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid score in variant summary')
            site = int(row['SITE_POS'])
            priority = (-score, abs(int(row['POS']) - site), site,
                        row['STRAND'], row['SITE_TYPE'])
        else:
            priority = (math.inf, math.inf, math.inf, '', '')
        if key not in selected or priority < priorities[key]:
            selected[key], priorities[key] = row, priority
    result = pd.DataFrame(selected.values(), columns=OUTPUT_COLUMNS)
    result.loc[result.STATUS.ne('OK'), 'PREDICTION'] = 'Not_scored'
    return result.loc[:, SUMMARY_COLUMNS]


def joined_identifiers(value):
    return ';'.join(sorted({part.strip() for part in str(value).split(';')
                            if part.strip() not in ('', '.')})) or '.'


def site_annotations(catalog, site_ids):
    """Use only transcripts annotated at the exact physical target site."""
    wanted = set(site_ids)
    annotations = {}
    columns = ['physical_site_id', 'gene_symbols', 'gene_ids',
               'refseq_transcript_ids', 'transcript_ids']
    for chunk in pd.read_csv(catalog, sep='\t', dtype=str, keep_default_na=False,
                             usecols=columns, chunksize=100000):
        for row in chunk.loc[chunk.physical_site_id.isin(wanted)].itertuples(index=False):
            if row.physical_site_id in annotations:
                raise ValueError(f'Duplicate target annotation: {row.physical_site_id}')
            genes = joined_identifiers(row.gene_symbols)
            if genes == '.':
                genes = joined_identifiers(row.gene_ids)
            transcripts = joined_identifiers(row.refseq_transcript_ids)
            if transcripts == '.':
                transcripts = joined_identifiers(row.transcript_ids)
            annotations[row.physical_site_id] = (genes, transcripts)
    missing = wanted - annotations.keys()
    if missing:
        raise ValueError(f'Target annotation missing for {len(missing)} sites')
    return annotations
