"""Read VCF variants and expand them into annotated variant–site events."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import re

import pandas as pd

from .data.windows import parse_physical_site_id, site_window
from .features.numeric.sequence import IndexedFasta
from .input import SIMPLE_COLUMNS
from .output import OUTPUT_COLUMNS

VCF_COLUMNS = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']


def open_text(path):
    path = Path(path)
    with path.open('rb') as handle:
        compressed = handle.read(2) == b'\x1f\x8b'
    return (gzip.open if compressed else open)(path, 'rt', encoding='utf-8-sig')


def is_vcf(path):
    """Recognize VCF headers and extensions, including gzip/BGZF files."""
    if str(path).lower().endswith(('.vcf', '.vcf.gz', '.vcf.bgz')):
        return True
    with open_text(path) as handle:
        first = handle.readline()
    return first.startswith(('##fileformat=VCF', '#CHROM\t'))


def chromosome_name(value):
    value = value.removeprefix('chr')
    if value in ('M', 'MT'):
        return 'chrM'
    if re.fullmatch(r'(?:[1-9]|1[0-9]|2[0-2]|X|Y)', value):
        return 'chr' + value
    return None


def variants(path):
    """Yield ALT alleles in record order; never use INFO or sample genotypes."""
    header = None
    version = None
    record = 0
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            if line.startswith('##'):
                if header is not None:
                    raise ValueError(f'VCF metadata after column header at line {line_number}')
                if line.startswith('##fileformat='):
                    if version is not None or not re.fullmatch(r'##fileformat=VCFv4\.[0-5]', line):
                        raise ValueError('Expected one VCFv4.0–4.5 fileformat declaration')
                    version = line
                if line.startswith(('##reference=', '##contig=')) and re.search(r'hg19|GRCh37|\bb37\b', line, re.I):
                    raise ValueError('VCF declares hg19/GRCh37; convert variants to GRCh38 and verify REF first')
                continue
            if line.startswith('#'):
                columns = line.split('\t')
                if header is not None or version is None or columns[:8] != VCF_COLUMNS:
                    raise ValueError('Expected VCF fileformat and tab-separated #CHROM/POS/ID/REF/ALT/QUAL/FILTER/INFO header')
                if len(columns) > 8 and (len(columns) < 10 or columns[8] != 'FORMAT'):
                    raise ValueError('VCF sample columns require FORMAT and at least one sample')
                if len(columns) != len(set(columns)):
                    raise ValueError('VCF column names must be unique')
                header = columns
                continue
            if header is None:
                raise ValueError(f'VCF header missing before line {line_number}')
            fields = line.split('\t')
            if len(fields) != len(header) or any(not field for field in fields):
                raise ValueError(f'Invalid VCF column count or empty field at line {line_number}')
            chrom, pos, identifier, ref, alts = fields[:5]
            if not re.fullmatch(r'[1-9][0-9]*', pos):
                raise ValueError(f'VCF POS must be a positive integer at line {line_number}')
            record += 1
            for alt in alts.split(','):
                if not alt:
                    raise ValueError(f'Empty VCF ALT at line {line_number}')
                yield record, chrom, int(pos), identifier, ref.upper(), alt.upper()
    if not record:
        raise ValueError('VCF contains no variant records')


class SiteIndex:
    def __init__(self, path):
        columns = ['physical_site_id', 'chrom', 'site_anchor_pos_1based', 'strand',
                   'site_type', 'default_mapping_eligible']
        table = pd.read_csv(path, sep='\t', usecols=columns, dtype=str, keep_default_na=False)
        if table.physical_site_id.duplicated().any():
            raise ValueError('Reference catalog contains duplicate physical sites')
        eligible = table.default_mapping_eligible.str.lower()
        if not eligible.isin(['true', 'false']).all():
            raise ValueError('Invalid mapping eligibility in reference catalog')
        table = table.loc[eligible.eq('true')].copy()
        table['site_anchor_pos_1based'] = pd.to_numeric(table.site_anchor_pos_1based, errors='raise')
        self.groups = {}
        for chrom, group in table.groupby('chrom', sort=False):
            sites = []
            for row in group.sort_values(['site_anchor_pos_1based', 'strand', 'site_type']).itertuples(index=False):
                site = parse_physical_site_id(row.physical_site_id)
                if (site.chrom, site.anchor_1based, site.strand, site.site_type) != (
                        row.chrom, row.site_anchor_pos_1based, row.strand, row.site_type):
                    raise ValueError(f'Inconsistent reference catalog site: {row.physical_site_id}')
                sites.append(site)
            self.groups[chrom] = ([site.anchor_1based for site in sites], sites)

    def targets(self, chrom, pos):
        anchors, sites = self.groups.get(chrom, ([], []))
        for site in sites[bisect_left(anchors, pos-5000):bisect_right(anchors, pos+5000)]:
            window = site_window(site)
            if window.start0 <= pos-1 < window.end0:
                yield site


@dataclass
class VcfMapping:
    events: pd.DataFrame
    rows: list
    record_count: int

    def save_events(self, path):
        """Write only prediction inputs; VCF annotations never enter features."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        try:
            self.events.to_csv(temporary, sep='\t', index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def report(self, predictions):
        lookup = {}
        for row in predictions.to_dict('records'):
            target = f'{int(row["SITE_POS"])}:{row["STRAND"]}:{row["SITE_TYPE"]}'
            key = (row['CHROM'], int(row['POS']), row['REF'], row['ALT'], target)
            if key in lookup:
                raise ValueError('Duplicate predicted variant–site pair')
            lookup[key] = row
        required = {(row['CHROM'], row['POS'], row['REF'], row['ALT'], row['TARGET_SITE'])
                    for row in self.rows if row['STATUS'] == 'OK'}
        if set(lookup) != required:
            raise ValueError('VCF mapping and predicted variant–site pairs differ')
        results = []
        for row in self.rows:
            result = {column: '.' for column in OUTPUT_COLUMNS}
            result.update({key: value for key, value in row.items() if key in OUTPUT_COLUMNS})
            result['SCORE'] = float('nan')
            if row['STATUS'] == 'OK':
                key = (row['CHROM'], row['POS'], row['REF'], row['ALT'], row['TARGET_SITE'])
                prediction = lookup[key]
                result.update(prediction)
                result['ID'] = row['ID']
            results.append(result)
        return pd.DataFrame(results, columns=OUTPUT_COLUMNS)


def map_vcf(path, catalog, fasta):
    """Expand every supported ALT to all eligible sites in its model window."""
    index = SiteIndex(catalog)
    rows, events, seen = [], [], set()
    record_count = 0
    with IndexedFasta(fasta) as reference:
        for record_count, source_chrom, pos, identifier, ref, alt in variants(path):
            chrom = chromosome_name(source_chrom)
            row = {'CHROM': chrom or source_chrom, 'POS': pos, 'ID': identifier,
                   'REF': ref, 'ALT': alt, 'TARGET_SITE': '.', 'STATUS': 'OK'}
            if chrom is None or chrom not in reference.index:
                row['STATUS'] = 'UNSUPPORTED_CONTIG'
            elif pos > reference.index[chrom][0]:
                row['STATUS'] = 'OUT_OF_RANGE'
            elif len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
                row['STATUS'] = 'UNSUPPORTED_ALLELE'
            elif reference.fetch_padded(chrom, pos, 1).upper() != ref:
                row['STATUS'] = 'REF_MISMATCH'
            if row['STATUS'] != 'OK':
                rows.append(row)
                continue
            targets = list(index.targets(chrom, pos))
            if not targets:
                rows.append(dict(row, STATUS='NO_TARGET'))
            for site in targets:
                target = f'{site.anchor_1based}:{site.strand}:{site.site_type}'
                rows.append(dict(row, TARGET_SITE=target))
                event = (chrom, pos, ref, alt, site.anchor_1based, site.strand, site.site_type)
                if event not in seen:
                    seen.add(event)
                    events.append(event)
    return VcfMapping(pd.DataFrame(events, columns=SIMPLE_COLUMNS), rows, record_count)
