#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/30 17:29
# @Author  : even
# @File    : exon_annotation.py
import pandas as pd
from features.utills.load_data import extract_gtf_info_fields

def dict_to_gtf_info_str(info_dict):
    if not isinstance(info_dict, dict) or len(info_dict) == 0:
        return ""
    key_value_str_list = []
    for key, value in info_dict.items():
        val_str = str(value)
        key_value_str = f'{key} "{val_str}"'
        key_value_str_list.append(key_value_str)
    return "; ".join(key_value_str_list) + ";"

def parse_gtf_info_to_dict(info_str):
    info_dict = {}
    if pd.isna(info_str) or not isinstance(info_str, str):
        return info_dict
    key_value_pairs = [pair.strip() for pair in info_str.rstrip(';').split('; ')]
    for pair in key_value_pairs:
        if not pair:
            continue
        parts = pair.split(' ', 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip().strip('"')
        try:
            if value.isdigit():
                value = int(value)
        except:
            pass
        info_dict[key] = value
    return info_dict

def get_acceptor_donor_match(pos, gtf_ts_df):
    acceptor_match = {}
    donor_match = {}

    if gtf_ts_df.empty or not isinstance(pos, int):
        return acceptor_match, donor_match

    df = gtf_ts_df.copy()

    df['info_dict'] = df['info'].apply(parse_gtf_info_to_dict)
    df['acceptor_pos'] = df['info_dict'].apply(lambda x: x.get('acceptor_pos', pd.NA))
    df['donor_pos'] = df['info_dict'].apply(lambda x: x.get('donor_pos', pd.NA))

    acceptor_valid_df = df.dropna(subset=['acceptor_pos']).copy()
    donor_valid_df = df.dropna(subset=['donor_pos']).copy()

    if not acceptor_valid_df.empty:
        acceptor_valid_df['distance'] = acceptor_valid_df['acceptor_pos'].apply(lambda x: abs(x - pos))
        min_acceptor_distance = acceptor_valid_df['distance'].min()
        closest_acceptor_df = acceptor_valid_df[acceptor_valid_df['distance'] == min_acceptor_distance].head(1)

        if not closest_acceptor_df.empty:
            row = closest_acceptor_df.iloc[0]
            basic_dict = {
                'start': str(row['start']),
                'end': str(row['end']),
                'strand': str(row['strand'])
            }
            info_inner_dict = row['info_dict']
            acceptor_match = {**basic_dict, **info_inner_dict}
            acceptor_pos_value = row['acceptor_pos']
            acceptor_dis_value = pos - acceptor_pos_value
            acceptor_match['acceptor_dis'] = acceptor_dis_value

    if not donor_valid_df.empty:
        donor_valid_df['distance'] = donor_valid_df['donor_pos'].apply(lambda x: abs(x - pos))
        min_donor_distance = donor_valid_df['distance'].min()
        closest_donor_df = donor_valid_df[donor_valid_df['distance'] == min_donor_distance].head(1)

        if not closest_donor_df.empty:
            row = closest_donor_df.iloc[0]
            basic_dict = {
                'start': str(row['start']),
                'end': str(row['end']),
                'strand': str(row['strand'])
            }
            info_inner_dict = row['info_dict']
            donor_match = {**basic_dict, **info_inner_dict}
            donor_pos_value = row['donor_pos']  #
            donor_dis_value = pos - donor_pos_value
            donor_match['donor_dis'] = donor_dis_value

    return acceptor_match, donor_match

# gtf annotation
def gtf_annotation(pos,gtf_df,ts_MANE):
    # select MANE nm_id
    ts_id_list=ts_MANE[(ts_MANE['start'] <= pos) & (ts_MANE['end'] >= pos)]['transcript_id'].values
    if len(ts_id_list)>0:
        ts_id=ts_id_list[0]
        gtf_ts_df=gtf_df[(gtf_df['transcript_id']==ts_id) & (gtf_df['structure_region']=='exon')]
    acceptor_match, donor_match=get_acceptor_donor_match(pos, gtf_ts_df)
    return acceptor_match, donor_match,ts_id_list

def get_bed_annotation(variant_path=None):
    gtf = extract_gtf_info_fields()
    variant_df = pd.read_csv(variant_path, sep='\t', dtype={'chrom': str})
    ts_MANE = gtf[(gtf['structure_region'] == 'transcript') & (gtf['isMANE'] == '1')]
    chrom_list = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                  '20', '21',
                  '22', 'X']
    variant_df['chrom'] = pd.Categorical(variant_df['chrom'], categories=chrom_list, ordered=True)

    variant_df['transcript'] = ""
    variant_df['acceptor_info'] = ""
    variant_df['donor_info'] = ""

    for chrom in chrom_list:
        gtf_chrom=gtf[gtf['chrom']=='chr'+chrom]
        variant_chrom=variant_df[variant_df['chrom']==chrom]
        ts_MANE_chrom=ts_MANE[ts_MANE['chrom']=='chr'+chrom]
        for idx, row in variant_chrom.iterrows():
            pos = row['pos']
            acceptor_info_dict,donor_info_dict,ts_id_list=gtf_annotation(pos,gtf_chrom,ts_MANE_chrom)
            acceptor_info_str = dict_to_gtf_info_str(acceptor_info_dict)
            donor_info_str = dict_to_gtf_info_str(donor_info_dict)

            variant_df.loc[idx, 'transcript'] = ts_id_list
            variant_df.loc[idx, 'acceptor_info'] = acceptor_info_str
            variant_df.loc[idx, 'donor_info'] = donor_info_str

        # variant_df.to_csv(outpath, sep='\t', index=False,quoting=csv.QUOTE_NONE)
    return variant_df

def main(
        variant_path=None,
        outpath=None
):
    get_bed_annotation(
        variant_path=variant_path,
        outpath=outpath
    )

if __name__ == '__main__':
    main(
        variant_path='../../dataset/toy.vcf',
        outpath=f'../../dataset/toy.gtf.txt'
    )
