#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/30 20:46
# @Author  : even
# @File    : features_annotation.py
from features.utills.load_data import *
from features.utills.get_phyloP import *
from features.utills.exon_annotation import get_bed_annotation
import math

def format_text_to_dict(text):
    key_value_pairs = text.rstrip(';').split('; ')
    result_dict = {}

    for pair in key_value_pairs:
        key, value_str = pair.split(' ', 1)
        pure_value = value_str.strip('"')
        result_dict[key] = pure_value
    return result_dict

def get_region_dict(region,acceptor_info,donor_info):
    if region=='acceptor':
        region_info=acceptor_info
    elif region=='donor':
        region_info=donor_info
    region_dict=format_text_to_dict(region_info)
    return region_dict

def get_dis(pos,region,strand,exon_start,exon_end):
    if strand=='+':
        if region=='acceptor':
            if pos>=exon_start:
                dis=pos-exon_start+1
            else:
                dis=pos-exon_start
        elif region=='donor':
            if pos<=exon_end:
                dis=pos-exon_end-1
            else:
                dis=pos-exon_end
    elif strand=='-':
        if region=='acceptor':
            if pos <= exon_end:
                dis = pos - exon_end - 1
            else:
                dis = pos - exon_end
        elif region=='donor':
            if pos >= exon_start:
                dis = pos - exon_start + 1
            else:
                dis = pos - exon_start
    else:
        dis=0
    return dis

def log_intron_length(intron_length, base='ln'):
    if intron_length < 0:
        raise ValueError("intron_length must be >= 0")
    if base == 'log10':
        return math.log10(intron_length + 1)
    elif base == 'ln':
        return math.log(intron_length + 1)
    else:
        raise ValueError("base must be 'log10' or 'ln'")

def get_intron_data(strand,region,exon_start,exon_end,last_exon_end,next_exon_start):
    if last_exon_end==0:
        last_intron_length=0
    else:
        last_intron_length=abs(int(exon_start)-int(last_exon_end))

    if next_exon_start==0:
        next_intron_length=0
    else:
        next_intron_length=abs(int(next_exon_start)-int(exon_end))

    if strand=='+':
        if region=='acceptor':
            intron_length=last_intron_length
        elif region=='donor':
            intron_length=next_intron_length
    elif strand=='-':
        if region=='acceptor':
            intron_length=next_intron_length
        elif region=='donor':
            intron_length=last_intron_length
    intron_length_logp=log_intron_length(intron_length)

    return intron_length, intron_length_logp

def get_flanking_start_end(strand,region_da,boundary,FLANKING_LEN=200):
    r_region_dict = {
        'donor_+': [boundary - FLANKING_LEN+1, boundary + FLANKING_LEN+1],
        'donor_-': [boundary - FLANKING_LEN-2, boundary + FLANKING_LEN-2],
        'acceptor_+': [boundary - FLANKING_LEN-2, boundary + FLANKING_LEN -2],
        'acceptor_-': [boundary - FLANKING_LEN+1, boundary + FLANKING_LEN+1],
    }
    key=region_da+'_'+strand
    start = r_region_dict[key][0]
    end = r_region_dict[key][1]
    return [start,end]

def features_annotation(variant_df,region='acceptor'):
    # fasta
    genome=get_fasta()

    # phyloP100
    bw_phyloP=load_data_phylop()

    # phastCons100way
    bw_phastCons=load_data_phastCons()

    # create new df
    basic_columns = ['chrom', 'pos', 'ref', 'alt','transcript']
    feature_df = variant_df[basic_columns]

    for idx,row in variant_df.iterrows():
        #----------------------
        #-------data load------
        #----------------------
        # index	pos_key	chrom	pos	ref	alt	NM_id source_ExtremeSplicing_variant_class_clinvar
        chrom=str(row['chrom'])
        pos=int(row['pos'])
        ref=row['ref'].upper()
        alt=row['alt'].upper()

        acceptor_info, donor_info=row['acceptor_info'],row['donor_info']

        region_dict=get_region_dict(region,acceptor_info,donor_info)
        exon_start,exon_end=int(region_dict['start']),int(region_dict['end'])
        strand=region_dict['strand']

        acceptor_info=row['acceptor_info']
        donor_info=row['donor_info']
        acceptor_dict=format_text_to_dict(acceptor_info)
        donor_dict=format_text_to_dict(donor_info)
        if region=='acceptor':
            region_dict=acceptor_dict
            boundary=int(region_dict['acceptor_pos'])
        elif region=='donor':
            region_dict=donor_dict
            boundary=int(region_dict['donor_pos'])
        longseq_start, longseq_end = get_flanking_start_end(strand, region, boundary, FLANKING_LEN=5000)
        longseq=genome["chr"+chrom][longseq_start:longseq_end].seq

        #----------------------------------------------------------------------------------------
        #--------------------------------get features data---------------------------------------
        #----------------------------------------------------------------------------------------
        # dis
        dis=get_dis(pos,region,strand,exon_start,exon_end)
        # exon_rank
        exon_rank=int(region_dict['exon_number'])
        # exon_last
        exon_count=int(region_dict['exon_count'])
        if exon_count-exon_rank==0:
            exon_last=1
        else:
            exon_last=0
        # exon_length/exon_length_mod3
        exon_length=exon_end-exon_start+1
        exon_length_mod3=exon_length%3
        # intron_length
        # intron_length_logp
        last_exon_end,next_exon_start=region_dict.get('prev_exon_end',0),region_dict.get('Next_exon_start',0)
        intron_length,intron_length_logp=get_intron_data(strand,region,exon_start,exon_end,last_exon_end,next_exon_start)
        # gc
        gc_5, gc_10, gc_25, gc_50, gc_100, gc_200 = compute_multi_gc(longseq)
        # merge
        basic_list=[strand,dis,exon_rank,exon_last,exon_length,exon_length_mod3,intron_length,intron_length_logp,gc_5, gc_10, gc_25, gc_50, gc_100, gc_200]
        feature_df.loc[idx,['strand','dis','exon_rank','exon_last','exon_length','exon_length_mod3','intron_length','intron_length_logp','gc_5', 'gc_10', 'gc_25', 'gc_50', 'gc_100', 'gc_200']]=basic_list

        # phyloP phastCons
        phylop_score=get_phylop_score(bw_phyloP, chrom, pos)
        # phastCons
        pc_5,pc_10,pc_20,pc_50=get_phastCons_score(bw_phastCons,chrom,pos)
        feature_df.loc[idx, ['phylop_score','phastCons_5','phastCons_10','phastCons_20','phastCons_50']] =[phylop_score,pc_5,pc_10,pc_20,pc_50]

        # longseq
        refseq=longseq.upper()
        feature_df.loc[idx, ['refseq']] =refseq
    close_bw(bw_phyloP)
    close_bw(bw_phastCons)
    return feature_df

def main():

    variant_path='../../dataset/toy.gtf.txt'
    feature_path = '../../dataset/toy.features.donor.txt'

    variant_df=get_bed_annotation(
        variant_path=variant_path,
    )

    feature_acceptor_df=features_annotation(
        variant_df=variant_df,
        feature_path=feature_path,
        region='acceptor')

    feature_donor_df=features_annotation(
        variant_df=variant_df,
        feature_path=feature_path,
        region='donor')


if __name__ == '__main__':
    main()