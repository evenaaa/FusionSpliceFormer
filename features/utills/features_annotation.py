#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/30 20:46
# @Author  : even
# @File    : features_annotation.py
from utills.get_r import *
from utills.get_rbp import *
from utills.load_data import *
from utills.get_mes_score5 import get_score5
from utills.get_mes_score3 import get_score3
from utills.get_esr import *
from utills.get_phyloP import *

def get_base400altseq(alt,pos,refseq,region):
    region_start = int(region.split(':')[1].split('-')[0]) + 1
    relative_pos = int(pos - region_start)
    preseq = refseq[:relative_pos]
    endseq = refseq[relative_pos + 1:]
    altseq = preseq + alt + endseq

    return altseq

# mes
def get_mes_altseq(pos,mes_location,mes_seq,alt):
    location_start=int(mes_location.split(':')[1].split('-')[0])+1
    location_end=int(mes_location.split(':')[1].split('-')[1])
    if pos>=location_start and pos<=location_end:
        relative_pos=pos-location_start
        seq1=mes_seq[:relative_pos]
        seq3=mes_seq[relative_pos+1:]
        altseq=seq1+alt+seq3
    else:
        altseq=mes_seq
    return altseq

def get_messcores(mes_seq,region_da,mes3dict,mes5dict):
    if region_da=='donor':
        mes_score=get_score5(mes_seq,mes5dict)
    elif region_da=='acceptor':
        mes_score=get_score3(mes_seq,mes3dict)
    return mes_score

def get_mes_features(strand,mes_seq,alt,pos,mes_location,region_da,mes3dict,mes5dict):
    mes_refseq = get_strand_seq(strand, mes_seq)
    mes_altseq = get_mes_altseq(pos, mes_location, mes_seq, alt)
    mes_altseq = get_strand_seq(strand, mes_altseq)
    mes_ref = get_messcores(mes_refseq, region_da, mes3dict, mes5dict)
    mes_alt = get_messcores(mes_altseq, region_da, mes3dict, mes5dict)
    diffrence_mes = mes_alt - mes_ref
    relative_mes=get_relative_score(refscore=mes_ref,altscore=mes_alt)

    messcores_list = [mes_ref, mes_alt, diffrence_mes,relative_mes]

    return messcores_list


def features_annotation(gtf_df,feature_path='dataset/annodata/merge.features.txt'):

    # freq matrix
    donor_freq_matrix,acceptor_freq_matrix=get_region_freq_matrix(gtf_df)

    # esr dict
    esr_dict = load_data_esr_dict()
    # mes dict
    mes5dict=load_data_mes5dict()
    mes3dict=load_data_mes3dict()

    # rbp: spliceaid df
    rbp_df = load_data_spliceaid()

    # phyloP100
    bw=load_data_phylop()

    # create new df
    # 选取指定的列创建新的 DataFrame
    basic_columns = ['chrom', 'pos', 'ref', 'alt', 'boundary', 'dis', 'region_da','region3',
                    'exon_start', 'exon_end', 'ts_start', 'ts_end', 'strand',
                    'gene_id', 'transcript_id', 'refseq_ts_id', 'gene_name',
                    'transcript_name', 'exon_number', 'exon_counts', 'protein_id',
                    'exon_id']
    feature_df = gtf_df[basic_columns]

    for idx,row in gtf_df.iterrows():
        #----------------------
        #-------data load------
        #----------------------
        # index	pos_key	chrom	pos	ref	alt	NM_id source_ExtremeSplicing_variant_class_clinvar
        chrom=str(row['chrom'])
        pos=int(row['pos'])
        exon_start=row['exon_start']
        exon_end=row['exon_end']
        strand=row['strand']
        ref=row['ref'].upper()
        alt=row['alt'].upper()
        region_da=row['region_da']

        r_seq=row['r_seq'].upper()
        r_location=row['r_location']
        r_freq_location=row['base400_location']
        r_freq_seq=row['base400_seq'].upper()
        # print(chrom,pos,ref,alt)
        # print('base400_seq',row['base400_seq'])
        # print(f'r_sw_seq',row['r_sw_seq'])
        r_sw_seq = row['r_sw_seq'].upper()
        r_next_seq=str(row['r_next_seq']).upper()

        esr_seq=row['esr_seq'].upper()
        mes_seq=row['mes_seq'].upper()
        mes_location=row['mes_location']
        rbp_seq=row['spliceaid_seq'].upper()
        #----------------------------------------------------------------------------------------
        #--------------------------------get features data---------------------------------------
        #----------------------------------------------------------------------------------------

        # 1.exon length
        exon_length=exon_end-exon_start+1
        exon_length_mod3=exon_length%3
        basic_list=[exon_length,exon_length_mod3]
        feature_df.loc[idx,['exon_length','exon_length_mod3']]=basic_list

        # 2. Rican ref:Information content (Ri) of the closest canonical donor site.
        # ri_can_ref, ri_can_alt, difference_ri_can, relative_ri_can,
        #                      max_Ri_cryptic_donor_window_ref, max_Ri_cryptic_donor_window_alt, difference_ri_crypt,relative_ri_crypt,
        #                      difference_r_next
        r_scores_list = get_r_features(strand, r_seq, pos, alt, region_da, r_location, donor_freq_matrix,
                                       acceptor_freq_matrix,r_sw_seq, r_freq_location, r_freq_seq, r_next_seq,ref)
        feature_df.loc[idx, ['ri_can_ref', 'ri_can_alt', 'diff_ri_can', 'relative_ri_can',
                     'max_Ri_cryptic_donor_window_ref', 'max_Ri_cryptic_donor_window_alt', 'diff_ri_crypt','relative_ri_crypt',
                     'diff_r_next']] = r_scores_list

        # 3.esr:ese_ref, ese_alt, ess_ref, ess_alt, differnce_refesr, differnce_altesr, differnce_esr,relative_esr
        esrscores_list=get_esr_features(strand,esr_seq,alt,esr_dict)
        feature_df.loc[idx, ['ese_ref', 'ese_alt', 'ess_ref', 'ess_alt', 'diff_refesr', 'diff_altesr', 'diff_esr',
                         'relative_esr']] =esrscores_list

        # 4.maxentscan:mes_ref, mes_alt, diffrence_mes,relative_mes
        messcores_list=get_mes_features(strand,mes_seq,alt,pos,mes_location,region_da,mes3dict,mes5dict)
        feature_df.loc[idx, ['mes_ref', 'mes_alt', 'diff_mes','relative_mes']] =messcores_list

        # 5.rbp:ref_pos_rbpscore,ref_neg_rbpscore,alt_pos_rbpscore,alt_neg_rbpscore,ref_rbpscore,alt_rbpscore,difference_rbpscore,relative_rbpscore
        rbp_refseq=get_strand_seq(strand,rbp_seq)
        rbp_altseq=get_rbp_altseq(rbp_seq,alt)
        rbpscores_list=get_rbp_scores(refseq=rbp_refseq, altseq=rbp_altseq, rbp_df=rbp_df)
        feature_df.loc[idx, ['ref_pos_rbpscore','ref_neg_rbpscore','alt_pos_rbpscore','alt_neg_rbpscore','ref_rbpscore','alt_rbpscore',
                         'diff_rbpscore','relative_rbpscore']] =rbpscores_list

        # 6.phyloP
        phylop_score=get_phylop_score(bw, chrom, pos)
        feature_df.loc[idx, ['phylop_score']] =phylop_score

        # 7.seq 400bp
        base400_refseq=row['base400_seq'].upper()
        base400_location=row['base400_location']
        feature_df.loc[idx, ['base400_refseq']] =base400_refseq
        base400_altseq=get_base400altseq(alt=alt,pos=pos,refseq=base400_refseq,region=base400_location)
        feature_df.loc[idx, ['base400_altseq']] =base400_altseq

        #--------------------------------------------------------------------------------
        #------------------------------output--------------------------------------------
        #--------------------------------------------------------------------------------
        # print(idx,chrom,pos,ref,alt)
    feature_df.to_csv(feature_path,sep='\t',index=False)
    close_bw(bw)
    return feature_df
