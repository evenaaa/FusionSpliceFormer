#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 12:05
# @Author  : even
# @File    : get_rbp.py
from utills.load_data import *
def get_rbp_altseq(rbp_seq,alt):
    alt_relative_pos=99
    altseq = str(rbp_seq[0:alt_relative_pos]) + alt + str(rbp_seq[alt_relative_pos:])
    return altseq

def get_rbp_match_index(site_list,seq):
    match_index=[]
    for i in range(len(site_list)):
        site=site_list[i]
        if site in seq:
            match_index.append(i)
    return match_index

def get_rbp_one_score(match_index,rbp_df):
    pos_rbpscore,neg_rbpscore=0,0
    for index in match_index:
        line=rbp_df.iloc[index]
        # A positive score was assigned to the target sequences that facilitate exon definition that is ESE (exonic splicing enhancer) and ISS (intronic splicing silencer) motifs.
        # According to the same criteria we assigned a negative score to the target sequences that facilitate intron definition that is ESS (exonic splicing silencer) and ISE (intronic splicing enhancer) motifs.
        score=line['SCORE']
        if score>0:
            pos_rbpscore+=score
        else:
            neg_rbpscore+=score
    return pos_rbpscore,neg_rbpscore

def get_rbp_scores(refseq,altseq,rbp_df):
    site_list=list(rbp_df['SITE'])

    # get match index of ref & alt
    match_index_ref = get_rbp_match_index(site_list, seq=refseq)
    match_index_alt = get_rbp_match_index(site_list, seq=altseq)
    # get score
    ref_pos_rbpscore,ref_neg_rbpscore = get_rbp_one_score(match_index_ref, rbp_df)
    alt_pos_rbpscore,alt_neg_rbpscore = get_rbp_one_score(match_index_alt, rbp_df)
    ref_rbpscore=ref_pos_rbpscore+ref_neg_rbpscore
    alt_rbpscore=alt_pos_rbpscore+alt_neg_rbpscore
    difference_rbpscore=alt_rbpscore-ref_rbpscore
    relative_rbpscore=get_relative_score(refscore=ref_rbpscore, altscore=alt_rbpscore)

    rbpscores_list=[ref_pos_rbpscore,ref_neg_rbpscore,alt_pos_rbpscore,alt_neg_rbpscore,ref_rbpscore,alt_rbpscore,difference_rbpscore,relative_rbpscore]
    # print(rbpscores_list)
    return rbpscores_list