#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 11:09
# @Author  : even
# @File    : get_r.py
from utills.load_data import *
import numpy as np
import pandas as pd
import os
import math

def complement(sequence):
    if len(sequence)>0:
        resequence=''
        for base in sequence:
            base=base.upper()
            if base == 'A':
                rebase='T'
            elif base == 'T':
                rebase='A'
            elif base == 'C':
                rebase='G'
            elif base == 'G':
                rebase='C'
            else:
                return None
            resequence=resequence+rebase
    resequence=resequence[::-1]
    return str(resequence)

def add_region_da_matrix(region_da,r_seq,counts,tmp_delete,row,region_da_counts,region_da_matrix):
    FLANKING_LEN=199

    ACGT_dict={
        'A':0,
        'C':1,
        'G':2,
        'T':3,
        'N':4
    }
    region_da_2base={
        'donor':'GT',
        'acceptor':'AG'
    }
    # if region_da == 'donor':
        # tmp = r_seq[499:501]
    tmp = r_seq[FLANKING_LEN:FLANKING_LEN + 2]
    if tmp != region_da_2base[region_da]:
        counts += 1
        tmp_delete.append(list(row))
    region_da_counts += 1
    for i in range(len(r_seq)):
            j = ACGT_dict[r_seq[i]]
            if j<4:
                region_da_matrix[i][j] += 1
    return tmp_delete,region_da_counts,region_da_matrix

def get_all_R_freq(variant_combined_df):
    tmp_delete=[]

    donor_matrix = [[0 for _ in range(4)] for _ in range(400)]
    acceptor_matrix = [[0 for _ in range(4)] for _ in range(400)]
    donor_counts=0
    acceptor_counts=0
    counts=0
    # for idx,row in variant_combined.iterrows():
    for idx,row in variant_combined_df.iterrows():
        strand=row['strand']
        r_freq_seq=row['r_freq_seq'].upper()
        region_da=row['region_da']

        if strand=='-':
            r_seq=complement(r_freq_seq)
        else:
            r_seq=r_freq_seq

        if region_da=='donor':
            tmp_delete,donor_counts,donor_matrix=add_region_da_matrix('donor', r_seq, counts, tmp_delete, row, donor_counts, donor_matrix)
        else:
            # acceptor
            tmp_delete,acceptor_counts,acceptor_matrix=add_region_da_matrix('acceptor', r_seq, counts, tmp_delete, row, acceptor_counts, acceptor_matrix)

    list2csv(tmp_delete,'dataset/tmp_delete.txt')
    # list2csv(tmp_delete,'../dataset/tmp_delete_donor.txt')
    # 将matrix中的每个元素都除以counts
    donor_matrix = [[element / donor_counts for element in row] for row in donor_matrix]
    acceptor_matrix = [[element / acceptor_counts for element in row] for row in acceptor_matrix]

    # df_donor_matrix=pd.DataFrame(donor_matrix)
    # df_acceptor_matrix=pd.DataFrame(acceptor_matrix)
    df_donor_matrix=pd.DataFrame(donor_matrix,columns=['A','C','G','T'])
    df_acceptor_matrix=pd.DataFrame(acceptor_matrix,columns=['A','C','G','T'])


    donor_matrix_path='dataset/annodata/freq/donor_r_freq.txt'
    acceptor_matrix_path='dataset/annodata/freq/acceptor_r_freq.txt'
    df_donor_matrix.to_csv(donor_matrix_path,sep='\t',index=False)
    df_acceptor_matrix.to_csv(acceptor_matrix_path,sep='\t',index=False)

    print(f'-------------------------------------------------------------')
    print(f'The data has been printed to file {donor_matrix_path}\t{acceptor_matrix_path}')
    print(f'-------------------------------------------------------------')
    return df_donor_matrix,df_acceptor_matrix

# get region freq matrix
def get_region_freq_matrix(variant_combined_df):
    freq_matrix_path='dataset/annodata/gtf/freq/freq.txt'

    freq_matrix_df=pd.read_csv(freq_matrix_path,sep='\t')
    donor_matrix_df, acceptor_matrix_df=freq_matrix_df[['donor_A','donor_C','donor_G','donor_T']],\
        freq_matrix_df[['acceptor_A','acceptor_C','acceptor_G','acceptor_T']]
    return donor_matrix_df,acceptor_matrix_df

def get_r_altseq(pos,r_location,r_seq,alt):
    location_start=int(r_location.split(':')[1].split('-')[0])+1
    location_end=int(r_location.split(':')[1].split('-')[1])
    if pos>=location_start and pos<=location_end:
        relative_pos=pos-location_start
        seq1=r_seq[:relative_pos]
        seq3=r_seq[relative_pos+1:]
        altseq=seq1+alt+seq3
    else:
        altseq=r_seq
    return altseq

def get_sw_freq(donor_freq_matrix,acceptor_freq_matrix,region_da):
    region_dict={
        'donor':donor_freq_matrix,
        'acceptor':acceptor_freq_matrix
    }
    return region_dict[region_da]

def r_iw(region_probability_matrix):
    epsilon = 1e-10  # 防止对数计算中的除零错误
    # information_content_matrix = 2 + np.log2(probability_matrix) - np.e
    d=1
    information_content_matrix = 2*d - (-np.log(np.maximum(region_probability_matrix, epsilon)) / np.log(2))
    return information_content_matrix

# 	Difference between Ri of ref and alt alleles of the closest acceptor site (0 if the variant does not affect the site).
def cal_r_i(seq,information_content_matrix):
    ACGT_dict={
            'A':0,
            'C':1,
            'G':2,
            'T':3,
    }
    information_content_matrix_seq=np.zeros_like(information_content_matrix)

    for i in range(len(seq)):
        j=ACGT_dict[seq[i]]
        information_content_matrix_seq[i][j]=information_content_matrix.iloc[i,j]

    # 计算每行的和
    r_i = np.sum(information_content_matrix_seq)
    return r_i


def get_Rican(r_seq,donor_freq_matrix,acceptor_freq_matrix,region_da):
    region_matrix = get_sw_freq(donor_freq_matrix, acceptor_freq_matrix, region_da)
    region_relative_pos_dict={
        'donor':[246,255],
        'acceptor':[226,253]
    }
    # relative pos
    region_relative_pos_key=region_da
    relative_freq_pos_start = region_relative_pos_dict[region_relative_pos_key][0]
    relative_freq_pos_end = region_relative_pos_dict[region_relative_pos_key][1]
    # region freq matrix
    aim_region_matrix = region_matrix[relative_freq_pos_start:relative_freq_pos_end]

    information_content_matrix = r_iw(region_probability_matrix=aim_region_matrix)
    r_i = cal_r_i(r_seq, information_content_matrix)
    return r_i



def get_r_slidingwindow_altseq(region_da,ref,alt,r_slidingwindow_refseq,strand):
    region_dict={
        'donor':9,
        'acceptor':27
    }
    length=region_dict[region_da]
    str1=r_slidingwindow_refseq[:length-1]
    str2=r_slidingwindow_refseq[length-1]
    str3=r_slidingwindow_refseq[length:]
    if strand=='-':
        alt=complement(alt)
    altseq=str1+alt+str3
    return altseq


def max_r_i_crypt_donor_window(pos,r_freq_location,r_slidingwindow_altseq,donor_freq_matrix,acceptor_freq_matrix,region_da,r_freq_seq):

    # ACGT region matrix
    region_matrix=get_sw_freq(donor_freq_matrix,acceptor_freq_matrix,region_da)
    # FLANKING_LEN = 499
    ACGT_dict={
            'A':0,
            'C':1,
            'G':2,
            'T':3,
    }
    region_sliding_window={
        'donor':9,
        'acceptor':27
    }

    sliding_window=region_sliding_window[region_da]
    seq_start=-1
    seq_end=sliding_window-1
    pos_start=pos-sliding_window
    pos_end=pos-1
    #--
    freq_pos_start=int(r_freq_location.split(':')[1].split('-')[0])+1
    relative_freq_pos_start=int(abs(pos_start-freq_pos_start))
    relative_freq_pos_end=int(abs(pos_end-freq_pos_start)+1)
    max_Ri_cryptic_donor_window=float('-inf')

    for i in range(sliding_window):
        seq_start+=1
        seq_end+=1
        pos_start+=1
        pos_end+=1
        relative_freq_pos_start+=1
        relative_freq_pos_end+=1
        #------------------------
        seq=r_slidingwindow_altseq[seq_start:seq_end]
        aim_region_matrix = region_matrix[relative_freq_pos_start:relative_freq_pos_end]
        aim_r_freq_seq = r_freq_seq[relative_freq_pos_start:relative_freq_pos_end]

        information_content_matrix=r_iw(region_probability_matrix=aim_region_matrix)
        r_i=cal_r_i(seq, information_content_matrix)

        max_Ri_cryptic_donor_window=max(r_i,max_Ri_cryptic_donor_window)
    return max_Ri_cryptic_donor_window

def get_r_features(strand,r_seq,pos,alt,region_da,r_location,donor_freq_matrix, acceptor_freq_matrix,r_sw_seq,r_freq_location,r_freq_seq,r_next_seq,ref):
    # 2. Rican ref:Information content (Ri) of the closest canonical donor site.
    r_refseq = get_strand_seq(strand, r_seq)
    r_altseq = get_r_altseq(pos, r_location, r_seq, alt)
    r_altseq = get_strand_seq(strand, r_altseq)

    ri_can_ref = get_Rican(r_refseq, donor_freq_matrix, acceptor_freq_matrix, region_da)
    ri_can_alt = get_Rican(r_altseq, donor_freq_matrix, acceptor_freq_matrix, region_da)
    difference_ri_can = ri_can_alt - ri_can_ref
    relative_ri_can = get_relative_score(refscore=ri_can_ref, altscore=ri_can_alt)

    # -------max Ri cryptic donor window
    r_sw_refseq = get_strand_seq(strand, r_sw_seq)
    r_sw_altseq = get_r_slidingwindow_altseq(region_da, ref,alt, r_sw_seq, strand)
    max_Ri_cryptic_donor_window_ref = max_r_i_crypt_donor_window(pos, r_freq_location, r_sw_refseq,
                                                                 donor_freq_matrix, acceptor_freq_matrix, region_da,
                                                                 r_freq_seq)
    max_Ri_cryptic_donor_window_alt = max_r_i_crypt_donor_window(pos, r_freq_location, r_sw_altseq,
                                                                 donor_freq_matrix, acceptor_freq_matrix, region_da,
                                                                 r_freq_seq)
    difference_ri_crypt = max_Ri_cryptic_donor_window_alt - max_Ri_cryptic_donor_window_ref
    relative_ri_crypt = get_relative_score(refscore=max_Ri_cryptic_donor_window_ref,
                                           altscore=max_Ri_cryptic_donor_window_alt)

    # -------r next:	Difference between Ri of the closest donor and the downstream (3′) donor site (0 if this is the donor site of the last intron).

    if r_next_seq !='NAN':
        r_next_refseq = get_strand_seq(strand, r_next_seq)
        r_next = get_Rican(r_next_refseq, donor_freq_matrix, acceptor_freq_matrix, region_da)
        difference_r_next = r_next - ri_can_ref
    else:
        difference_r_next = 0
    r_scores_list = [ri_can_ref, ri_can_alt, difference_ri_can, relative_ri_can,
                     max_Ri_cryptic_donor_window_ref, max_Ri_cryptic_donor_window_alt, difference_ri_crypt,relative_ri_crypt,
                     difference_r_next]
    return r_scores_list