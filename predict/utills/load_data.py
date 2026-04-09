#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/9 19:37
# @Author  : even
# @File    : load_data.py
import pandas as pd
def load_num_head():
    head = [
        'dis', 'exon_rank', 'exon_last', 'exon_length', 'exon_length_mod3',
        'intron_length', 'intron_length_logp',
        'gc_5', 'gc_10', 'gc_25', 'gc_50', 'gc_100', 'gc_200',
        'phylop_score', 'phastCons_5', 'phastCons_10',
        'phastCons_20', 'phastCons_50'
    ]
    return head

def load_freq_matrix():
    acceptor_freq_matrix_path='dataset/annodata/freq/acceptor_freq.txt'
    donor_freq_matrix_path='dataset/annodata/freq/donor_freq.txt'
    acceptor_freq_matrix_df = pd.read_csv(acceptor_freq_matrix_path, sep='\t')
    donor_freq_matrix_df = pd.read_csv(donor_freq_matrix_path, sep='\t')

    return acceptor_freq_matrix_df,donor_freq_matrix_df