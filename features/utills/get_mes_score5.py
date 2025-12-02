#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/12 15:23
# @Author  : even
# @File    : get_mes_score5.py
import math
def log2(val):
    return math.log(val) / math.log(2)

def getrest(seq):
    seq.upper()
    seq=seq[:3]+seq[5:9]
    return seq

def scoreconsensus(seq):
    # Convert the sequence to uppercase
    seq = seq.upper()
    # Background frequency
    bgd = {'A': 0.27, 'C': 0.23, 'G': 0.23, 'T': 0.27}
    # Conservative Mode 1
    cons1 = {'A': 0.004, 'C': 0.0032, 'G': 0.9896, 'T': 0.0032}
    # Conservative Mode 2
    cons2 = {'A': 0.0034, 'C': 0.0039, 'G': 0.0042, 'T': 0.9884}

    # Split the sequence into a list of individual characters
    seq_list = list(seq)
    # Calculate the score
    if len(seq_list) < 5:
        raise ValueError("Sequence is too short.")
    addscore = cons1[seq_list[3]] * cons2[seq_list[4]] / (bgd[seq_list[3]] * bgd[seq_list[4]])
    return addscore

def get_score5(mes_seq,mes_dict):
    mes_seq_sep=getrest(mes_seq)
    me2x5_score = mes_dict[mes_seq_sep]
    score = log2(scoreconsensus(mes_seq) * me2x5_score)
    return score
