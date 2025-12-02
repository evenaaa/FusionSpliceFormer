#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 10:56
# @Author  : even
# @File    : get_mes_score3.py
import math

def hashseq(seq):
    seq = seq.upper()
    seq = ''.join('0' if nucleotide == 'A' else
                  '1' if nucleotide == 'C' else
                  '2' if nucleotide == 'G' else
                  '3' for nucleotide in seq)
    base4_powers = (1, 4, 16, 64, 256, 1024, 4096, 16384) # 4^0 4^1 4^2 4^3 4^5 4^6 4^7
    return sum(int(seq[i]) * base4_powers[len(seq) - i - 1] for i in range(len(seq)))

def maxentscore(seq, metables):
    sc = []
    sc.append(metables[0].get(hashseq(seq[:7]), 0))
    sc.append(metables[1].get(hashseq(seq[7:14]), 0))
    sc.append(metables[2].get(hashseq(seq[14:21]), 0))
    sc.append(metables[3].get(hashseq(seq[4:11]), 0))
    sc.append(metables[4].get(hashseq(seq[11:18]), 0))
    sc.append(metables[5].get(hashseq(seq[4:7]), 0))
    sc.append(metables[6].get(hashseq(seq[7:11]), 0))
    sc.append(metables[7].get(hashseq(seq[11:14]), 0))
    sc.append(metables[8].get(hashseq(seq[14:18]), 0))
    sc=[float(i) for i in sc]
    finalscore = sc[0] * sc[1] * sc[2] * sc[3] * sc[4] / (sc[5] * sc[6] * sc[7] * sc[8])
    return finalscore

def getrest(seq):
    return seq[:18] + seq[20:23]

def scoreconsensus(seq):
    seq = seq.upper()
    bgd = {'A': 0.27, 'C': 0.23, 'G': 0.23, 'T': 0.27}
    cons1 = {'A': 0.9903, 'C': 0.0032, 'G': 0.0034, 'T': 0.0030}
    cons2 = {'A': 0.0027, 'C': 0.0037, 'G': 0.9905, 'T': 0.0030}
    addscore = cons1.get(seq[18], 0) * cons2.get(seq[19], 0) / (bgd.get(seq[18], 0) * bgd.get(seq[19], 0))
    return addscore

def log2(val):
    return math.log(val) / math.log(2)

def get_score3(mes_seq,mes3table):
    mes_seq.upper()
    str_line = mes_seq.upper()
    scores=log2(scoreconsensus(str_line) * maxentscore(getrest(str_line), mes3table))
    return scores