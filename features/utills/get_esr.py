#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 12:04
# @Author  : even
# @File    : get_esr.py
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
                # print(sequence)
                return None
            resequence=resequence+rebase
    resequence=resequence[::-1]
    return str(resequence)

def get_strand_seq(strand,seq):
    if strand=='-':
        seq=complement(seq)
    return seq

def get_relative_score(refscore,altscore):
    if refscore==0:
        relative_score=0
    else:
        relative_score=(altscore-refscore)/refscore
    return relative_score

def get_esr_altseq(esr_seq,alt):
    altseq = str(esr_seq[0:5]) + alt + str(esr_seq[6:])
    return altseq

# ESR
def get_esrscores(esr_seq,esr_dict):
    ess = 0
    ese = 0
    length = 6
    for i in range(length):
        seq6mer = esr_seq[i:length + i]
        keys_list = list(esr_dict.keys())
        # ------ref
        if seq6mer in keys_list:
            tmpscore = esr_dict[seq6mer]
            if tmpscore > 0:
                ese = ese + tmpscore
            else:
                ess = ess + tmpscore
    return [ess,ese]


def get_esr_features(strand,esr_seq,alt,esr_dict):
    esr_refseq = get_strand_seq(strand, esr_seq)
    esr_altseq = get_esr_altseq(esr_seq, alt)
    esr_altseq = get_strand_seq(strand, esr_altseq)
    ess_ref, ese_ref = get_esrscores(esr_refseq, esr_dict)
    ess_alt, ese_alt = get_esrscores(esr_altseq, esr_dict)

    differnce_refesr = ese_ref - ess_ref
    differnce_altesr = ese_alt - ess_alt
    differnce_esr = differnce_altesr - differnce_refesr
    relative_esr=get_relative_score(refscore=differnce_refesr,altscore=differnce_altesr)

    esrscores_list = [ese_ref, ese_alt, ess_ref, ess_alt, differnce_refesr, differnce_altesr, differnce_esr,relative_esr]

    return esrscores_list