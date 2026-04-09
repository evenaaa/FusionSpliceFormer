#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 22:27
# @Author  : even
# @File    : get_phyloP.py
import numpy as np
from Bio.SeqUtils import gc_fraction
def get_phylop_score(bw, chrom, pos):
    chrom = 'chr' + str(chrom)
    start = int(pos) - 1   # 1-based -> 0-based
    end = int(pos)         # half-open
    values = bw.values(chrom, start, end)
    return float(values[0]) if len(values) > 0 else np.nan

def fetch_phastcons_window(
    bw,
    chrom: str,
    pos_1based: int,
    window: int
):
    chrom = 'chr' + str(chrom)
    pos0 = int(pos_1based) - 1
    start = max(0, pos0 - window)
    end = pos0 + window + 1   # half-open

    values = bw.values(
        chrom,
        start,
        end,
        missing=np.nan,
        oob=np.nan
    )
    values = values[~np.isnan(values)]
    return values

def get_phastCons_score(bw_phastCons, chrom, pos):
    pc_5 = np.mean(fetch_phastcons_window(bw_phastCons, chrom, pos, 5))
    pc_10 = np.mean(fetch_phastcons_window(bw_phastCons, chrom, pos, 10))
    pc_20 = np.mean(fetch_phastcons_window(bw_phastCons, chrom, pos, 20))
    pc_50 = np.mean(fetch_phastcons_window(bw_phastCons, chrom, pos, 50))
    return pc_5, pc_10, pc_20, pc_50

def compute_multi_gc(seq: str):
    seq = seq.strip().upper()
    center = len(seq) // 2

    gc_5   = gc_fraction(seq[center-5   : center+6])
    gc_10  = gc_fraction(seq[center-10  : center+11])
    gc_25  = gc_fraction(seq[center-25  : center+26])
    gc_50  = gc_fraction(seq[center-50  : center+51])
    gc_100 = gc_fraction(seq[center-100 : center+101])
    gc_200 = gc_fraction(seq[center-200 : center+201])

    return gc_5, gc_10, gc_25, gc_50, gc_100, gc_200

def close_bw(bw):
    bw.close()