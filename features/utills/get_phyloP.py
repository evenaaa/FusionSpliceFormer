#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 22:27
# @Author  : even
# @File    : get_phyloP.py
def get_phylop_score(bw,chrom,pos):
    chrom='chr'+str(chrom)
    start=int(pos)-1
    end=int(pos)
    values = bw.values(chrom, start, end)[0]
    return values

def close_bw(bw):
    bw.close()

