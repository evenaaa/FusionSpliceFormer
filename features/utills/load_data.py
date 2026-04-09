#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/23 14:55
# @Author  : even
# @File    : load_data.py
import pandas as pd
import pybigtools
from pyfaidx import Fasta
import re
pattern = re.compile(r"^NM_\d+")

# load exon data
def load_exon_data():
    prepath=f'dataset/annodata/gtf/'
    path=prepath+'gencode.v47.refseq.ec.gtf'
    df=pd.read_csv(path,sep='\t')
    return df

def load_data_phylop():
    prepath=f'dataset/annodata/conservation/'
    phylop_fileoutpath=prepath+'hg38.phyloP100way.bw'
    bw = pybigtools.open(phylop_fileoutpath, "r")
    return bw

def load_data_phastCons():
    prepath=f'dataset/annodata/conservation/'
    phastCons_fileoutpath=prepath+'hg38.phastCons100way.bw'
    bw = pybigtools.open(phastCons_fileoutpath, "r")
    return bw

def get_region_freq_matrix(

):
    acceptor_freq_matrix = 'dataset/annodata/freq/acceptor_freq.txt'
    donor_freq_matrix = 'dataset/annodata/freq/donor_freq.txt'
    donor_matrix_df=pd.read_csv(donor_freq_matrix,sep='\t')
    acceptor_matrix_df=pd.read_csv(acceptor_freq_matrix,sep='\t')
    return donor_matrix_df,acceptor_matrix_df

# gtf
def parse_gtf_info_to_columns(info_str):
    gene_id = ""
    transcript_id = ""
    isMANE = ""
    if pd.isna(info_str) or info_str.strip() == "":
        return (gene_id, transcript_id, isMANE)

    gene_id_match = re.search(r'gene_id\s+"([^"]+)"', info_str)
    if gene_id_match:
        gene_id = gene_id_match.group(1)

    transcript_id_match = re.search(r'transcript_id\s+"([^"]+)"', info_str)
    if transcript_id_match:
        transcript_id = transcript_id_match.group(1)

    isMANE_match = re.search(r'isMANE\s+(\d+)', info_str)
    if isMANE_match:
        isMANE = isMANE_match.group(1)

    return (gene_id, transcript_id, isMANE)

def extract_gtf_info_fields():
    gtf_path='dataset/annodata/gtf/gencode.v47.anno.gtf'
    gtf = pd.read_csv(gtf_path, sep='\t')
    required_cols = ['info']
    for col in required_cols:
        if col not in gtf.columns:
            raise ValueError(f"GTF error")

    parsed_info = gtf['info'].apply(parse_gtf_info_to_columns)

    gtf['gene_id'] = [item[0] for item in parsed_info]
    gtf['transcript_id'] = [item[1] for item in parsed_info]
    gtf['isMANE'] = [item[2] for item in parsed_info]

    return gtf

def get_fasta():
    fa_path='dataset/annodata/fa/GRCh38.p14.genome.fa'
    genome = Fasta(fa_path)
    return genome