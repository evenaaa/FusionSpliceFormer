#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/23 14:55
# @Author  : even
# @File    : load_data.py
import pandas as pd
import os
import pyBigWig
def list2csv(outputdata, csv_file_path):
    # 将列表数据写入CSV文件
    with open(csv_file_path, 'w',encoding='utf-8') as csvfile:
        for row in outputdata:
            # 将每行数据转换为制表符分隔的字符串并写入文件
            row=[str(i) for i in row]
            line = '\t'.join(row)
            csvfile.write(line + '\n')
    print(f'-------------------------------------------------------------')
    print(f'The data has been printed to file {csv_file_path}')
    print(f'-------------------------------------------------------------')


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
            elif base == 'N':
                rebase='N'
            else:
                # print(sequence)
                return None
            resequence=resequence+rebase
    resequence=resequence[::-1]
    return str(resequence)

# load exon data
def load_exon_data():
    # prepath=f'D:\\EvenData\\spliceAPT\\dataset\\'
    prepath=f'dataset/annodata/gtf/'
    path=prepath+'gencode.v47.refseq.ec.gtf'
    df=pd.read_csv(path,sep='\t')
    # print(df)
    return df

# load variant data
def load_variant_data():
    # prepath=f'D:\\EvenData\\spliceAPT\\dataset\\'
    prepath=f'dataset/annodata/rawdata/'
    path=prepath+'merge.unique.snv.txt'
    df=pd.read_csv(path,sep='\t',dtype={'chrom':str})
    # print(df)
    return df

def load_merge_tmp_data():
    prepath=f'dataset/annodata/'
    path=prepath+'merge.tmp.snv.txt'
    # path=prepath+'merge.tmp3.txt'
    df=pd.read_csv(path,sep='\t')
    return df

def load_bedout_data_rfreq():
    prepath=f'dataset/annodata/beddata/'
    r_freq_fileoutpath=prepath+'bedout.r_sw_freq.txt'
    r_freq_df=pd.read_csv(r_freq_fileoutpath,sep='\t',header=None,names=['r_freq'])
    return r_freq_df

def load_bedout_data_r():
    prepath=f'dataset/annodata/beddata/'
    r_fileoutpath=prepath+'bedout.r.txt'
    r_df=pd.read_csv(r_fileoutpath,sep='\t',header=None,names=['r'])
    return r_df

def load_bedout_data_sw():
    prepath=f'dataset/annodata/beddata/'
    r_slidingwindow_fileoutpath=prepath+'bedout.r_slidingwindow.txt'
    r_slidingwindow_df=pd.read_csv(r_slidingwindow_fileoutpath,sep='\t',header=None,names=['r_sw'])
    return r_slidingwindow_df

def load_bedout_data_r_next():
    prepath=f'dataset/annodata/beddata/'
    r_next_fileoutpath=prepath+'bedout.r_next.txt'
    r_next_df=pd.read_csv(r_next_fileoutpath,sep='\t',header=None,names=['r_next'])
    return r_next_df

def load_bedout_data_esr():
    prepath=f'dataset/annodata/beddata/'
    esr_fileoutpath=prepath+'bedout.esr.txt'
    esr_df=pd.read_csv(esr_fileoutpath,sep='\t',header=None,names=['esr'])
    return esr_df

def load_bedout_data_mes():
    prepath=f'dataset/annodata/beddata/'
    maxentscan_fileoutpath=prepath+'bedout.maxentscan.txt'
    maxentscan_df=pd.read_csv(maxentscan_fileoutpath,sep='\t',header=None,names=['mes'])
    return maxentscan_df

def load_data_esr_dict():
    prepath=f'dataset/annodata/'
    esr_fileoutpath=prepath+'ESRSeq_score.txt'
    esr_df=pd.read_csv(esr_fileoutpath,sep='\t')
    esr_dict = {}
    for index, row in esr_df.iterrows():
        seq = row['sequence']
        score = row['score']
        esr_dict[seq] = score
    return esr_dict

def load_data_mes5dict():
    prepath=f'dataset/annodata/maxentscan/'
    mes_fileoutpath=prepath+'maxentscan.txt'
    mes=pd.read_csv(mes_fileoutpath,sep='\t')
    mes_dict = {}
    for idx, row in mes.iterrows():
        seq = row['seq']
        score = row['score']
        mes_dict[seq] = score
    return mes_dict

def load_data_mes3dict():
# def makemaxentscores(dir_path):
    dir_path="dataset/annodata/maxentscan/splicemodels/"
    file_list = ['me2x3acc1', 'me2x3acc2', 'me2x3acc3', 'me2x3acc4',
                 'me2x3acc5', 'me2x3acc6', 'me2x3acc7', 'me2x3acc8', 'me2x3acc9']
    metables = []
    for file_name in file_list:
        file_path = os.path.join(dir_path, file_name)
        try:
            with open(file_path, 'r') as score_file:
                metable = {}
                n = 0
                for line in score_file:
                    line = line.strip().replace(' ', '')
                    metable[n] = line
                    n += 1
                metables.append(metable)
        except IOError:
            print(f"Can't open {file_name}!")
            return None
    return metables

def load_data_phylop():
    prepath=f'dataset/annodata/PhyloP/'
    phylop_fileoutpath=prepath+'hg38.phyloP100way.bw'
    bw = pyBigWig.open(phylop_fileoutpath, "r")
    return bw

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