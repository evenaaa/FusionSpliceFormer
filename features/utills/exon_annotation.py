#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/30 17:29
# @Author  : even
# @File    : exon_annotation.py
import pandas as pd
import numpy as np
import re
import pybedtools
import os
from utills.get_r import *
pattern = re.compile(r"^NM_\d+")

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'create dir\t{dir}')

# r seq
def get_r_start_end(strand,region_da,boundary,dis_refalt):
    r_region_dict = {
        'donor_+': [3, 6],
        'donor_-': [7, 2],
        'acceptor_+': [26, 1],
        'acceptor_-': [2, 25],
    }
    key = region_da + '_' + strand
    boundary_start = r_region_dict[key][0]
    boundary_end = r_region_dict[key][1]
    r_start = boundary - boundary_start
    r_end = boundary + boundary_end+dis_refalt
    return [r_start,r_end]

def get_donor_acceptor(strand,col_name):
    donor_acceptor_dict={
        'start_+':'acceptor',
        'end_+':'donor',
        'start_-':'donor',
        'end_-':'acceptor'
    }
    key=col_name+'_'+strand
    region=donor_acceptor_dict[key]
    return region

def get_dis(pos,boundary,col_name):
    #col_name:start/end
    dis=pos-boundary
    if col_name=='start':
        if dis>=0:
            dis=dis+1
    elif col_name=='end':
        if dis<=0:
            dis=dis-1
    return dis


def get_region3(region_da,pos,exon_start,exon_end):
    region3=region_da
    if pos>=exon_start and pos<=exon_end:
        region3='exonic'
    return region3

# gtf annotation
def gtf_annotation(pos,nm_id,gtf_df,ts_MANE):
    is_tsmatch=0
    gtf_annotation_list=[]

    # match nm_id:
    if isinstance(nm_id, str) and pattern.match(nm_id):
        gtf_ts_df=gtf_df[gtf_df['refseq_ts_id']==nm_id]
        # match
        if gtf_ts_df.empty:
            is_tsmatch=0
        else:
            is_tsmatch=1
    # None nm_id
    if is_tsmatch==0:
        # select MANE nm_id
        ts_id_list=ts_MANE[(ts_MANE['start'] <= pos) & (ts_MANE['end'] >= pos)]['transcript_id'].values
        if len(ts_id_list)==1:
            is_tsmatch=1
            ts_id=ts_id_list[0]
            gtf_ts_df=gtf_df[gtf_df['transcript_id']==ts_id]
    # start gtd annotation
    if is_tsmatch:
        start_diff = np.abs(pos-gtf_ts_df['start'])
        end_diff = np.abs(pos-gtf_ts_df['end'])

        # find boundary
        min_diffs = pd.concat([start_diff, end_diff], axis=1).min(axis=1)
        global_min_diff = min_diffs.min()
        min_index = min_diffs[min_diffs == global_min_diff].index.tolist()
        col_name = 'start' if start_diff[min_index[0]] == global_min_diff else 'end'
        specified_index=min_index[0]
        ts_start=gtf_ts_df[gtf_ts_df['type']=='transcript']['start'].values[0]
        ts_end=gtf_ts_df[gtf_ts_df['type']=='transcript']['end'].values[0]
        aim_exon=gtf_ts_df.loc[specified_index]
        boundary=aim_exon[col_name]

        # define variables
        exon_start,exon_end,strand,gene_id,transcript_id,refseq_ts_id,gene_name,level,isMANESelect,transcript_name,exon_number,\
            exon_counts,protein_id,exon_id=aim_exon['start'],aim_exon['end'],aim_exon['strand'],aim_exon['gene_id'],\
            aim_exon['transcript_id'],aim_exon['refseq_ts_id'],aim_exon['gene_name'],aim_exon['level'],aim_exon['isMANESelect'],\
            aim_exon['transcript_name'],aim_exon['exon_number'],aim_exon['exon_counts'],aim_exon['protein_id'],aim_exon['exon_id']
        region=get_donor_acceptor(strand,col_name)
        dis=get_dis(pos,boundary,col_name)
        region3=get_region3(region_da=region, pos=pos,exon_start=exon_start,exon_end=exon_end)
        next_boundary=None
        if exon_number<exon_counts and region=='donor':
            next_exon = gtf_ts_df.loc[specified_index + 1]
            next_boundary=next_exon[col_name]

        gtf_annotation_list=[boundary,dis,region,region3,exon_start,exon_end,next_boundary,ts_start,ts_end,strand,gene_id,transcript_id,refseq_ts_id,gene_name,level,isMANESelect,transcript_name,exon_number,\
            exon_counts,protein_id,exon_id]
    return is_tsmatch,gtf_annotation_list

# get sequence start position and sequence end position
def get_slidingwindow_start_end(pos,slidingwindow,dis_refalt):
    slidingwindow_start=pos-slidingwindow
    slidingwindow_end=pos+slidingwindow-1+dis_refalt

    return [slidingwindow_start,slidingwindow_end]

# get sequence start position and sequence end position from region donor and region acceptor
def get_region_slidingwindow_start_end(region,pos,dis_refalt):
    region_length_dict={
        'donor':9,
        'acceptor':27
    }
    slidingwindow=region_length_dict[region]
    slidingwindow_start, slidingwindow_end=get_slidingwindow_start_end(pos,slidingwindow,dis_refalt)
    return [slidingwindow_start,slidingwindow_end]

# maxentscan's start and end
def get_maxentscan_start_end(strand,region_da,boundary , dis_refalt):
    # donor:[exon 3bp][intron 6bp]
    FLANKING_LEN_exon5ss=3
    FLANKING_LEN_intron5ss=6

    # acceptor [intron 20bp][exon 3bp]
    FLANKING_LEN_intron3ss=20
    FLANKING_LEN_exon3ss=3

    mes_region_dict = {
        'donor_+': [FLANKING_LEN_exon5ss, FLANKING_LEN_intron5ss],
        'donor_-': [FLANKING_LEN_intron5ss+1, FLANKING_LEN_exon5ss-1],
        'acceptor_+': [FLANKING_LEN_intron3ss+1,FLANKING_LEN_exon3ss-1],
        'acceptor_-': [FLANKING_LEN_exon3ss,FLANKING_LEN_intron3ss-1],
    }
    key = region_da + '_' + strand
    boundary_start = mes_region_dict[key][0]
    boundary_end = mes_region_dict[key][1]
    start = boundary - boundary_start
    end = boundary + boundary_end + dis_refalt
    return [start,end]

# r sliding window for freq
def get_flanking_start_end(strand,region_da,boundary):
    FLANKING_LEN=200
    r_region_dict = {
        'donor_+': [boundary - FLANKING_LEN+1, boundary + FLANKING_LEN+1],
        'donor_-': [boundary - FLANKING_LEN-2, boundary + FLANKING_LEN-2],
        'acceptor_+': [boundary - FLANKING_LEN-2, boundary + FLANKING_LEN -2],
        'acceptor_-': [boundary - FLANKING_LEN+1, boundary + FLANKING_LEN+1],
    }
    key=region_da+'_'+strand
    start = r_region_dict[key][0]
    end = r_region_dict[key][1]
    return [start,end]

def list2csv(outputdata, csv_file_path):
    with open(csv_file_path, 'w',encoding='utf-8') as csvfile:
        for row in outputdata:
            row=[str(i) for i in row]
            line = '\t'.join(row)
            csvfile.write(line + '\n')
    print(f'Crete BedTools input file : {csv_file_path}')

def get_bed_region(variant_path,base400_fileinpath,r_fileinpath,r_slidingwindow_fileinpath,r_next_fileinpath,esr_fileinpath,spliceaid_fileinpath,\
                   maxentscan_fileinpath,
                   gtf_path='dataset/annodata/gtf/gencode.v47.refseq.ec.gtf',):
    # gtf
    gtf=pd.read_csv(gtf_path,sep='\t')
    # variant
    variant_df=pd.read_csv(variant_path,sep='\t',dtype={'chrom':str})
    # choose MANE
    ts_MANE = gtf[(gtf['type'] == 'transcript') & (gtf['isMANESelect'] == 'MANE_Select')]
    # chrom_list
    chrom_list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    # sort by chrom firstly
    variant_df['chrom'] = pd.Categorical(variant_df['chrom'], categories=chrom_list, ordered=True)
    # sort by chrom and pos
    variant_df = variant_df.sort_values(by=['chrom', 'pos'])
    # reset index
    variant_df = variant_df.reset_index(drop=True)

    # add new cols
    gtf_columns=[
    'boundary','dis','region_da','region3','exon_start', 'exon_end', 'next_next_boundary',
     'ts_start', 'ts_end', 'strand', 'gene_id', 'transcript_id','refseq_ts_id', 'gene_name',
     'level', 'isMANESelect', 'transcript_name', 'exon_number', 'exon_counts', 'protein_id', 'exon_id']
    for col in gtf_columns:
        variant_df[col]=None
    # output matrix
    base400_outputlist = []
    r_outputlist = []
    r_slidingwindow_outputlist = []
    r_next_outputlist=[]
    esr_outputlist = []
    spliceaid_outputlist = []
    maxentscan_outputlist = []

    # mapping in chr
    for chrom in chrom_list:
        gtf_chrom=gtf[gtf['chrom']=='chr'+chrom]
        variant_chrom=variant_df[variant_df['chrom']==chrom]
        ts_MANE_chrom=ts_MANE[ts_MANE['chrom']=='chr'+chrom]
        for idx, row in variant_chrom.iterrows():
            chrom = row['chrom']
            chrom_str='chr'+chrom
            pos = row['pos']
            ref=row['ref']
            alt=row['alt']
            ts_id=row['nm_id']
            is_tsmatch, gtf_annotation_list=gtf_annotation( pos, ts_id, gtf_chrom,ts_MANE_chrom)
            if is_tsmatch:
                boundary,dis,region_da,region3, exon_start, exon_end, next_boundary, ts_start, ts_end, strand, gene_id, transcript_id,\
                    refseq_ts_id, gene_name, level, isMANESelect, transcript_name, exon_number, exon_counts, protein_id, exon_id \
                =gtf_annotation_list
                if abs(dis)<=100:
                    # -------
                    dis_refalt=abs(len(ref)-len(alt))
                    # --------------
                    # -----1.r------
                    # --------------
                    # print(r_freq_outline)
                    # 1.1 r seq
                    r_start, r_end = get_r_start_end(strand, region_da, boundary,dis_refalt)
                    r_outline = [chrom_str, r_start, r_end]
                    r_outputlist.append(r_outline)

                    # 1.2 r sliding window seq
                    r_slidingwindow_start, r_slidingwindow_end = get_region_slidingwindow_start_end(region_da, pos,dis_refalt)
                    r_slidingwindow_outline = [chrom_str, str(int(r_slidingwindow_start)), str(int(r_slidingwindow_end))]
                    r_slidingwindow_outputlist.append(r_slidingwindow_outline)
                    # 1.3 next r
                    if next_boundary!=None:
                        r_next_start,r_next_end=get_r_start_end(strand,region_da,next_boundary,dis_refalt)
                        r_next_outline=[chrom_str,r_next_start,r_next_end]
                        if r_next_start!=r_next_end:
                            r_next_outputlist.append(r_next_outline)
                    # --------------
                    # -----2.esr----
                    # --------------
                    esr_start, esr_end = get_slidingwindow_start_end(pos, slidingwindow=6 , dis_refalt=dis_refalt)
                    # output
                    esr_outline = [chrom_str, str(int(esr_start)), str(int(esr_end))]
                    esr_outputlist.append(esr_outline)

                    # --------------
                    # --3.spliceAid-
                    # --------------
                    length=100
                    spliceaid_start = pos - length
                    spliceaid_end = pos + length+dis_refalt
                    spliceaid_outline = [chrom_str, str(int(spliceaid_start)), str(int(spliceaid_end))]
                    spliceaid_outputlist.append(spliceaid_outline)

                    # --------------
                    # --4.maxentscan-
                    # --------------
                    maxentscan_start, maxentscan_end = get_maxentscan_start_end(strand, region_da, boundary , dis_refalt)
                    maxentscan_outline = [chrom_str, maxentscan_start, maxentscan_end]
                    maxentscan_outputlist.append(maxentscan_outline)

                    # --------------
                    # --4.base400 seq-
                    # --------------

                    base400_start, base400_end = get_flanking_start_end(strand, region_da, boundary)
                    base400_outline = [chrom_str, base400_start, base400_end]
                    base400_outputlist.append(base400_outline)
                    # --------------
                    # ----output----
                    # --------------
                    outline=list(row)+gtf_annotation_list
                    variant_df.loc[idx, gtf_columns] = gtf_annotation_list
    list2csv(r_outputlist, r_fileinpath)
    list2csv(base400_outputlist, base400_fileinpath)
    list2csv(r_slidingwindow_outputlist, r_slidingwindow_fileinpath)
    list2csv(r_next_outputlist,r_next_fileinpath)
    list2csv(esr_outputlist, esr_fileinpath)
    list2csv(spliceaid_outputlist, spliceaid_fileinpath)
    list2csv(maxentscan_outputlist, maxentscan_fileinpath)
    variant_gtf_df = variant_df[variant_df['boundary'].notnull()]
    return variant_gtf_df

# get all sequence data for preparing feature annotaion
def save_seq(base400_fileinpath,r_fileinpath,r_slidingwindow_fileinpath,r_next_fileinpath,esr_fileinpath,spliceaid_fileinpath,maxentscan_fileinpath,
             base400_fileoutpath,r_fileoutpath,r_slidingwindow_fileoutpath,r_next_fileoutpath,esr_fileoutpath,spliceaid_fileoutpath,maxentscan_fileoutpath):
    get_fasta_sequence(bedin_file_path=base400_fileinpath, bedout_file_path=base400_fileoutpath)
    get_fasta_sequence(bedin_file_path=r_fileinpath, bedout_file_path=r_fileoutpath)
    get_fasta_sequence(bedin_file_path=r_slidingwindow_fileinpath, bedout_file_path=r_slidingwindow_fileoutpath)
    get_fasta_sequence(bedin_file_path=r_next_fileinpath,bedout_file_path=r_next_fileoutpath)
    get_fasta_sequence(bedin_file_path=esr_fileinpath, bedout_file_path=esr_fileoutpath)
    get_fasta_sequence(bedin_file_path=spliceaid_fileinpath, bedout_file_path=spliceaid_fileoutpath)
    get_fasta_sequence(bedin_file_path=maxentscan_fileinpath, bedout_file_path=maxentscan_fileoutpath)

def get_fasta_sequence(bedin_file_path,bedout_file_path,fasta_file = 'dataset/annodata/fa/GRCh38.p14.genome.fa'):
    # create BedTool from local data
    bed = pybedtools.BedTool(bedin_file_path)

    # get sequence
    sequences = bed.sequence(fi=fasta_file)

    # save sequence to path
    sequences.save_seqs(bedout_file_path)

    # delete bedin data
    try:
        os.remove(bedin_file_path)
        # print(f"Successfully deleted {bedin_file_path}")
    except FileNotFoundError:
        print(f"Error: {bedin_file_path} not found.")
    except PermissionError:
        print(f"Error: Permission denied when trying to delete {bedin_file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred while deleting {bedin_file_path}: {e}")

def get_bed_seq(bedout_df):
    columns_name=bedout_df.columns.tolist()[0]
    # print('columns_name:',columns_name)
    # print(bedout_df)

    # odd
    odd_indices = bedout_df.index[::2]

    # even
    even_indices = bedout_df.index[1::2]

    # Get the data of odd and even rows from the original DataFrame.
    odd_rows = bedout_df.loc[odd_indices]
    even_rows = bedout_df.loc[even_indices]

    odd_rows_name=columns_name+'_location'
    even_rows_name=columns_name+'_seq'
    # Convert the data in odd and even rows into a sequence and then add it as a column to a new DataFrame.
    new_bedout = pd.DataFrame({
        odd_rows_name: odd_rows.values.flatten(),
        even_rows_name: even_rows.values.flatten()
    })

    # Reset index
    new_bedout.reset_index(drop=True, inplace=True)

    print(f'finish get {columns_name} bed seq')
    return new_bedout

# merge vairant dataframe and 6 bedout dataframes
def get_r_next_fillnull(r_next_df,variant_df):
    # Extract the indices of the rows where the 'next_next_boundary' column is not None (represented by NaN in pandas)
    valid_index = variant_df[variant_df['next_next_boundary'].notna()].index

    # Make sure that the number of rows in r_next_df is consistent with the length of valid_index.
    if len(r_next_df) == len(valid_index):
        r_next_df.index = valid_index
        # print(r_next_df)
        return r_next_df
    else:
        print("The number of rows in r_next_df does not match the number of indices that meet the conditions, so the index cannot be modified.")

def merge_variant(variant_gtf_df,base400_fileoutpath,r_fileoutpath,r_slidingwindow_fileoutpath,r_next_fileoutpath,esr_fileoutpath,spliceaid_fileoutpath,maxentscan_fileoutpath):
    # data input
    base400_origindf=pd.read_csv(base400_fileoutpath,sep='\t',header=None,names=['base400'])
    r_origindf=pd.read_csv(r_fileoutpath,sep='\t',header=None,names=['r'])
    r_slidingwindow_origindf=pd.read_csv(r_slidingwindow_fileoutpath,sep='\t',header=None,names=['r_sw'])
    r_next_origindf=pd.read_csv(r_next_fileoutpath,sep='\t',header=None,names=['r_next'])
    esr_origindf=pd.read_csv(esr_fileoutpath,sep='\t',header=None,names=['esr'])
    spliceaid_origindf=pd.read_csv(spliceaid_fileoutpath,sep='\t',header=None,names=['spliceaid'])
    maxentscan_origindf=pd.read_csv(maxentscan_fileoutpath,sep='\t',header=None,names=['mes'])
    # sepertate bedoutdata:location & seq
    base400_df=get_bed_seq(base400_origindf)

    r_df=get_bed_seq(r_origindf)
    r_slidingwindow_df=get_bed_seq(r_slidingwindow_origindf)
    r_next_df=get_bed_seq(r_next_origindf)
    r_next_df=get_r_next_fillnull(r_next_df,variant_gtf_df)

    esr_df=get_bed_seq(esr_origindf)
    spliceaid_df=get_bed_seq(spliceaid_origindf)
    maxentscan_df=get_bed_seq(maxentscan_origindf)

    # concat DataFrame
    variant_combined = pd.concat([variant_gtf_df,base400_df,r_df, r_slidingwindow_df,r_next_df,esr_df,spliceaid_df,maxentscan_df], axis=1)

    delete_path_list=[base400_fileoutpath,r_fileoutpath,r_slidingwindow_fileoutpath,r_next_fileoutpath,esr_fileoutpath,spliceaid_fileoutpath,maxentscan_fileoutpath]
    for delete_path in delete_path_list:
        try:
            os.remove(delete_path)
        except FileNotFoundError:
            print(f"Error: {delete_path} not found.")
        except PermissionError:
            print(f"Error: Permission denied when trying to delete {delete_path}.")
        except Exception as e:
            print(f"An unexpected error occurred while deleting {delete_path}: {e}")

    return variant_combined


def exon_annotation(
        variant_path='dataset/toy.txt',
        beddata_path='dataset/annodata/beddata/'
):
    #-----------------------------
    # bedin
    check_dir(beddata_path)
    base400_fileinpath=beddata_path+'bedin.base400.txt'
    r_fileinpath=beddata_path+'bedin.r.txt'
    r_slidingwindow_fileinpath=beddata_path+'bedin.r_slidingwindow.txt'
    r_next_fileinpath=beddata_path+'bedin.r_next.txt'
    esr_fileinpath=beddata_path+'bedin.esr.txt'
    spliceaid_fileinpath=beddata_path+'bedin.spliceaid.txt'
    maxentscan_fileinpath=beddata_path+'bedin.maxentscan.txt'
    #------------------------
    # bedout
    base400_fileoutpath=beddata_path+'bedout.base400.txt'
    r_fileoutpath=beddata_path+'bedout.r.txt'
    r_slidingwindow_fileoutpath=beddata_path+'bedout.r_slidingwindow.txt'
    r_next_fileoutpath=beddata_path+'bedout.r_next.txt'
    esr_fileoutpath=beddata_path+'bedout.esr.txt'
    spliceaid_fileoutpath=beddata_path+'bedout.spliceaid.txt'
    maxentscan_fileoutpath=beddata_path+'bedout.maxentscan.txt'


    variant_gtf_df=get_bed_region(variant_path,base400_fileinpath,r_fileinpath,r_slidingwindow_fileinpath,r_next_fileinpath,esr_fileinpath,\
                   spliceaid_fileinpath,maxentscan_fileinpath)

    # get bedtools dataoutput
    save_seq(base400_fileinpath,r_fileinpath,r_slidingwindow_fileinpath,r_next_fileinpath,esr_fileinpath,spliceaid_fileinpath,maxentscan_fileinpath,
             base400_fileoutpath,r_fileoutpath,r_slidingwindow_fileoutpath,r_next_fileoutpath,esr_fileoutpath,spliceaid_fileoutpath,maxentscan_fileoutpath)

    merge_df=merge_variant(variant_gtf_df,base400_fileoutpath,r_fileoutpath,r_slidingwindow_fileoutpath,r_next_fileoutpath,esr_fileoutpath,spliceaid_fileoutpath,maxentscan_fileoutpath)

    return merge_df
