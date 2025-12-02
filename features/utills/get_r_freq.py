#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/10 12:31
# @Author  : even
# @File    : get_r_freq.py
import pandas as pd
def list2csv(outputdata, csv_file_path):
    # Write the list data to a CSV file
    with open(csv_file_path, 'w',encoding='utf-8') as csvfile:
        for row in outputdata:
            # Convert each row of data into a tab-separated string and write it to a file
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
                return None
            resequence=resequence+rebase
    resequence=resequence[::-1]
    return str(resequence)

def load_gtf():
    df=pd.read_csv('../../dataset/bed/gencode.v47lift37.basic.annotation.gtf',sep='\t')
    return df

def get_ts_id_dict(df):
    gene_ts_dict={
        # key : gene_id
        # value : transcript_id_list
    }
    gene_df=df[df['type']=='gene']
    ts_df=df[df['type']=='transcript']
    gene_id_list=gene_df['gene_id'].tolist()
    for gene_id in gene_id_list:
        ts_gene_df=ts_df[ts_df['gene_id']==gene_id]
        ts_id_list=ts_gene_df['transcript_id'].tolist()
        gene_ts_dict[gene_id]=ts_id_list
    return gene_ts_dict

def get_ts_region_list(exon_list,region_list):
    for exon in exon_list:
        if exon not in region_list:
            region_list.append(exon)
    return region_list

def get_region_freq_bedin(region_list):
    output=[]
    for line in region_list:
        chrom=line.split('_')[0]
        pos=int(line.split('_')[1])
        bed_start=pos-100
        bed_end=pos+100
        outline=[chrom,bed_start,bed_end]
        output.append(outline)
    return output

def get_seq_region(df,gene_ts_dict):
    donor_list,acceptor_list=[],[]
    exon_df=df[df['type']=='exon']
    for gene_id,ts_id_list in gene_ts_dict.items():
        gene_donor_list, gene_acceptor_list = [], []
        strand=df[(df['type'] == 'gene') & (df['gene_id'] == gene_id)]['strand'].values[0]
        for ts_id in ts_id_list:
            ts_exon_df = exon_df[exon_df['transcript_id'] == ts_id]
            exon_start_list = ts_exon_df['start'].tolist()
            exon_end_list = ts_exon_df['end'].tolist()
            if strand=='+':
                gene_acceptor_list=get_ts_region_list(exon_start_list,gene_acceptor_list)
                gene_donor_list=get_ts_region_list(exon_end_list,gene_donor_list)
            elif strand=='-':
                gene_acceptor_list=get_ts_region_list(exon_end_list,gene_acceptor_list)
                gene_donor_list=get_ts_region_list(exon_start_list,gene_donor_list)
        donor_list.extend(gene_donor_list)
        acceptor_list.extend(gene_acceptor_list)
    donor_output=get_region_freq_bedin(donor_list)
    list2csv(donor_output,'../../dataset/bed/freq/donor.freq.test.bedin')
    acceptor_output=get_region_freq_bedin(acceptor_list)
    list2csv(acceptor_output,'../../dataset/bed/freq/acceptor.freq.test.bedin')

def get_start_end_row(strand,exon_start,exon_end,exon_list):
    region_da_strand_dict={
        'start_+':'acceptor',
        'end_+':'donor',
        'start_-':'donor',
        'end_-':'acceptor'
    }
    start_key='start_'+strand
    end_key='end_'+strand
    start_region_da=region_da_strand_dict[start_key]
    end_region_da=region_da_strand_dict[end_key]

    # add site,region_da,region_se
    start_list=exon_list+[exon_start,start_region_da,'start']
    end_list=exon_list+[exon_end,end_region_da,'end']
    return start_list,end_list

def get_region_site(df,gene_ts_dict):
    # donor_list,acceptor_list=[],[]
    # Create an empty DataFrame with the same column names as df2
    out_data=[df.columns.tolist()+['gene_start','gene_end','transcript_start','transcript_end','site','region_da','region_se']]

    for gene_id,ts_id_list in gene_ts_dict.items():
        gene_df=df[df['gene_id']==gene_id]
        gene_list=gene_df[gene_df['type']=='gene'].values.tolist()[0]

        # add gene_start,gene_end,ts_start,ts_end,site,region_da,region_se
        gene_start=gene_df[gene_df['type']=='gene']['start'].values[0]
        gene_end=gene_df[gene_df['type']=='gene']['end'].values[0]
        gene_list+=[gene_start,gene_end,None,None,None,None,None]

        # out gene data
        for ts_id in ts_id_list:
            ts_df=gene_df[gene_df['transcript_id']==ts_id]
            ts_list=ts_df[ts_df['type']=='transcript'].values.tolist()[0]
            # add gene_start,gene_end,ts_start,ts_end,site,region_da,region_se
            ts_start=ts_df[ts_df['type']=='transcript']['start'].values[0]
            ts_end=ts_df[ts_df['type']=='transcript']['end'].values[0]
            ts_list+=[gene_start,gene_end,ts_start,ts_end,None,None,None]
            # out ts data
            # out_data.append(ts_list)
            exon_df=ts_df[ts_df['type']=='exon']
            for idx,row in exon_df.iterrows():
                strand=row['strand']
                exon_start=row['start']
                exon_end=row['end']
                exon_list=row.values.tolist()+[gene_start,gene_end,ts_start,ts_end]

                start_list, end_list=get_start_end_row(strand,exon_start,exon_end,exon_list)
                out_data.append(start_list)
                out_data.append(end_list)
    list2csv(out_data,'../../dataset/bed/gencode.v47.da.gtf')

def get_unique_exon_gtf():
    df=pd.read_csv('../../dataset/bed/gencode.v47.da.gtf',sep='\t')
    df=df[df['type']=='exon']
    df_unique = df.drop_duplicates(subset='chrom_site_region_da_key', keep='first')
    df_unique.to_csv('../../dataset/bed/gencode.v47.da.exon.gtf',sep='\t')


def get_region_bedin():
    df=pd.read_csv('../../dataset/bed/gencode.v47.da.exon.gtf',sep='\t')
    region_da_list=['donor','acceptor']
    for region in region_da_list:
        region_df=df[df['region_da']==region]
        out_data=[]
        for idx,row in region_df.iterrows():
            chrom=row['chrom'].split('chr')[1]
            pos=row['site']
            strand=row['strand']
            region_da=row['region_da']
            if region_da=='donor':
                if strand=='+':
                    bed_start = pos - 100+1
                    bed_end = pos + 100+1
                elif strand=='-':
                    bed_start = pos - 100+4
                    bed_end = pos + 100+4
            elif region_da=='acceptor':
                if strand=='+':
                    bed_start = pos - 100-2
                    bed_end = pos + 100-2
                elif strand=='-':
                    bed_start = pos - 100-5
                    bed_end = pos + 100-5
            outline = [chrom, bed_start, bed_end]
            out_data.append(outline)
        list2csv(out_data,f'../../dataset/bed/freq/{region}.freq.bedin')

def get_bed_seq(bedout_df):
    columns_name=bedout_df.columns.tolist()[0]
    # Odd-numbered rows index
    odd_indices = bedout_df.index[bedout_df.index % 2 != 0]

    # Even-numbered row index
    even_indices = bedout_df.index[bedout_df.index % 2 == 0]

    # Extract the data of odd and even rows from the original DataFrame
    odd_rows = bedout_df.loc[odd_indices]
    even_rows = bedout_df.loc[even_indices]

    odd_rows_name=columns_name+'_seq'
    even_rows_name=columns_name+'_location'
    # Convert the data of odd rows and even rows into a sequence, and then add it as a column to the new DataFrame.
    new_bedout = pd.DataFrame({
        odd_rows_name: odd_rows.values.flatten(),
        even_rows_name: even_rows.values.flatten()
    })

    # Reset the index
    new_bedout.reset_index(drop=True, inplace=True)

    return new_bedout





