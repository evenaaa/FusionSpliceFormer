#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/9 19:45
# @Author  : even
# @File    : encode.py
from predict.utills.dataset import *
from predict.utills.load_data import *
import os
import torch
import pickle
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r_matrix(donor_matrix_df, acceptor_matrix_df):
    acceptor_r = r_iw(region_probability_matrix=acceptor_matrix_df)
    donor_r = r_iw(region_probability_matrix=donor_matrix_df)

    entropy_dict = {
        'acceptor': acceptor_r,
        'donor': donor_r,
    }
    return entropy_dict

def r_iw(region_probability_matrix=[0.248457,0.231477,0.244537,0.275530]):
    epsilon = 1e-10
    d=1
    information_content_matrix = 2*d - (-np.log(np.maximum(region_probability_matrix, epsilon)) / np.log(2))

    return information_content_matrix
def reverse_complement(seq):
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]

_DNA_INT_MAP = np.full(256, 4, dtype=np.int8)
_DNA_INT_MAP[ord('A')] = 0
_DNA_INT_MAP[ord('a')] = 0
_DNA_INT_MAP[ord('C')] = 1
_DNA_INT_MAP[ord('c')] = 1
_DNA_INT_MAP[ord('G')] = 2
_DNA_INT_MAP[ord('g')] = 2
_DNA_INT_MAP[ord('T')] = 3
_DNA_INT_MAP[ord('t')] = 3

def seq_to_int_array(seq):
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    return _DNA_INT_MAP[seq_bytes]

def get_refseq_list(df):
    refseq_list = []
    for idx, row in df.iterrows():
        strand = row['strand']
        longseq = row['refseq']
        refseq = longseq
        if strand == '-':
            refseq = reverse_complement(refseq)
        ref_one_hot = seq_to_int_array(refseq)
        refseq_list.append(ref_one_hot)
    return refseq_list

def get_altseq_list(df, region):
    altseq_list = []
    for idx, row in df.iterrows():
        ref, alt = row['ref'], row['alt']
        strand = row['strand']
        longseq = row['refseq']
        dis, pos = row['dis'], row['pos']
        refseq = longseq
        altseq = get_altseq(refseq, dis, strand, alt, ref, pos, region)
        if strand == '-':
            altseq = reverse_complement(altseq)
        alt_one_hot = seq_to_int_array(altseq)
        altseq_list.append(alt_one_hot)
    return altseq_list

def get_altseq(refseq, dis, strand, alt, ref, pos, region):
    relative_pos = 0

    if region == 'acceptor':
        if strand == '+':
            if dis > 0:
                dis -= 1
            relative_pos = 5000 + int(dis) + 1
        elif strand == '-':
            if dis < 0:
                dis += 1
            relative_pos = 5000 + int(dis) - 2

    elif region == 'donor':
        if strand == '+':
            if dis < 0:
                dis += 1
            relative_pos = 5000 + int(dis) - 2
        elif strand == '-':
            if dis > 0:
                dis -= 1
            relative_pos = 5000 + int(dis) + 1

    if relative_pos < 0 or relative_pos >= len(refseq):
        return refseq
    altseq = refseq[:relative_pos] + alt + refseq[relative_pos + 1:]
    return altseq

def construct_entropy_icseq(region, strand, dis, tartget_seq, entropy_dict):
    acceptor_r = entropy_dict['acceptor']
    donor_r = entropy_dict['donor']
    dis = int(dis)

    if region == 'acceptor':
        region_r = acceptor_r
    elif region == 'donor':
        region_r = donor_r
    else:
        raise ValueError(f"Unknown region: {region}")

    r_dict = {
        'A': region_r['A'],
        'C': region_r['C'],
        'G': region_r['G'],
        'T': region_r['T'],
        'N': region_r['N']
    }

    temp_list = []
    position = 0

    for base in tartget_seq:
        base_upper = base.upper()

        try:
            if base_upper not in r_dict:
                val = 0.0
            else:
                freq_df = r_dict[base_upper]
                if position in freq_df.index:
                    val = freq_df.loc[position]
                else:
                    val = 0.0
        except Exception as e:
            print(f"获取碱基{base_upper}位置{position}的值异常：{e}，使用背景值")
            val = 0.0
        temp_list.append(val)
        position += 1

    return temp_list

def get_entropy_icseq(df, region, entropy_dict, seq='ref'):
    maskseq_list = []
    for idx, row in df.iterrows():
        ref, alt = row['ref'], row['alt']
        pos = row['pos']
        dis = row['dis']
        strand = row['strand']
        longseq = row['refseq']
        refseq = longseq.upper()

        if seq == 'ref':
            tartget_seq = refseq
        elif seq == 'alt':
            tartget_seq = get_altseq(refseq, dis, strand, alt, ref, pos, region)
        else:
            raise ValueError(f"Unknown seq type: {seq}")
        if strand == '-':
            tartget_seq = reverse_complement(tartget_seq)

        entropy_maskseq = construct_entropy_icseq(region, strand, dis, tartget_seq, entropy_dict)
        maskseq_list.append(entropy_maskseq)

    return maskseq_list

def get_target_inputs(df, region, scaler_path):
    acceptor_freq_matrix_df,donor_freq_matrix_df = load_freq_matrix()

    entropy_dict = r_matrix(donor_freq_matrix_df, acceptor_freq_matrix_df)

    inputs = {}

    refseq = get_refseq_list(df)
    refseq_np = np.stack(refseq, axis=0)
    inputs['refseq'] = torch.LongTensor(refseq_np)

    altseq = get_altseq_list(df, region)
    altseq_np = np.stack(altseq, axis=0)
    inputs['altseq'] = torch.LongTensor(altseq_np)

    ic_refseq = get_entropy_icseq(df, region, entropy_dict, seq='ref')
    ic_refseq_np = np.stack(ic_refseq, axis=0).astype(np.float32)
    inputs['ic_refseq'] = torch.FloatTensor(ic_refseq_np)

    ic_altseq = get_entropy_icseq(df, region, entropy_dict, seq='alt')
    ic_altseq_np = np.stack(ic_altseq, axis=0).astype(np.float32)
    inputs['ic_altseq'] = torch.FloatTensor(ic_altseq_np)

    if scaler_path is not None:
        if not os.path.exists(scaler_path):
            os.makedirs(scaler_path)

        num_cols = load_num_head()
        num_df = df[num_cols]

        scaler_file = os.path.join(scaler_path, f'{region}.scaler.pkl')
        scaler = pickle.load(open(scaler_file, 'rb'))

        num_x = scaler.transform(num_df)
        inputs['num_site'] = torch.FloatTensor(num_x)

    return inputs