#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/17 13:55
# @Author  : even
# @File    : predict.py
import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from models.models import SpliceModel
from models.trainers import SpliceTrainer

def print_prompt(regionname):
    print('\033[1;36m┌────────────────────────────────────────────┐\033[0m')
    print(f'\033[1;36m│        ► Region {regionname.upper()}: Finish │\033[0m')
    print('\033[1;36m└────────────────────────────────────────────┘\033[0m')


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'create dir\t{dir}')

class DLDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, index):
        return {k: self.inputs[k][index] for k in self.inputs}

    def __len__(self):
        return len(self.inputs['x'])
        # return len(self.inputs['labels'])

def make_test_dataloader(
        scaler_path,
        manu_feature,
        ref_sequence,
        alt_sequence,
        batch_size=128,
        max_k=5,
        device=None
):
    scaler = pickle.load(open(scaler_path, 'rb'))
    manu_feature = scaler.transform(manu_feature)
    inputs = {}
    for k in range(1, max_k + 1):
        inputs[f'ref_ids_{k}'] = torch.LongTensor(ref_sequence[str(k)]).to(device)
        inputs[f'alt_ids_{k}'] = torch.LongTensor(alt_sequence[str(k)]).to(device)

    scaler = StandardScaler()
    scaler.fit(manu_feature)
    manu_x = scaler.transform(manu_feature)

    inputs['x'] = torch.FloatTensor(manu_x).to(device)
    # inputs['labels'] = torch.LongTensor(labels).to(device)
    ds = DLDataset(inputs)
    dataloader = DataLoader(ds, shuffle=False, batch_size=batch_size)

    return dataloader

def load_region_features(df,regionname):
    region_dict={
    # 32
    'acceptor':[
        'dis',
        'exon_length','exon_length_mod3',
        'ri_can_ref','ri_can_alt',
        'diff_ri_can','relative_ri_can','max_Ri_cryptic_donor_window_ref',
        'max_Ri_cryptic_donor_window_alt','diff_ri_crypt','relative_ri_crypt',
        'ese_ref','ese_alt','ess_ref','ess_alt','diff_refesr','diff_altesr',
        'diff_esr','relative_esr',
        'mes_ref','mes_alt','diff_mes','relative_mes',
        'ref_pos_rbpscore','ref_neg_rbpscore','alt_pos_rbpscore','alt_neg_rbpscore',
        'ref_rbpscore','alt_rbpscore','diff_rbpscore','relative_rbpscore',
        'phylop_score'
        ],

        # 29
    'donor':[
        'dis','exon_length','exon_length_mod3',
        'ri_can_ref','ri_can_alt',
         'diff_ri_can','relative_ri_can','max_Ri_cryptic_donor_window_ref',
         'max_Ri_cryptic_donor_window_alt','diff_ri_crypt','relative_ri_crypt',
         'diff_r_next',
         'mes_ref','mes_alt','diff_mes','relative_mes',
         'ref_pos_rbpscore',
         'ref_neg_rbpscore','alt_pos_rbpscore','alt_neg_rbpscore','ref_rbpscore',
         'alt_rbpscore','diff_rbpscore','relative_rbpscore',
        'phylop_score'
         ],
    # 33
    'exonic':[
        'dis','exon_length','exon_length_mod3','ri_can_ref','ri_can_alt','diff_ri_can',
        'relative_ri_can','max_Ri_cryptic_donor_window_ref','max_Ri_cryptic_donor_window_alt',
        'diff_ri_crypt','relative_ri_crypt','diff_r_next',
        'ese_ref','ese_alt','ess_ref',
        'ess_alt','diff_refesr','diff_altesr','diff_esr','relative_esr',
        'mes_ref','mes_alt','diff_mes','relative_mes',
        'ref_pos_rbpscore','ref_neg_rbpscore',
        'alt_pos_rbpscore','alt_neg_rbpscore','ref_rbpscore','alt_rbpscore','diff_rbpscore',
        'relative_rbpscore',
        'phylop_score'
        ]
    }
    df=df[region_dict[regionname]]
    manu_dim=len(region_dict[regionname])
    return manu_dim,df

def predict(
        scaler_path,
        weight_file,
        manu_feature,
        ref_sequence,
        alt_sequence,
        manu_dim
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader=make_test_dataloader(
        scaler_path,
        manu_feature,
        ref_sequence,
        alt_sequence,
        batch_size=128,
        max_k=5,
        device=None
        )

    model = SpliceModel(manu_dim=manu_dim,max_k=5)

    model.load_state_dict(torch.load(weight_file,map_location=device))
    model.to(device=device)
    model.eval()
    trainer = SpliceTrainer(model)
    trainer.model = model

    pred_labels,pred_probs = trainer.inference(test_loader)
    return pred_labels,pred_probs

def region_predict(
        features_file='dataset/toy.features.vcf',
        seq_file='dataset/seq_ids.json',
        pred_file='dataset/pred.csv'
):
    regionnames=[
        'acceptor',
        'donor',
        'exonic',
    ]

    modelpath='dataset/modeldata'
    pred_cols=[
        'chrom','pos','ref','alt','region3','exon_start','exon_end','ts_start','ts_end','strand',
        'gene_id','transcript_id','refseq_ts_id','gene_name','transcript_name','exon_number',
        'exon_counts','protein_id','exon_id'
    ]
    features_df=pd.read_csv(features_file,sep='\t')
    pred_df=features_df[pred_cols]
    pred_df['pred_label']=None
    pred_df['pred_prob']=None
    with open(seq_file, 'r', encoding='utf-8') as f:
        seq_ids = json.load(f)
    for regionname in regionnames:
        region_features_df=features_df[features_df['region3']==regionname]
        manu_dim,manu_feature=load_region_features(region_features_df,regionname)
        manu_feature = manu_feature.fillna(manu_feature.mean())
        ref_sequence =seq_ids[regionname+'_ref_ids']
        alt_sequence =seq_ids[regionname+'_alt_ids']
        pred_labels,pred_prob=predict(
            scaler_path=os.path.join(modelpath,regionname, f'scaler.pkl'),
            weight_file=os.path.join(modelpath,regionname, f'weights.pkl'),
            manu_feature=manu_feature,
            ref_sequence=ref_sequence,
            alt_sequence=alt_sequence,
            manu_dim=manu_dim
        )

        mask = pred_df['region3'] == regionname
        if len(pred_labels) == mask.sum():
            pred_df.loc[mask, 'pred_label'] = pred_labels
            pred_df.loc[mask, 'pred_prob'] = pred_prob
        else:
            raise ValueError(f"pred_labels长度({len(pred_labels)})与目标行数({mask.sum()})不匹配！")

        print(pred_df['pred_label'])
        print(pred_df['pred_prob'])
        print_prompt(regionname)
    print(pred_df)
    pred_df.to_csv(pred_file,index=False,sep='\t')

def main():
    parser = argparse.ArgumentParser(description='This is a script for prediction.')
    parser.add_argument('-vcf', help='It must have input vcf file', default=r'dataset/toy.features.vcf')
    parser.add_argument('-seq', help='It must have input seq file', default=r'dataset/seq_ids.json')
    parser.add_argument('-out', help='Prediction output file.', default=r'dataset/toy.pred.csv')

    # Parse command-line parameters
    args = parser.parse_args()

    # Get the value of the -i parameter
    features_file = args.vcf
    seq_file = args.seq
    pred_file = args.out

    region_predict(
        features_file=features_file,
        seq_file=seq_file,
        pred_file=pred_file
    )
if __name__ == '__main__':
    main()