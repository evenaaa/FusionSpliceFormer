#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/17 13:55
# @Author  : even
# @File    : predict.py
from predict.models.model import FusionSpliceFormer
from predict.utills.trainers import FusionTrainer
from predict.utills.encode import *
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediction(region='acceptor', model_config=None, model_save_path=None,
              dataloader=None, basic_df=None):
    model = FusionSpliceFormer(**model_config)
    ckpt_path = os.path.join(model_save_path, 'acceptor.pkl')
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    test_trainer = FusionTrainer(
        model=model,
        device=device,
    )
    result = test_trainer.inference(dataloader)
    basic_df['region'] = region
    final_df = pd.concat([basic_df.reset_index(drop=True),
                          result.reset_index(drop=True)], axis=1)
    return final_df

def region_prediction_pipeline(region='acceptor',df=None,model_path='dataset/models'):

    model_config = {
        'num_name': 'num_site',
        'num_feat_dim': 18,
    }
    merged_inputs = get_target_inputs(
        df=df,
        region=region,
        scaler_path=model_path
    )
    batch_size=128
    ds = FSFDataset(merged_inputs)
    dataloader = DataLoader(ds, shuffle=False, batch_size=batch_size)

    basic_head=['chrom','pos','ref','alt','transcript','strand','dis']
    predict_df=prediction(
        region=region,
        model_config=model_config,
        model_save_path=model_path,
        dataloader=dataloader,
        basic_df=df[basic_head],
    )
    return predict_df
def prediction_pipeline(acceptor_df,donor_df,outfile='dataset/toy.predict.csv'):

    acceptor_predict_df = region_prediction_pipeline(region='acceptor', df=acceptor_df)
    donor_predict_df = region_prediction_pipeline(region='donor', df=donor_df)

    combined_df = pd.concat([acceptor_predict_df, donor_predict_df], ignore_index=True)

    combined_df = combined_df.sort_values(
        by=['chrom', 'pos', 'ref', 'alt'],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)
    combined_df.to_csv(outfile, index=False)


if __name__ == '__main__':
    prediction_pipeline()