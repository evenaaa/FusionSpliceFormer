#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 16:18
# @Author  : even
# @File    : trainers.py
import torch
from tqdm import tqdm

class SpliceTrainer:
    def __init__(self, model,device=None):
        self.model = model
        self.model.to(device)
        self.device = device

    def inference(self, dataloader):
        self.model.eval()
        pred_labels = []
        pred_probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move the data and labels to the designated device.
                batch = {k: batch[k].to(self.device) for k in batch}
                output = self.model(**batch)
                _pred_labels = torch.argmax(output.logits, dim=-1)
                _pred_probs = torch.softmax(output.logits, dim=-1)
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.cpu().numpy()[:,1].tolist()
        return pred_labels,pred_probs
