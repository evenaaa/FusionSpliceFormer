#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/15 14:17
# @Author  : even
# @File    : models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

class TransformerEncoder_Module(nn.Module):
    def __init__(self, input_channels, seq_len=400, embed_dim=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.encoder(x).transpose(1, 2)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_ref = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_alt = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ref = nn.LayerNorm(embed_dim)
        self.norm_alt = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ref, alt):
        ref_attn, _ = self.attn_ref(ref, alt, alt)
        ref = self.norm_ref(ref + self.dropout(ref_attn))
        alt_attn, _ = self.attn_alt(alt, ref, ref)
        alt = self.norm_alt(alt + self.dropout(alt_attn))
        return ref, alt

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([1 - alpha, alpha])
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets = targets.long()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        logpt = torch.log(pt + 1e-9)
        at = self.alpha.to(logits.device).gather(0, targets)
        loss = -at * ((1 - pt) ** self.gamma) * logpt
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class SpliceModel(nn.Module):
    def __init__(self, max_k=5, embedding_dim=64, manu_dim=24, dropout_rate=0.2,
                 seq_len=400, output_dim=256):
        super().__init__()
        self.max_k = max_k
        self.output_dim = output_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=(5 ** k) + 2, embedding_dim=embedding_dim)
            for k in range(1, max_k + 1)
        ])

        self.encoders = nn.ModuleList([
            TransformerEncoder_Module(embedding_dim, seq_len, output_dim)
            for _ in range(max_k)
        ])

        self.cross_attn = CrossAttention(embed_dim=output_dim)

        self.manu_proj = nn.Linear(manu_dim, output_dim)
        manu_encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=4, batch_first=True)
        self.manu_encoder = nn.TransformerEncoder(manu_encoder_layer, num_layers=2)

        self.dropout = nn.Dropout(dropout_rate)
        # self.fc = nn.Linear(output_dim * (2 * max_k), 2)
        self.fc = nn.Linear(output_dim * (2 * max_k + 1), 2)

    def forward(self, **inputs):
        features = []

        for i in range(self.max_k):
            ref = self.embeddings[i](inputs[f'ref_ids_{i + 1}']).permute(0, 2, 1)
            alt = self.embeddings[i](inputs[f'alt_ids_{i + 1}']).permute(0, 2, 1)

            ref_feat = self.encoders[i](ref).permute(0, 2, 1)
            alt_feat = self.encoders[i](alt).permute(0, 2, 1)

            # Cross Attention
            ref_feat, alt_feat = self.cross_attn(ref_feat, alt_feat)

            ref_pool, _ = torch.max(ref_feat, dim=1)
            alt_pool, _ = torch.max(alt_feat, dim=1)
            features.extend([ref_pool, alt_pool])

        x = inputs['x']
        x_seq = self.manu_proj(x).unsqueeze(1)
        manu_encoded = self.manu_encoder(x_seq)
        manu_feat = manu_encoded.squeeze(1)
        features.append(manu_feat)

        output = torch.cat(features, dim=1)
        output = self.dropout(output)
        logits = self.fc(output)

        loss = None
        labels = inputs.get("labels", None)
        if labels is not None:
            loss_fct = BinaryFocalLoss(gamma=2.0, alpha=0.75)
            loss = loss_fct(logits, labels)

        return ModelOutput(loss=loss, logits=logits)
