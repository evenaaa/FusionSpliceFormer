#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/12 14:18
# @Author  : even
# @File    : model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MaskAwarePositionalEncoding(nn.Module):
    """
    语义位置编码 [升级版：支持 Ref 和 Alt 双 Mask]：
    利用拆分后的 4 通道 Mask (Ref正/负, Alt正/负) 作为位置偏置。
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1, mask_channels: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # === [修改] 输入维度变成 4 (Ref_Pos, Ref_Neg, Alt_Pos, Alt_Neg) ===
        self.mask_bias_mlp = nn.Sequential(
            nn.Linear(mask_channels, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x_seq, mask_features):
        abs_pe = self.pe[:, :x_seq.size(1), :]
        semantic_bias = self.mask_bias_mlp(mask_features)
        out = x_seq + abs_pe + (semantic_bias * self.scale)
        return self.dropout(out)

class GatedFusion(nn.Module):
    def __init__(self, dim_seq, dim_alt, dim_num, out_dim):
        super().__init__()
        total_dim = dim_seq + dim_alt + dim_num

        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.gate_fc = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, v_seq, v_alt, v_num):
        concat_feat = torch.cat([v_seq, v_alt, v_num], dim=-1)
        h = self.fusion_fc(concat_feat)
        g = self.gate_fc(concat_feat)
        return h * g

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation='gelu', bn_momentum=0.01):
        super().__init__()
        assert in_channels == out_channels, "In/Out channels must match for ResidualBlock"

        self.in_channels = in_channels
        self.out_channels = out_channels

        paddingAmount = int(dilation * (kernel_size - 1) / 2)

        if activation == 'gelu':
            self.activate = nn.GELU()
        else:
            self.activate = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(in_channels, momentum=bn_momentum)
        self.convlayer1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, dilation=dilation,
            stride=1, padding=paddingAmount, padding_mode='zeros'
        )

        self.bn2 = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        self.convlayer2 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, dilation=dilation,
            stride=1, padding=paddingAmount, padding_mode='zeros'
        )

        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.bn1(x)
        x = self.activate(x)
        x = self.convlayer1(x)

        x = self.bn2(x)
        x = self.activate(x)
        x = self.convlayer2(x)

        x += residual
        return x

class MultiScaleConvStem(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, activation='gelu'):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_dim = out_channels // 4

        def make_branch(kernel_size, dilation):
            proj_padding = (kernel_size - 1) // 2

            return nn.Sequential(
                nn.Conv1d(in_channels, branch_dim, kernel_size=kernel_size, padding=proj_padding),
                ResidualBlock(
                    in_channels=branch_dim,
                    out_channels=branch_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation
                )
            )

        self.branch1 = make_branch(kernel_size=1, dilation=1)
        self.branch2 = make_branch(kernel_size=11, dilation=1)
        self.branch3 = make_branch(kernel_size=21, dilation=5)
        self.branch4 = make_branch(kernel_size=41, dilation=25)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.dropout(out)
        return out.transpose(1, 2)

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU()
        ) if in_dim != hidden_dims[0] else nn.Identity()

        layers = []
        curr_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            if h_dim != curr_dim:
                layers.append(nn.Linear(curr_dim, h_dim))
                curr_dim = h_dim
            layers.append(ResBlock(curr_dim, dropout))

        self.res_layers = nn.Sequential(*layers)
        self.out_proj = nn.Linear(curr_dim, out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_layers(x)
        return self.out_proj(x)

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor):
        attn_scores = self.attention(x)
        weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights

class CenterPriorGate(nn.Module):
    def __init__(self, seq_len: int, initial_focus_radius: float = 200.0):
        super().__init__()
        positions = torch.arange(seq_len, dtype=torch.float32)
        center = seq_len / 2.0
        gaussian_prior = torch.exp(-((positions - center) ** 2) / (2 * initial_focus_radius ** 2))
        self.spatial_weight = nn.Parameter(gaussian_prior.unsqueeze(0).unsqueeze(-1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.min_retention = 0.05

    def forward(self, x):
        gate_mask = torch.sigmoid(self.spatial_weight * self.gamma)
        gate_mask = torch.clamp(gate_mask, min=self.min_retention)
        return x * gate_mask

class ModelOutput:
    def __init__(self, loss=None, logits=None, attention_weights=None):
        self.loss = loss
        self.logits = logits
        self.attention_weights = attention_weights

class FusionSpliceFormer(nn.Module):
    def __init__(self,
                 num_name: str = 'quantitative_features',
                 ref_mask_name: str = 'ic_refseq',
                 alt_mask_name: str = 'ic_altseq',
                 num_feat_dim: int = 18,
                 seq_vocab_dim: int = 4,
                 mask_dim: int = 1,
                 seq_len: int = 10000,
                 d_model: int = 64,
                 n_head: int = 4,
                 num_transformer_layers: int = 3,
                 mlp_hidden_dims: list = [128, 128],
                 fusion_out_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 aux_loss_weight: float = 0.5,
                 flank: int = 50
                 ):
        super().__init__()
        self.num_name = num_name
        self.ref_mask_name = ref_mask_name
        self.alt_mask_name = alt_mask_name
        self.aux_loss_weight = aux_loss_weight
        self.flank = flank
        self.seq_vocab_dim = seq_vocab_dim

        self.actual_seq_len = self.flank * 2 if self.flank is not None else seq_len

        self.loss_fct = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.mask_channels = mask_dim * 4
        self.seq_input_dim = (seq_vocab_dim * 2) + self.mask_channels

        self.conv_stem = MultiScaleConvStem(in_channels=self.seq_input_dim, out_channels=d_model, dropout=dropout)
        self.center_gate = CenterPriorGate(seq_len=self.actual_seq_len)
        self.pos_encoding = MaskAwarePositionalEncoding(
            d_model, max_len=self.actual_seq_len, dropout=dropout, mask_channels=self.mask_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.seq_pooling = AttentionPooling(d_model)
        self.seq_classifier = nn.Linear(d_model, num_classes)

        self.num_mlp = ResidualMLP(in_dim=num_feat_dim, hidden_dims=mlp_hidden_dims, out_dim=d_model, dropout=dropout)
        self.fusion_gate = GatedFusion(dim_seq=d_model, dim_alt=0, dim_num=d_model, out_dim=fusion_out_dim)
        self.classifier = nn.Linear(fusion_out_dim, num_classes)

    def _encode_sequence(self, seq_input, mask_features):
        x = self.conv_stem(seq_input)
        x = self.center_gate(x)
        x = self.pos_encoding(x, mask_features)
        x = self.transformer(x)
        v, attn = self.seq_pooling(x)
        return v, attn

    def forward(self, **inputs):
        ref_idx = inputs['refseq'].long()
        alt_idx = inputs['altseq'].long()
        ref_mask_raw = inputs.get(self.ref_mask_name, None)
        alt_mask_raw = inputs.get(self.alt_mask_name, None)

        if self.flank is not None:
            center = ref_idx.size(1) // 2
            start, end = center - self.flank, center + self.flank
            ref_idx = ref_idx[:, start:end]
            alt_idx = alt_idx[:, start:end]
            if ref_mask_raw is not None: ref_mask_raw = ref_mask_raw[:, start:end]
            if alt_mask_raw is not None: alt_mask_raw = alt_mask_raw[:, start:end]

        refseq = F.one_hot(ref_idx, num_classes=5)[..., :self.seq_vocab_dim].float()
        altseq = F.one_hot(alt_idx, num_classes=5)[..., :self.seq_vocab_dim].float()

        ref_mask = torch.tanh(ref_mask_raw).unsqueeze(-1) if ref_mask_raw.dim() == 2 else torch.tanh(ref_mask_raw)
        ref_pos, ref_neg = F.relu(ref_mask), F.relu(-ref_mask)

        alt_mask = torch.tanh(alt_mask_raw).unsqueeze(-1) if alt_mask_raw.dim() == 2 else torch.tanh(alt_mask_raw)
        alt_pos, alt_neg = F.relu(alt_mask), F.relu(-alt_mask)

        mask_features = torch.cat([ref_pos, ref_neg, alt_pos, alt_neg], dim=-1).float()

        combined_input = torch.cat([refseq, altseq, mask_features], dim=-1).float()
        v_seq, attn_weights = self._encode_sequence(combined_input, mask_features)

        num_feat = inputs[self.num_name]
        num_feat = torch.nan_to_num(num_feat, nan=0.0)
        v_num = self.num_mlp(num_feat.float())

        if self.training:
            dropout_mask = (torch.rand(v_num.size(0), 1, device=v_num.device) > 0.3).float()
            v_num = v_num * dropout_mask

        dummy_alt = torch.empty(v_seq.size(0), 0, device=v_seq.device)
        fusion_feat = self.fusion_gate(v_seq, dummy_alt, v_num)

        logits = self.classifier(fusion_feat)
        seq_logits = self.seq_classifier(v_seq)

        loss = None
        labels = inputs.get("labels", None)
        if labels is not None:
            target = labels[:, 1].float() if labels.dim() > 1 else labels.float().view(-1)
            loss_main = self.loss_fct(torch.sigmoid(logits[:, 1]), target)
            loss_seq = self.loss_fct(torch.sigmoid(seq_logits[:, 1]), target)
            loss = loss_main + self.aux_loss_weight * loss_seq

        return ModelOutput(loss=loss, logits=logits, attention_weights=attn_weights)