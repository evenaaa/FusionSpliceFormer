"""Sequence encoding and feature fusion for the released inference graph."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F
from .layers import (DilatedResidualBlock, SITE_LENGTH, LearnedAnchorPool,
    LearnedSegmentCompression, _sinusoidal_positions, receptive_field_bp,
    DNA_VOCAB_SIZE, PairedDifferenceProjection)

REGIONS = ('acceptor', 'donor')
REGION_TO_INDEX = {'acceptor': 0, 'donor': 1}
REGION_TO_SELECTOR_CLASS = {'acceptor': 1, 'donor': 2}
REGION_TO_TARGET_INDEX = {'acceptor': 5000, 'donor': 4999}

@dataclass
class PredictionOutput:
    logits: torch.Tensor
    representations: dict[str, torch.Tensor] | None = None

class SequenceEncoder(nn.Module):
    """Base-resolution CNN, Selector, context tokens and shared Transformer."""
    def __init__(self, *, site_length: int=SITE_LENGTH, d_model: int=64, n_head: int=4, cnn_channels: int=32, cnn_dilations: Iterable[int]=(1, 2, 4, 8, 16, 32, 64), stem_kernel_size: int=7, global_token_count: int=400, site_anchor_windows: Iterable[int]=(51, 101), variant_anchor_windows: Iterable[int]=(51, 101), candidates_per_type_per_allele: int=16, transformer_layers: int=2, feedforward_multiplier: int=4, encoder_dropout: float=0.0) -> None:
        super().__init__()
        self.site_length = int(site_length)
        self.d_model = int(d_model)
        self.cnn_channels = int(cnn_channels)
        self.cnn_dilations = tuple((int(value) for value in cnn_dilations))
        self.stem_kernel_size = int(stem_kernel_size)
        self.global_token_count = int(global_token_count)
        self.site_anchor_windows = tuple((int(value) for value in site_anchor_windows))
        self.variant_anchor_windows = tuple((int(value) for value in variant_anchor_windows))
        self.candidates_per_type_per_allele = int(candidates_per_type_per_allele)
        self.transformer_layers = int(transformer_layers)
        if self.site_length != SITE_LENGTH:
            raise ValueError('selector model requires exactly 10,000 nt')
        if self.d_model <= 0 or self.d_model % int(n_head) or self.d_model % 2:
            raise ValueError('d_model must be even and divisible by n_head')
        if self.global_token_count <= 0 or self.site_length % self.global_token_count:
            raise ValueError('global_token_count must exactly divide 10,000')
        if not 0 < self.candidates_per_type_per_allele < self.site_length:
            raise ValueError('invalid candidates_per_type_per_allele')
        if not 0.0 <= float(encoder_dropout) < 1.0:
            raise ValueError('encoder_dropout must be in [0, 1)')
        for windows in (self.site_anchor_windows, self.variant_anchor_windows):
            if not windows or any((int(value) <= 0 or int(value) % 2 == 0 for value in windows)):
                raise ValueError('anchor windows must be non-empty positive odd integers')
        self.receptive_field_bp = receptive_field_bp(self.cnn_dilations, self.stem_kernel_size)
        self.base_embedding = nn.Embedding(DNA_VOCAB_SIZE, self.cnn_channels)
        self.stem = nn.Conv1d(self.cnn_channels, self.cnn_channels, kernel_size=self.stem_kernel_size, padding=self.stem_kernel_size // 2)
        self.residual_blocks = nn.ModuleList((DilatedResidualBlock(self.cnn_channels, dilation) for dilation in self.cnn_dilations))
        self.high_resolution_norm = nn.GroupNorm(math.gcd(self.cnn_channels, 8), self.cnn_channels)
        self.splice_selector = nn.Sequential(nn.Conv1d(self.cnn_channels, 16, kernel_size=1), nn.GELU(), nn.Conv1d(16, 3, kernel_size=1))
        self.compression_width = self.site_length // self.global_token_count
        self.learned_compression = LearnedSegmentCompression(input_channels=self.cnn_channels, output_channels=self.d_model, sequence_length=self.site_length, token_count=self.global_token_count)
        self.anchor_projection = nn.Linear(self.cnn_channels, self.d_model)
        self.site_pools = nn.ModuleList((LearnedAnchorPool(self.cnn_channels, width) for width in self.site_anchor_windows))
        self.variant_pools = nn.ModuleList((LearnedAnchorPool(self.cnn_channels, width) for width in self.variant_anchor_windows))
        self.anchor_count = 2 + len(self.site_pools) + len(self.variant_pools)
        self.candidate_projection = nn.Linear(self.cnn_channels, self.d_model)
        self.candidate_score_projection = nn.Linear(3, self.d_model, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cls_type = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.anchor_types = nn.Parameter(torch.zeros(1, self.anchor_count, self.d_model))
        self.global_type = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.candidate_type_embedding = nn.Embedding(2, self.d_model)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(n_head), dim_feedforward=self.d_model * int(feedforward_multiplier), dropout=float(encoder_dropout), activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.transformer_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(self.d_model)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_type, std=0.02)
        nn.init.normal_(self.anchor_types, std=0.02)
        nn.init.normal_(self.global_type, std=0.02)
        nn.init.normal_(self.candidate_type_embedding.weight, std=0.02)
        global_centres = torch.arange(self.global_token_count, dtype=torch.float32) * self.compression_width + (self.compression_width - 1) / 2.0
        self.register_buffer('global_token_centres', global_centres)

    def high_resolution_features(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.size(1) != self.site_length:
            raise ValueError(f'expected sequence shape [B, {self.site_length}], got {tuple(sequence.shape)}')
        if torch.any(sequence < 0) or torch.any(sequence >= DNA_VOCAB_SIZE):
            raise ValueError('DNA sequence contains an out-of-vocabulary code')
        features = self.base_embedding(sequence.long()).transpose(1, 2)
        features = self.stem(features)
        for block in self.residual_blocks:
            features = block(features)
        return F.gelu(self.high_resolution_norm(features))

    def _candidate_union(self, reference_logits: torch.Tensor, alternate_logits: torch.Tensor, target_selector_classes: torch.Tensor, target_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference_logits.shape != alternate_logits.shape:
            raise ValueError('REF/ALT selector logits differ in shape')
        (batch_size, classes, length) = reference_logits.shape
        if classes != 3 or length != self.site_length:
            raise ValueError('selector logits must have shape [B, 3, 10000]')
        keys_per_sample: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            keys = []
            for selector_class in (1, 2):
                sources = (reference_logits, alternate_logits)
                k = self.candidates_per_type_per_allele
                for values in sources:
                    positions = torch.topk(values[batch_index, selector_class], k=k, sorted=False).indices
                    keys.append(selector_class * self.site_length + positions)
            target_key = (target_selector_classes[batch_index] * self.site_length + target_positions[batch_index]).reshape(1)
            unique_keys = torch.unique(torch.cat([*keys, target_key]), sorted=True)
            keys_per_sample.append(unique_keys)
        maximum = max((int(values.numel()) for values in keys_per_sample))
        positions = torch.zeros((batch_size, maximum), dtype=torch.long, device=reference_logits.device)
        types = torch.ones_like(positions)
        padding = torch.ones((batch_size, maximum), dtype=torch.bool, device=reference_logits.device)
        for (batch_index, keys) in enumerate(keys_per_sample):
            count = int(keys.numel())
            positions[batch_index, :count] = keys.remainder(self.site_length)
            types[batch_index, :count] = torch.div(keys, self.site_length, rounding_mode='floor')
            padding[batch_index, :count] = False
        return (positions, types, padding)

    def _anchor_features(self, features: torch.Tensor, variant_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        site_positions = torch.full_like(variant_positions, self.site_length // 2)
        rows = features.transpose(1, 2)
        batch_indices = torch.arange(batch_size, device=features.device)
        values = [rows[batch_indices, site_positions]]
        values.extend((pool(features, site_positions) for pool in self.site_pools))
        values.append(rows[batch_indices, variant_positions])
        values.extend((pool(features, variant_positions) for pool in self.variant_pools))
        positions = torch.cat([site_positions[:, None].expand(-1, 1 + len(self.site_pools)), variant_positions[:, None].expand(-1, 1 + len(self.variant_pools))], dim=1)
        return (self.anchor_projection(torch.stack(values, dim=1)), positions)

class FeatureFusion(nn.Module):
    def __init__(self, *, acceptor_numeric_dim=38, donor_numeric_dim=27,
                 d_model=64, n_head=4, site_length=10000, cnn_channels=32,
                 cnn_dilations=(1,2,4,8,16,32,64), stem_kernel_size=7,
                 global_token_count=400, site_anchor_windows=(51,101),
                 variant_anchor_windows=(51,101), candidates_per_type_per_allele=16,
                 transformer_layers=2, dropout=0.2, encoder_dropout=0.0):
        super().__init__()
        self.d_model=d_model
        self.numeric_dims={'acceptor':acceptor_numeric_dim,'donor':donor_numeric_dim}
        self.sequence_encoder=SequenceEncoder(site_length=site_length,d_model=d_model,
            n_head=n_head,cnn_channels=cnn_channels,cnn_dilations=cnn_dilations,
            stem_kernel_size=stem_kernel_size,global_token_count=global_token_count,
            site_anchor_windows=site_anchor_windows,variant_anchor_windows=variant_anchor_windows,
            candidates_per_type_per_allele=candidates_per_type_per_allele,
            transformer_layers=transformer_layers,encoder_dropout=encoder_dropout)
        self.sequence_difference=PairedDifferenceProjection(d_model,dropout)
        self.numeric_encoders=nn.ModuleDict({region:nn.Sequential(
            nn.Linear(dimension,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(d_model,d_model),nn.LayerNorm(d_model),nn.GELU())
            for region,dimension in self.numeric_dims.items()})
        self.delta_encoder=nn.Sequential(nn.Linear(9,32),nn.LayerNorm(32),nn.GELU())
    @staticmethod
    def _normalise_region(region: str) -> str:
        value = str(region).strip().lower()
        if value not in REGIONS:
            raise ValueError(f'region must be one of {REGIONS}, got {region!r}')
        return value

    @staticmethod
    def _variant_positions(reference: torch.Tensor, alternate: torch.Tensor) -> torch.Tensor:
        if reference.shape != alternate.shape or reference.ndim != 2:
            raise ValueError('site REF/ALT must have matching [B, L] shapes')
        mismatch = reference.ne(alternate)
        counts = mismatch.sum(dim=1)
        if not torch.all(counts == 1):
            observed = torch.unique(counts).detach().cpu().tolist()
            raise ValueError(f'SNV prediction requires exactly one REF/ALT difference; observed {observed}')
        return mismatch.to(dtype=torch.int64).argmax(dim=1)

    def _delta_features(self, reference_logits: torch.Tensor, alternate_logits: torch.Tensor, target_selector_class: int, target_positions: torch.Tensor) -> torch.Tensor:
        reference = torch.softmax(reference_logits, dim=1)
        alternate = torch.softmax(alternate_logits, dim=1)
        difference = alternate - reference
        batch_indices = torch.arange(reference.size(0), device=reference.device)
        target_ref = reference[batch_indices, target_selector_class, target_positions]
        target_alt = alternate[batch_indices, target_selector_class, target_positions]
        target_delta = target_alt - target_ref
        (acceptor_gain, acceptor_gain_pos) = difference[:, 1].max(dim=1)
        (acceptor_loss, acceptor_loss_pos) = (-difference[:, 1]).max(dim=1)
        (donor_gain, donor_gain_pos) = difference[:, 2].max(dim=1)
        (donor_loss, donor_loss_pos) = (-difference[:, 2]).max(dim=1)
        if target_selector_class == 1:
            (gain_position, loss_position) = (acceptor_gain_pos, acceptor_loss_pos)
        else:
            (gain_position, loss_position) = (donor_gain_pos, donor_loss_pos)
        scale = float(self.sequence_encoder.site_length // 2)
        gain_distance = (gain_position.float() - target_positions.float()) / scale
        loss_distance = (loss_position.float() - target_positions.float()) / scale
        return torch.stack([target_ref, target_alt, target_delta, acceptor_gain, acceptor_loss, donor_gain, donor_loss, gain_distance, loss_distance], dim=1)
