from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F

DNA_VOCAB_SIZE = 5


class PairedDifferenceProjection(nn.Module):
    """Preserve REF, ALT, signed change and magnitude of change."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(int(d_model) * 4, int(d_model)),
            nn.LayerNorm(int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self, reference: torch.Tensor, alternate: torch.Tensor
    ) -> torch.Tensor:
        difference = alternate - reference
        return self.projection(
            torch.cat(
                [reference, alternate, difference, torch.abs(difference)], dim=-1
            )
        )

SITE_LENGTH = 10_000


def receptive_field_bp(dilations: Iterable[int], stem_kernel_size: int = 7) -> int:
    """Return the exact receptive field of the two-convolution residual stack."""

    values = tuple(int(value) for value in dilations)
    if not values or any(value <= 0 for value in values):
        raise ValueError("dilations must be a non-empty sequence of positive integers")
    if int(stem_kernel_size) <= 0 or int(stem_kernel_size) % 2 == 0:
        raise ValueError("stem_kernel_size must be a positive odd integer")
    # The stem contributes k-1 additional positions.  Each residual block has
    # two kernel-3 convolutions, each adding 2*dilation positions.
    return 1 + (int(stem_kernel_size) - 1) + 4 * sum(values)


class DilatedResidualBlock(nn.Module):
    """Length-preserving deterministic residual block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive")
        group_count = math.gcd(channels, 8)
        self.norm1 = nn.GroupNorm(group_count, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(group_count, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.conv1(F.gelu(self.norm1(features)))
        features = self.conv2(F.gelu(self.norm2(features)))
        return residual + features


class LearnedAnchorPool(nn.Module):
    """Attention-pool a centred, possibly boundary-truncated feature window."""

    def __init__(self, channels: int, window_size: int) -> None:
        super().__init__()
        channels = int(channels)
        window_size = int(window_size)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("anchor window_size must be a positive odd integer")
        self.channels = channels
        self.window_size = window_size
        self.query = nn.Parameter(torch.empty(channels))
        nn.init.normal_(self.query, std=channels ** -0.5)

    def forward(
        self, features: torch.Tensor, centre_positions: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, C, L]")
        batch_size, channels, sequence_length = features.shape
        if channels != self.channels:
            raise ValueError(
                f"expected {self.channels} feature channels, got {channels}"
            )
        positions = centre_positions.long().reshape(-1)
        if positions.numel() != batch_size:
            raise ValueError("one anchor centre is required per sample")
        if torch.any(positions < 0) or torch.any(positions >= sequence_length):
            raise ValueError("anchor centre lies outside the sequence")

        radius = self.window_size // 2
        offsets = torch.arange(
            -radius, radius + 1, device=features.device, dtype=torch.long
        )
        raw_indices = positions[:, None] + offsets[None, :]
        valid = (raw_indices >= 0) & (raw_indices < sequence_length)
        indices = raw_indices.clamp(0, sequence_length - 1)
        batch_indices = torch.arange(batch_size, device=features.device)[:, None]
        windows = features.transpose(1, 2)[batch_indices, indices]
        scores = torch.einsum("bwc,c->bw", windows, self.query)
        scores = scores / math.sqrt(self.channels)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(windows * weights.unsqueeze(-1), dim=1)


class LearnedSegmentCompression(nn.Module):
    """Compress equal-width segments with shared learned attention weights.

    The same query and projection are used for every segment.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        sequence_length: int,
        token_count: int,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.sequence_length = int(sequence_length)
        self.token_count = int(token_count)
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("compression channels must be positive")
        if self.token_count <= 0 or self.sequence_length % self.token_count:
            raise ValueError("token_count must exactly divide sequence_length")
        self.segment_width = self.sequence_length // self.token_count
        self.query = nn.Parameter(torch.empty(self.input_channels))
        self.projection = nn.Linear(self.input_channels, self.output_channels)
        nn.init.normal_(self.query, std=self.input_channels ** -0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, C, L]")
        batch_size, channels, sequence_length = features.shape
        if channels != self.input_channels or sequence_length != self.sequence_length:
            raise ValueError(
                "compression input does not match its declared channels/length"
            )
        segments = (
            features.reshape(
                batch_size,
                channels,
                self.token_count,
                self.segment_width,
            )
            .permute(0, 2, 3, 1)
        )
        scores = torch.einsum("btwc,c->btw", segments, self.query)
        scores = scores / math.sqrt(self.input_channels)
        weights = torch.softmax(scores, dim=2)
        pooled = torch.sum(segments * weights.unsqueeze(-1), dim=2)
        return self.projection(pooled)


def _sinusoidal_positions(
    positions: torch.Tensor,
    d_model: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create deterministic absolute position encodings for arbitrary positions."""

    if int(d_model) <= 0 or int(d_model) % 2:
        raise ValueError("d_model must be a positive even integer")
    positions = positions.to(dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    angles = positions.unsqueeze(-1) * frequencies
    encoding = torch.empty(
        (*positions.shape, d_model), device=positions.device, dtype=torch.float32
    )
    encoding[..., 0::2] = torch.sin(angles)
    encoding[..., 1::2] = torch.cos(angles)
    return encoding.to(dtype=dtype)

