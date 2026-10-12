"""FusionSpliceFormer prediction architecture."""
import torch
from torch import nn
from ._backbone import (FeatureFusion, SequenceEncoder, PredictionOutput,
    REGION_TO_TARGET_INDEX, REGION_TO_SELECTOR_CLASS, REGION_TO_INDEX, _sinusoidal_positions)

MODEL_CONFIG = {'architecture_version': 'fusionspliceformer_inference_v1', 'acceptor_numeric_dim': 38, 'donor_numeric_dim': 27, 'd_model': 64, 'n_head': 4, 'site_length': 10000, 'cnn_channels': 32, 'cnn_dilations': [1, 2, 4, 8, 16, 32, 64], 'stem_kernel_size': 7, 'global_token_count': 400, 'site_anchor_windows': [51, 101], 'variant_anchor_windows': [51, 101], 'candidates_per_type_per_allele': 16, 'transformer_layers': 2, 'dropout': 0.2, 'encoder_dropout': 0.0}

class TokenEncoder(SequenceEncoder):

    def __init__(self, initialized):
        nn.Module.__init__(self)
        self.__dict__.update(initialized.__dict__.copy())
        self._modules = initialized._modules.copy()
        self._parameters = initialized._parameters.copy()
        self._buffers = initialized._buffers.copy()

    def tokens_with_change(self, features, probabilities, variants, positions, types, padding, target_mask, regions, change_token):
        """Build context and candidate tokens, followed by the score-change token."""
        batch = features.size(0)
        global_tokens = self.learned_compression(features)
        (anchors, anchor_positions) = self._anchor_features(features, variants)
        rows = torch.arange(batch, device=features.device)[:, None]
        candidates = self.candidate_projection(features.transpose(1, 2)[rows, positions]) + self.candidate_score_projection(probabilities.transpose(1, 2)[rows, positions]) + self.candidate_type_embedding((types - 1).clamp(0, 1)) + _sinusoidal_positions(positions, self.d_model, dtype=features.dtype)
        candidates = candidates.masked_fill(padding.unsqueeze(-1), 0.0)
        cls = self.cls_token.to(features.dtype).expand(batch, -1, -1) + self.cls_type.to(features.dtype)
        anchors = anchors + self.anchor_types.to(features.dtype) + _sinusoidal_positions(anchor_positions, self.d_model, dtype=features.dtype)
        global_tokens = global_tokens + self.global_type.to(features.dtype) + _sinusoidal_positions(self.global_token_centres[None].expand(batch, -1), self.d_model, dtype=features.dtype)
        tokens = torch.cat((cls, anchors, candidates, global_tokens, change_token), 1)
        mask = torch.cat((torch.zeros(batch, 1 + self.anchor_count, dtype=torch.bool, device=features.device), padding, torch.zeros(batch, self.global_token_count + 1, dtype=torch.bool, device=features.device)), 1)
        return (tokens, mask)

    def forward_pair_with_change(self, reference, alternate, targets, variants, target_types, regions, build_change):
        features = self.high_resolution_features(torch.cat((reference, alternate), 0))
        logits = self.splice_selector(features)
        (rf, af) = features.chunk(2, 0)
        (rl, al) = logits.chunk(2, 0)
        (positions, types, padding) = self._candidate_union(rl.detach(), al.detach(), target_types, targets)
        target_mask = positions.eq(targets[:, None]) & types.eq(target_types[:, None]) & ~padding
        if not torch.all(target_mask.sum(1) == 1):
            raise RuntimeError('Each event must contain exactly one typed target candidate')
        (raw, encoded_change, change_token) = build_change(rl, al)
        change_token = change_token.to(features.dtype).unsqueeze(1)
        result = []
        for (feat, score) in ((rf, rl), (af, al)):
            (tokens, mask) = self.tokens_with_change(feat, score.softmax(1), variants, positions, types, padding, target_mask, regions, change_token)
            result.append(self.output_norm(self.transformer(tokens, src_key_padding_mask=mask)[:, 0]))
        return dict(reference=result[0], alternate=result[1], candidate_positions=positions, candidate_types=types, candidate_padding=padding, target_candidate_mask=target_mask, raw_delta=raw, delta=encoded_change, change_token=change_token)

class FusionSpliceFormer(FeatureFusion):
    def __init__(self):
        config={k:v for k,v in MODEL_CONFIG.items() if k!='architecture_version'}
        super().__init__(**config)
        self.sequence_encoder=TokenEncoder(self.sequence_encoder)
        self.change_token_projection=nn.Linear(32,self.d_model)
        dimension=2*self.d_model
        self.event_heads=nn.ModuleDict({region:nn.Sequential(nn.LayerNorm(dimension),
            nn.Linear(dimension,dimension),nn.GELU(),nn.Dropout(config['dropout']),
            nn.Linear(dimension,2)) for region in self.numeric_dims})
        self.model_config={k:list(v) if isinstance(v,list) else v for k,v in MODEL_CONFIG.items()}

    def forward(self, *, region, site_refseq, site_altseq, numeric, target_index=None, variant_index=None, return_representations=False):
        region = self._normalise_region(region)
        if numeric.ndim != 2 or numeric.size(1) != self.numeric_dims[region]:
            raise ValueError('Wrong numerical feature dimension')
        variants = self._variant_positions(site_refseq, site_altseq)
        if variant_index is not None and (not torch.equal(variants, variant_index.long().reshape(-1))):
            raise ValueError('Variant indices disagree with REF/ALT')
        targets = torch.full_like(variants, REGION_TO_TARGET_INDEX[region]) if target_index is None else target_index.long().reshape(-1)
        if not torch.all(targets == REGION_TO_TARGET_INDEX[region]):
            raise ValueError('Incorrect event target position')
    
        def build_change(rl, al):
            raw = self._delta_features(rl, al, REGION_TO_SELECTOR_CLASS[region], targets)
            encoded = self.delta_encoder(raw)
            return (raw, encoded, self.change_token_projection(encoded))
        paired = self.sequence_encoder.forward_pair_with_change(site_refseq, site_altseq, targets, variants, torch.full_like(variants, REGION_TO_SELECTOR_CLASS[region]), torch.full_like(variants, REGION_TO_INDEX[region]), build_change)
        sequence = self.sequence_difference(paired['reference'], paired['alternate'])
        numbers = self.numeric_encoders[region](numeric.float())
        fused = torch.cat((sequence, numbers), 1)
        logits = self.event_heads[region](fused)
        representations = None
        if return_representations:
            representations = {**paired, 'sequence': sequence, 'numeric': numbers, 'fused': fused, 'variant_position': variants}
        return PredictionOutput(logits=logits, representations=representations)

    @classmethod
    def from_model_config(cls, config):
        if config != MODEL_CONFIG:
            raise ValueError('Checkpoint does not match the released inference architecture')
        return cls()
