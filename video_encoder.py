import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.checkpoint import checkpoint


class VideoEncoder(nn.Module):
    """Static + temporal motion encoder.

    Returns:
        {
            "static": [latent_dim],
            "motion": [T, latent_dim],
        }
    """

    def __init__(self, latent_dim=128, temporal_hidden=256, feature_chunk_size=8, checkpoint_backbone=True):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_chunk_size = max(1, int(feature_chunk_size))
        self.checkpoint_backbone = bool(checkpoint_backbone)
        self.frame_norm = nn.LayerNorm(512)
        self.static_attn = nn.Sequential(
            nn.Linear(512, temporal_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(temporal_hidden, 1),
        )
        self.temporal = nn.GRU(
            input_size=1024,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.static_fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, temporal_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(temporal_hidden, latent_dim),
        )
        self.motion_fc = nn.Sequential(
            nn.LayerNorm(temporal_hidden * 2),
            nn.Linear(temporal_hidden * 2, temporal_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(temporal_hidden, latent_dim),
        )

    def _temporal_inputs(self, feats):
        prev = torch.cat([feats[:1], feats[:-1]], dim=0)
        deltas = feats - prev
        return torch.cat([feats, deltas], dim=-1)

    def _encode_frame_features(self, frames):
        chunks = []
        for start in range(0, frames.shape[0], self.feature_chunk_size):
            frame_chunk = frames[start:start + self.feature_chunk_size]
            if self.training and self.checkpoint_backbone:
                feat_chunk = checkpoint(self.backbone, frame_chunk, use_reentrant=False)
            else:
                feat_chunk = self.backbone(frame_chunk)
            chunks.append(feat_chunk)
        return torch.cat(chunks, dim=0)

    def forward(self, frames):
        if isinstance(frames, (list, tuple)):
            batch = []
            for img in frames:
                if img.ndim == 4:
                    batch.append(img.squeeze(0))
                else:
                    batch.append(img)
            frames = torch.stack(batch, dim=0)

        feats = self.frame_norm(self._encode_frame_features(frames))  # [T, 512]
        attn_logits = self.static_attn(feats).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=0).unsqueeze(-1)
        static_feat = (attn * feats).sum(dim=0)
        static_latent = self.static_fc(static_feat)
        temporal_inputs = self._temporal_inputs(feats).unsqueeze(0)
        motion_seq, _ = self.temporal(temporal_inputs)
        motion_latents = self.motion_fc(motion_seq.squeeze(0))
        return {"static": static_latent, "motion": motion_latents}
