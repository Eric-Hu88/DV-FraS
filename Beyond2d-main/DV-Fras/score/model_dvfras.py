from pathlib import Path
import sys

import torch
import torch.nn as nn

PRETRAIN_DIR = Path(__file__).resolve().parents[1] / "pre_train"
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from models_dcmae import Block, PatchEmbed, get_2d_sincos_pos_embed


class DCMAEEncoder(nn.Module):
    """Decoder-free ViT-S/16 encoder transferred from DC-MAE."""

    def __init__(self, img_size=224, in_chans=1, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, 16, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._initialize_weights()

    def _initialize_weights(self):
        position = get_2d_sincos_pos_embed(
            self.embed_dim, self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(position).unsqueeze(0))
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image):
        patches = self.patch_embed(image) + self.pos_embed[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(image.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]


class LocalROIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 32, 64, 128, 256)
        blocks = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image):
        return self.pool(self.blocks(image)).flatten(1)


def _checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


class FHAssessmentNet(nn.Module):
    def __init__(self, pretrain_path=None, num_classes=4):
        super().__init__()
        self.encoder_global = DCMAEEncoder()
        if pretrain_path:
            state = _checkpoint_state(pretrain_path)
            encoder_state = {}
            for key, value in state.items():
                key = key.removeprefix("module.").removeprefix("encoder_global.")
                if key in self.encoder_global.state_dict():
                    encoder_state[key] = value
            missing, unexpected = self.encoder_global.load_state_dict(encoder_state, strict=False)
            if unexpected or any(not key.startswith("pos_embed") for key in missing):
                raise RuntimeError(
                    f"Incompatible DC-MAE checkpoint. Missing={missing}, unexpected={unexpected}"
                )

        self.encoder_local = LocalROIEncoder()
        self.film = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 2 * 2 * 256),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384 + 256),
            nn.Linear(384 + 256, num_classes),
        )
        self._l2sp_reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.encoder_global.named_parameters()
        } if pretrain_path else {}

    def _apply(self, function):
        super()._apply(function)
        self._l2sp_reference = {
            name: function(reference) for name, reference in self._l2sp_reference.items()
        }
        return self

    def get_l2_sp_loss(self):
        if not self._l2sp_reference:
            return next(self.parameters()).new_zeros(())
        return sum(
            (parameter - self._l2sp_reference[name]).pow(2).sum()
            for name, parameter in self.encoder_global.named_parameters()
        )

    def forward(self, global_images, local_images):
        """
        global_images: [B, 2 views, 1, 224, 224]
        local_images: [B, 2 views, 2 cortices, 1, 224, 224]
        """
        batch_size = global_images.shape[0]
        global_tokens = self.encoder_global(global_images.flatten(0, 1))
        global_features = global_tokens.mean(dim=1).reshape(batch_size, 2, 384)

        local_features = self.encoder_local(local_images.flatten(0, 2))
        local_features = local_features.reshape(batch_size, 2, 2, 256)

        film = self.film(global_features).reshape(batch_size, 2, 2, 2, 256)
        gamma, beta = film.unbind(dim=3)
        modulated_local = gamma * local_features + beta
        fused = torch.cat(
            (global_features.unsqueeze(2).expand(-1, -1, 2, -1), modulated_local),
            dim=-1,
        )
        return self.classifier(fused)  # [B, AP/LAT, two cortices, four scores]
