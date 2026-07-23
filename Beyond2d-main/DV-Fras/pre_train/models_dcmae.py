import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


# --------------------------------------------------------
# Fixed 2D sine-cosine positional embeddings
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid.shape[1], grid.shape[2])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("The positional-embedding dimension must be even.")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("Each 1D positional-embedding dimension must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# --------------------------------------------------------
# Transformer modules
# --------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        if any(size % patch != 0 for size, patch in zip(self.img_size, self.patch_size)):
            raise ValueError(
                f"img_size={self.img_size} must be divisible by patch_size={self.patch_size}."
            )
        self.grid_size = tuple(
            size // patch for size, patch in zip(self.img_size, self.patch_size)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"Expected input size {self.img_size}, but received {tuple(x.shape[-2:])}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class CrossViewGuidance(nn.Module):
    """Multi-head cross-attention for masked positions of the target view."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, mask_queries, complete_guidance):
        """
        mask_queries: [B, N_mask, D_d]
        complete_guidance: [B, N, D_d]
        """
        batch_size, num_queries, dim = mask_queries.shape
        num_keys = complete_guidance.shape[1]
        head_dim = dim // self.num_heads

        q = (
            self.w_q(mask_queries)
            .reshape(batch_size, num_queries, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.w_k(complete_guidance)
            .reshape(batch_size, num_keys, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.w_v(complete_guidance)
            .reshape(batch_size, num_keys, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        guided = (attn @ v).transpose(1, 2).reshape(batch_size, num_queries, dim)
        return self.proj(guided)


class DCMAE(nn.Module):
    """
    Dual-view cross-guided masked autoencoder.

    For each reconstruction direction:
      1. only visible target-view tokens are encoded;
      2. the complete orthogonal view supplies keys and values;
      3. shared mask-token queries generate features only for masked positions;
      4. visible and guided masked-position features are reassembled before decoding.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        patch_h, patch_w = self.patch_embed.patch_size
        self.patch_area = patch_h * patch_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=False,
        )
        self.cross_guidance = CrossViewGuidance(
            decoder_embed_dim,
            decoder_num_heads,
            qkv_bias=True,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.patch_area * in_chans,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        encoder_pos = get_2d_sincos_pos_embed(
            self.embed_dim,
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(encoder_pos).unsqueeze(0))

        decoder_pos = get_2d_sincos_pos_embed(
            self.decoder_embed_dim,
            self.patch_embed.grid_size,
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos).unsqueeze(0))

        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def patchify(self, imgs):
        patch_h, patch_w = self.patch_embed.patch_size
        height = imgs.shape[2] // patch_h
        width = imgs.shape[3] // patch_w
        x = imgs.reshape(
            imgs.shape[0],
            imgs.shape[1],
            height,
            patch_h,
            width,
            patch_w,
        )
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(
            imgs.shape[0],
            height * width,
            patch_h * patch_w * imgs.shape[1],
        )

    def unpatchify(self, patches):
        patch_h, patch_w = self.patch_embed.patch_size
        grid_h, grid_w = self.patch_embed.grid_size
        channels = self.decoder_pred.out_features // (patch_h * patch_w)
        x = patches.reshape(
            patches.shape[0],
            grid_h,
            grid_w,
            patch_h,
            patch_w,
            channels,
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            patches.shape[0],
            channels,
            grid_h * patch_h,
            grid_w * patch_w,
        )

    def _embed_image(self, image):
        return self.patch_embed(image) + self.pos_embed[:, 1:, :]

    def _encode_patch_tokens(self, patch_tokens):
        batch_size = patch_tokens.shape[0]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(
            batch_size, -1, -1
        )
        x = torch.cat([cls, patch_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 1:, :]

    @staticmethod
    def random_masking(tokens, mask_ratio):
        """
        Independently mask each example.

        Returned binary mask follows the MAE convention:
          0 = visible/kept, 1 = masked/removed.
        """
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), received {mask_ratio}.")

        batch_size, num_tokens, dim = tokens.shape
        num_visible = int(num_tokens * (1.0 - mask_ratio))
        if num_visible < 1 or num_visible >= num_tokens:
            raise ValueError(
                f"mask_ratio={mask_ratio} leaves {num_visible} visible tokens "
                f"out of {num_tokens}."
            )

        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_visible]
        ids_mask = ids_shuffle[:, num_visible:]

        visible = torch.gather(
            tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
        )
        mask = torch.ones(batch_size, num_tokens, device=tokens.device)
        mask.scatter_(1, ids_keep, 0.0)
        return visible, mask, ids_keep, ids_mask

    @staticmethod
    def _gather_positions(position_embeddings, indices):
        batch_size = indices.shape[0]
        dim = position_embeddings.shape[-1]
        expanded = position_embeddings.expand(batch_size, -1, -1)
        return torch.gather(
            expanded,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, dim),
        )

    def forward_decoder_guided(
        self,
        visible_own,
        complete_other,
        ids_keep_own,
        ids_mask_own,
    ):
        """Reconstruct one view with complete orthogonal-view guidance."""
        batch_size = visible_own.shape[0]

        visible_own = self.decoder_embed(visible_own)
        complete_other = self.decoder_embed(complete_other)

        pos_visible = self._gather_positions(self.decoder_pos_embed, ids_keep_own)
        pos_masked = self._gather_positions(self.decoder_pos_embed, ids_mask_own)
        visible_own = visible_own + pos_visible

        mask_queries = self.mask_token.expand(
            batch_size, ids_mask_own.shape[1], -1
        ) + pos_masked
        complete_guidance = complete_other + self.decoder_pos_embed

        guided_masked = self.cross_guidance(mask_queries, complete_guidance)

        full_tokens = torch.empty(
            batch_size,
            self.num_patches,
            self.decoder_embed_dim,
            device=visible_own.device,
            dtype=visible_own.dtype,
        )
        full_tokens.scatter_(
            1,
            ids_keep_own.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
            visible_own,
        )
        full_tokens.scatter_(
            1,
            ids_mask_own.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
            guided_masked,
        )

        x = full_tokens
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        prediction = self.decoder_pred(x)
        return prediction, guided_masked, full_tokens

    def forward_loss(self, images, predictions, mask):
        target = self.patchify(images)
        patch_loss = (predictions - target).pow(2).mean(dim=-1)
        denominator = mask.sum().clamp_min(1.0)
        return (patch_loss * mask).sum() / denominator

    @staticmethod
    def consistency_loss(complete_ap, complete_lat):
        pooled_ap = complete_ap.mean(dim=1)
        pooled_lat = complete_lat.mean(dim=1)
        cosine = F.cosine_similarity(pooled_ap, pooled_lat, dim=-1)
        return 1.0 - cosine.mean()

    def forward(
        self,
        img_ap,
        img_lat,
        mask_ratio=0.60,
        lambda_consist=0.10,
        return_details=False,
    ):
        ap_tokens = self._embed_image(img_ap)
        lat_tokens = self._embed_image(img_lat)

        ap_visible, mask_ap, ids_keep_ap, ids_mask_ap = self.random_masking(
            ap_tokens, mask_ratio
        )
        lat_visible, mask_lat, ids_keep_lat, ids_mask_lat = self.random_masking(
            lat_tokens, mask_ratio
        )

        # Masked-target encodings: X_tilde^v -> Z_tilde^v.
        encoded_visible_ap = self._encode_patch_tokens(ap_visible)
        encoded_visible_lat = self._encode_patch_tokens(lat_visible)

        # Complete guidance encodings: X^v -> Z^v.
        encoded_complete_ap = self._encode_patch_tokens(ap_tokens)
        encoded_complete_lat = self._encode_patch_tokens(lat_tokens)

        pred_ap, _, reassembled_ap = self.forward_decoder_guided(
            visible_own=encoded_visible_ap,
            complete_other=encoded_complete_lat,
            ids_keep_own=ids_keep_ap,
            ids_mask_own=ids_mask_ap,
        )
        pred_lat, _, reassembled_lat = self.forward_decoder_guided(
            visible_own=encoded_visible_lat,
            complete_other=encoded_complete_ap,
            ids_keep_own=ids_keep_lat,
            ids_mask_own=ids_mask_lat,
        )

        loss_rec_ap = self.forward_loss(img_ap, pred_ap, mask_ap)
        loss_rec_lat = self.forward_loss(img_lat, pred_lat, mask_lat)
        loss_consist = self.consistency_loss(
            reassembled_ap,
            reassembled_lat,
        )
        loss = loss_rec_ap + loss_rec_lat + lambda_consist * loss_consist

        outputs = (loss, pred_ap, pred_lat, mask_ap, mask_lat)
        if not return_details:
            return outputs

        details = {
            "loss_rec_ap": loss_rec_ap,
            "loss_rec_lat": loss_rec_lat,
            "loss_consist": loss_consist,
        }
        return outputs + (details,)

    def encode_image(self, image):
        """Encode a complete image and return its patch-token features."""
        return self._encode_patch_tokens(self._embed_image(image))


def dcmae_vit_small_patch16(**kwargs):
    return DCMAE(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
