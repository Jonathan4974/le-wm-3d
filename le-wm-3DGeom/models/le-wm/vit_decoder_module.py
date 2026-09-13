import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q, kv):
        """
        q : (B, P, D)
        kv: (B, 1, D)
        """

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
        )

        q = q + attn_out
        q = q + self.mlp(self.norm2(q))

        return q
    
class VisualizationDecoder(nn.Module):
    def __init__(
        self,
        emb_dim=192,
        decoder_dim=512,
        depth=6,
        num_heads=8,
        image_size=224,
        patch_size=16,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        self.cls_proj = nn.Linear(
            emb_dim,
            decoder_dim,
        )

        self.query_tokens = nn.Parameter(
            torch.randn(
                1,
                self.num_patches,
                decoder_dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=decoder_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(decoder_dim)

        self.patch_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * 3,
        )

    def unpatchify(self, patches):
        """
        patches:
            (B,196,768)

        return:
            (B,3,224,224)
        """

        B = patches.shape[0]
        p = self.patch_size

        h = w = self.image_size // p

        patches = patches.reshape(
            B,
            h,
            w,
            p,
            p,
            3,
        )

        patches = patches.permute(
            0,
            5,
            1,
            3,
            2,
            4,
        )

        images = patches.reshape(
            B,
            3,
            self.image_size,
            self.image_size,
        )

        return images

    def forward(self, emb):
        """
        emb:
            (B,192)
        """

        B = emb.shape[0]

        kv = self.cls_proj(emb).unsqueeze(1)

        q = self.query_tokens.expand(B, -1, -1)

        for block in self.blocks:
            q = block(q, kv)

        q = self.norm(q)

        patch_pixels = self.patch_pred(q)

        img = self.unpatchify(patch_pixels)

        return img