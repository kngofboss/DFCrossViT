import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.vit_model_face import FeedForward

class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q_input, kv_input):
        B, Nq, _ = q_input.shape
        Nk = kv_input.shape[1]
        H = self.heads

        q_input = self.norm_q(q_input)
        kv_input = self.norm_kv(kv_input)

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = q.view(B, Nq, H, -1).transpose(1, 2)
        k = k.view(B, Nk, H, -1).transpose(1, 2)
        v = v.view(B, Nk, H, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.to_out(out), attn

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        #self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, q, kv):
        out, attn = self.cross_attn(q, kv)
        q = q + out
        #q = q + self.ffn(q)
        return q, attn

class DualImageCrossViT(nn.Module):
    def __init__(self, hybrid_vit, emb_dim=512, depth=1, heads=4, mlp_dim=1024, dropout=0.1, num_patches=64):
        super().__init__()
        self.hybrid_vit = hybrid_vit
        self.cross_attn12 = CrossAttention(dim=emb_dim, heads=heads, dim_head=64, dropout=dropout)
        self.cross_attn21 = CrossAttention(dim=emb_dim, heads=heads, dim_head=64, dropout=dropout)
        self.ffn = FeedForward(emb_dim, mlp_dim, dropout)
        self.fc1 = nn.Linear(num_patches * emb_dim, emb_dim)
        self.fc2 = nn.Linear(num_patches * emb_dim, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)

    def encode(self, img):
        cls_token, patch_tokens, attns = self.hybrid_vit(img)
        return cls_token, patch_tokens

    def forward(self, img1, img2, model_type='mix_swap_cls', heatmap=False):
        cls1, patch1 = self.encode(img1)
        cls2, patch2 = self.encode(img2)
        attn12_list = []
        attn21_list = []

        if model_type == 'mix_swap_cls':
            kv2 = torch.cat([cls1.unsqueeze(1), patch2], dim=1)
            kv1 = torch.cat([cls2.unsqueeze(1), patch1], dim=1)
            cross1, attn12 = self.cross_attn12(patch1, kv2)
            patch1 = patch1 + cross1

            cross2, attn21 = self.cross_attn21(patch2, kv1)
            patch2 = patch2 + cross2
        patch1 = patch1 + self.ffn(patch1)
        patch2 = patch2 + self.ffn(patch2)
        x1 = patch1.flatten(start_dim=1)
        x2 = patch2.flatten(start_dim=1)
        x1 = self.fc1(x1)
        x1 = self.bn1(x1)
        x2 = self.fc2(x2)
        x2 = self.bn2(x2)

        x1_embeddings = F.normalize(x1, dim=-1)
        x2_embeddings = F.normalize(x2, dim=-1)
        if heatmap:
            return (
                x1_embeddings,
                x2_embeddings,
                [attn12, attn21],
                [patch1, patch2]
            )

        return x1_embeddings, x2_embeddings


class DualImageCrossViT1(nn.Module):
    def __init__(
            self,
            hybrid_vit,
            emb_dim=512,
            depth=1,
            heads=4,
            mlp_dim=1024,
            dropout=0.1
    ):
        super().__init__()
        self.hybrid_vit = hybrid_vit

        self.cross_attn12 = CrossAttention(dim=emb_dim, heads=heads, dim_head=64, dropout=dropout)
        self.cross_attn21 = CrossAttention(dim=emb_dim, heads=heads, dim_head=64, dropout=dropout)
        self.patch_ffn = FeedForward(emb_dim, mlp_dim, dropout)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)

    def encode(self, img):
        cls_token, patch_tokens, attns = self.hybrid_vit(img)
        return cls_token, patch_tokens

    def forward(self, img1, img2, model_type='patch', use_cls=False, heatmap=False):
        cls1, patch1 = self.encode(img1)
        cls2, patch2 = self.encode(img2)
        kv2 = torch.cat([cls2.unsqueeze(1), patch2], dim=1)  # Shape: (B, 1 + N_patches, C)
        kv1 = torch.cat([cls1.unsqueeze(1), patch1], dim=1)  # Shape: (B, 1 + N_patches, C)
        mix_kv2 = torch.cat([cls1.unsqueeze(1), patch2], dim=1)
        mix_kv1 = torch.cat([cls2.unsqueeze(1), patch1], dim=1)
        if model_type == 'patch':
            patch1_orig = patch1
            patch2_orig = patch2

            cross1, attn12 = self.cross_attn12(patch1, patch2_orig)
            patch1 = patch1 + cross1

            cross2, attn21 = self.cross_attn21(patch2, patch1_orig)
            patch2 = patch2 + cross2

        elif model_type == 'mix_no_swap_cls':
            cross1, attn12 = self.cross_attn12(patch1, kv2)
            patch1 = patch1 + cross1

            cross2, attn21 = self.cross_attn21(patch2, kv1)
            patch2 = patch2 + cross2

        elif model_type == 'mix_swap_cls':
            cross1, attn12 = self.cross_attn12(patch1, mix_kv2)
            patch1 = patch1 + cross1

            cross2, attn21 = self.cross_attn21(patch2, mix_kv1)
            patch2 = patch2 + cross2

        elif model_type == 'mix_swap_cls_1':
            cross1, attn12 = self.cross_attn12(cls1.unsqueeze(1), mix_kv2)
            cls1 = cls1 + cross1.squeeze(1)

            cross2, attn21 = self.cross_attn21(cls2.unsqueeze(1), mix_kv1)
            cls2 = cls2 + cross2.squeeze(1)

        elif model_type == 'full_cls':
            cross1, attn12 = self.cross_attn12(kv1, kv2)
            patch1 = patch1 + cross1[:, 1:, :]
            cls1 = cls1 + cross1[:, 0, :]

            cross2, attn21 = self.cross_attn21(kv2, kv1)
            patch2 = patch2 + cross2[:, 1:, :]
            cls2 = cls2 + cross2[:, 0, :]

        elif model_type == 'full_cls_1':
            cross1, attn12 = self.cross_attn12(kv1, mix_kv2)
            patch1 = patch1 + cross1[:, 1:, :]
            cls1 = cls1 + cross1[:, 0, :]

            cross2, attn21 = self.cross_attn21(kv2, mix_kv1)
            patch2 = patch2 + cross2[:, 1:, :]
            cls2 = cls2 + cross2[:, 0, :]

        if use_cls:
            cls1 = cls1 + self.patch_ffn(cls1)
            cls2 = cls2 + self.patch_ffn(cls2)
            x1 = self.bn1(cls1)
            x2 = self.bn2(cls2)
        else:
            patch1 = patch1 + self.patch_ffn(patch1)
            patch2 = patch2 + self.patch_ffn(patch2)

            # pooled patch feature
            patch_feat1 = patch1.mean(dim=1)
            patch_feat2 = patch2.mean(dim=1)

            x1 = self.bn1(patch_feat1)
            x2 = self.bn2(patch_feat2)

        # 2. Thực hiện Normalize L2 cho embeddings
        x1_embeddings = F.normalize(x1, dim=-1)
        x2_embeddings = F.normalize(x2, dim=-1)
        if heatmap:
            return (
                x1_embeddings,
                x2_embeddings,
                [attn12, attn21],
                [patch1, patch2]

            )

        return x1_embeddings, x2_embeddings
