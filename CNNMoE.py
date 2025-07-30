import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=stride, padding=1, groups=in_ch)
        self.pw   = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.dw(x))
        return self.relu(self.pw(x))


class SparseMoEEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=32,
                 emb_dim=64, num_experts=8, k=2):
        super().__init__()
        self.patch_size  = patch_size
        self.emb_dim     = emb_dim
        self.num_experts = num_experts
        self.k           = k

        # 1) ultra‑fast patch extractor
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # 2) gating MLP
        self.gate = nn.Sequential(
            nn.Linear(in_channels * patch_size * patch_size, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, num_experts)
        )

        # 3) expert heads
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(in_channels, emb_dim, stride=2),
                DepthwiseSeparableConv(emb_dim,     emb_dim, stride=2),
                nn.AdaptiveAvgPool2d(1)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        p, D, E, k = self.patch_size, self.emb_dim, \
                     self.num_experts, self.k
        h_p, w_p = H // p, W // p
        N = h_p * w_p

        # ---- 1) extract & prepare patches ----
        # [B, C*p*p, N] → [B, N, C*p*p]
        patches_flat = self.unfold(x).transpose(1,2)

        # ---- 2) gating ----
        logits       = self.gate(patches_flat)            # [B, N, E]
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)  # [B, N, k]
        weights      = F.softmax(topk_vals, dim=-1)           # [B, N, k]

        # ---- 3) run each expert once over all patches ----
        B_N = B * N
        # back to spatial patches: [B, N, C*p*p] → [B*N, C, p, p]
        patches_spatial = patches_flat.reshape(B_N, C, p, p)

        # expert_outs: [B*N, E, D]
        expert_outs = torch.stack([
            h(patches_spatial).reshape(B_N, D)
            for h in self.expert_heads
        ], dim=1)

        # ---- 4) build routing via one‑hot + weighted sum ----
        idx_flat = topk_idx.reshape(B_N, k)   # [B*N, k]
        w_flat   = weights.reshape(B_N, k)    # [B*N, k]

        one_hot  = F.one_hot(idx_flat, num_classes=E).to(x.dtype)  # [B*N, k, E]
        routing  = (one_hot * w_flat.unsqueeze(-1)).sum(dim=1)     # [B*N, E]

        # ---- 5) fuse experts in one einsum ----
        # out: [B*N, D]
        out = torch.einsum("ne,ned->nd", routing, expert_outs)

        # ---- 6) reshape back to [B, emb_dim, h_p, w_p] ----
        return out.reshape(B, h_p, w_p, D).permute(0,3,1,2).contiguous()


class CNNMoE(nn.Module):
    def __init__(self, in_channels=3, patch_size=32,
                 emb_dim=64, num_experts=8, k=2,
                 out_channels=1):
        super().__init__()
        # script the encoder once for max fusion
        enc = SparseMoEEncoder(in_channels, patch_size,
                               emb_dim, num_experts, k)
        self.encoder = torch.jit.script(enc)

        # same lightweight decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(emb_dim,     emb_dim,     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=patch_size,
                        mode='bilinear', align_corners=False),
            nn.Conv2d(emb_dim,     emb_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim//2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)   # [B, emb_dim, H/patch, W/patch]
        return self.decoder(x)
