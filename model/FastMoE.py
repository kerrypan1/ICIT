import torch
import torch.nn as nn
from fmoe.layers import FMoE
from fmoe.gates.gshard_gate import GShardGate

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=stride, padding=1, groups=in_ch)
        self.pw   = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.dw(x))
        return self.relu(self.pw(x))


class FastMoEPatchExpert(nn.Module):
    """
    CNN expert that takes a 16x16x3 patch (flattened to 768) → CNN → 1x1xemb_dim → back to hidden_size.
    """
    def __init__(self,
                 d_model,
                 in_channels=3,
                 patch_size=16,
                 emb_dim=256):
        super().__init__()
        self.in_ch     = in_channels
        self.p         = patch_size
        self.emb_dim   = emb_dim
        self.hidden_sz = d_model

        # CNN stack: 16→8→4→2→1, output channels = emb_dim
        self.conv_layers = nn.Sequential(
            DepthwiseSeparableConv(self.in_ch,      64, stride=2),
            DepthwiseSeparableConv(     64,     128, stride=2),
            DepthwiseSeparableConv(    128,     self.emb_dim, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        # Project from emb_dim → hidden_size
        self.to_hidden = nn.Linear(self.emb_dim, self.hidden_sz)
        # Residual path
        self.residual  = nn.Linear(self.hidden_sz, self.hidden_sz)

    def forward(self, x, count):
        # x: [num_tokens, hidden_sz]
        N, _ = x.shape
        patches = x.view(N, self.in_ch, self.p, self.p)     # [N, C, p, p]
        feats   = self.conv_layers(patches)                 # [N, emb_dim, 1, 1]
        feats   = feats.view(N, self.emb_dim)               # [N, emb_dim]
        out     = self.to_hidden(feats)                     # [N, hidden_sz]
        res     = self.residual(x)                          # [N, hidden_sz]
        return out + res


class FastMoEEncoder(nn.Module):
    """
    FastMoE Encoder:
    - extract 16×16 patches
    - flatten to d_model=768
    - route through FastMoE experts (GShardGate, top-2)
    - project to emb_dim for decoder
    """
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_dim=256,
                 num_experts=8,
                 k=2,
                 world_size=1):
        super().__init__()
        self.p         = patch_size
        self.C         = in_channels
        self.emb_dim   = emb_dim
        self.hidden_sz = self.C * self.p * self.p
        self.num_experts = num_experts

        # patch extraction
        self.unfold = nn.Unfold(kernel_size=self.p, stride=self.p)

        def expert_factory(d_model):
            return FastMoEPatchExpert(
                d_model=d_model,
                in_channels=self.C,
                patch_size=self.p,
                emb_dim=self.emb_dim
            )

        # FastMoE layer with GShardGate (top-2)
        self.moe = FMoE(
            d_model=self.hidden_sz,
            num_expert=num_experts,
            expert=expert_factory,
            top_k=k,
            world_size=world_size,
            gate=GShardGate,
        )
        self.moe.gate.loss = torch.tensor(0.0, device=next(self.parameters()).device)
        print("Gate class:", self.moe.gate.__class__.__name__)

        # project from patch output (hidden_sz) → emb_dim
        self.to_emb = nn.Linear(self.hidden_sz, self.emb_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches    = self.unfold(x).transpose(1, 2)    # [B, N, D]
        B, N, D    = patches.shape
        flat       = patches.reshape(B * N, D)         # [B*N, D]

        moe_out = self.moe(flat)                        # [B*N, hidden_sz]
        
        l_aux = self.moe.gate.get_loss(clear=True)
        if l_aux is None:
            l_aux = torch.tensor(0.0, device=x.device)
        moe_out = moe_out.view(B, N, self.hidden_sz)   # [B, N, hidden_sz]

        emb   = self.to_emb(moe_out)                   # [B, N, emb_dim]
        h_p   = H // self.p
        w_p   = W // self.p
        feats = emb.transpose(1, 2).view(B, self.emb_dim, h_p, w_p)
        return feats, l_aux

class CNNMoE(nn.Module):
    """
    Full CNN-MoE model:
    - input 256x256x3
    - FastMoE encoder → 16x16xemb_dim
    - decoder upsamples back to 256x256xout_channels
    """
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_dim=256,
                 num_experts=8,
                 k=2,
                 out_channels=1,
                 world_size=1):
        super().__init__()
        self.encoder = FastMoEEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_experts=num_experts,
            k=k,
            world_size=world_size
        )
        skip_size = 256 // patch_size
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_dim // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(skip_size)
        )

        self.decoder_pre = nn.Sequential(
            nn.Conv2d(emb_dim + emb_dim // 4, emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True)
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 2, emb_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim // 8),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 8, emb_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim // 16),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(emb_dim // 16, emb_dim // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_dim // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        skip     = self.skip_conv(x)
        encoded, l_aux = self.encoder(x)
        combined = torch.cat([encoded, skip], dim=1)

        x = self.decoder_pre(combined)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.final_conv(x)
        return x, l_aux

'''
class FastMoEEncoder(nn.Module):
    """
    FastMoE Encoder:
    - extract 16×16 patches
    - flatten to d_model=768
    - route through FastMoE experts (GShardGate, top-2)
    - **apply attention + FFN over tokens**
    - project to emb_dim for decoder
    """
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_dim=256,
                 num_experts=8,
                 k=2,
                 world_size=1):
        super().__init__()
        self.p         = patch_size
        self.C         = in_channels
        self.emb_dim   = emb_dim
        self.hidden_sz = self.C * self.p * self.p
        self.num_experts = num_experts

        # patch extraction
        self.unfold = nn.Unfold(kernel_size=self.p, stride=self.p)

        def expert_factory(d_model):
            return FastMoEPatchExpert(
                d_model=d_model,
                in_channels=self.C,
                patch_size=self.p,
                emb_dim=self.emb_dim
            )

        # FastMoE layer with GShardGate (top-2)
        self.moe = FMoE(
            d_model=self.hidden_sz,
            num_expert=num_experts,
            expert=expert_factory,
            top_k=k,
            world_size=world_size,
            gate=GShardGate,
        )
        self.moe.gate.loss = torch.tensor(0.0, device=next(self.parameters()).device)
        print("Gate class:", self.moe.gate.__class__.__name__)

        # lightweight Transformer block over tokens (post-expert, pre-projection)
        # pick a valid number of heads for hidden_sz
        def _pick_heads(d):
            for h in (16, 12, 8, 6, 4, 3, 2, 1):
                if d % h == 0:
                    return h
            return 1
        heads = _pick_heads(self.hidden_sz)

        self.attn_norm = nn.LayerNorm(self.hidden_sz)
        self.attn = nn.MultiheadAttention(self.hidden_sz, heads, batch_first=True)
        self.attn_drop = nn.Dropout(0.1)

        self.ffn_norm = nn.LayerNorm(self.hidden_sz)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_sz, 4 * self.hidden_sz),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.hidden_sz, self.hidden_sz),
            nn.Dropout(0.1),
        )

        # project from patch output (hidden_sz) → emb_dim
        self.to_emb = nn.Linear(self.hidden_sz, self.emb_dim)

    def _pos_emb_1d(self, dim, positions, *, device, dtype):
        # positions: [M] in [-1, 1]
        half = dim // 2
        omega = torch.arange(half, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / half))
        out = torch.einsum('m,d->md', positions, omega)       # [M, dim/2]
        return torch.cat([torch.sin(out), torch.cos(out)], 1) # [M, dim]

    def _build_2d_sincos_pos_embed(self, dim, h, w, *, device, dtype):
        # normalized coords in [-1, 1]
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        # split channels: dim_h + dim_w = dim (favor even split)
        dim_h = dim // 2
        dim_w = dim - dim_h
        emb_y = self._pos_emb_1d(dim_h, ys, device=device, dtype=dtype)   # [h, dim_h]
        emb_x = self._pos_emb_1d(dim_w, xs, device=device, dtype=dtype)   # [w, dim_w]
        pos = torch.cat([
            emb_y[:, None, :].expand(h, w, dim_h),
            emb_x[None, :, :].expand(h, w, dim_w)
        ], dim=2)                                                         # [h, w, dim]
        return pos.view(1, h * w, dim)                                    # [1, N, dim]

    def forward(self, x):
        B, C, H, W = x.shape
        patches    = self.unfold(x).transpose(1, 2)    # [B, N, D]
        B, N, D    = patches.shape
        flat       = patches.reshape(B * N, D)         # [B*N, D]

        moe_out = self.moe(flat)                       # [B*N, hidden_sz]
        l_aux = self.moe.gate.get_loss(clear=True)
        if l_aux is None:
            l_aux = torch.zeros((), device=x.device, dtype=moe_out.dtype)
        else:
            l_aux = l_aux.to(moe_out.dtype)

        tokens = moe_out.view(B, N, self.hidden_sz)    # [B, N, hidden_sz]

        # add 2D positional (location) embedding over tokens
        h_p, w_p = H // self.p, W // self.p
        pos = self._build_2d_sincos_pos_embed(
            self.hidden_sz, h_p, w_p, device=x.device, dtype=tokens.dtype
        )                                              # [1, N, hidden_sz]
        tokens = tokens + pos                          # position-aware tokens

        # attention + FFN block
        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.attn_drop(attn_out)
        ffn_in = self.ffn_norm(tokens)
        tokens = tokens + self.ffn(ffn_in)

        emb   = self.to_emb(tokens)                    # [B, N, emb_dim]
        feats = emb.transpose(1, 2).view(B, self.emb_dim, h_p, w_p)
        return feats, l_aux
'''
