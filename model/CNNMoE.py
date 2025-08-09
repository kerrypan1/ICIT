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


class PatchExpertCNN(nn.Module):
    """Efficient CNN expert that processes 16x16x3 patches to 1x1x256"""
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        # 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        self.conv1 = DepthwiseSeparableConv(in_channels, 64, stride=2)  # 16->8
        self.conv2 = DepthwiseSeparableConv(64, 128, stride=2)          # 8->4
        self.conv3 = DepthwiseSeparableConv(128, out_channels, stride=2) # 4->2
        self.pool = nn.AdaptiveAvgPool2d(1)                            # 2->1
        
    def forward(self, x):
        # x: [B, 3, 16, 16] -> [B, 256, 1, 1]
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)  # [B, 256]


class SparseMoEEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16,
                 emb_dim=256, num_experts=8, k=2):
        super().__init__()
        self.patch_size  = patch_size
        self.emb_dim     = emb_dim
        self.num_experts = num_experts
        self.k           = min(k, num_experts)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        gate_input_dim = in_channels * patch_size * patch_size  # 768
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, emb_dim // 2),  # 768 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(emb_dim // 2, num_experts)     # 128 -> 8
        )

        self.expert_heads = nn.ModuleList([
            PatchExpertCNN(in_channels, emb_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        p, D, E, k = self.patch_size, self.emb_dim, \
                     self.num_experts, self.k
        h_p, w_p = H // p, W // p
        N = h_p * w_p

        # [B, C*p*p, N] → [B, N, C*p*p]
        patches_flat = self.unfold(x).transpose(1,2)

        logits = self.gate(patches_flat)                    # [B, N, E]
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)  # [B, N, k]
        weights = F.softmax(topk_vals, dim=-1)               # [B, N, k]

        B_N = B * N
        patches_spatial = patches_flat.reshape(B_N, C, p, p)

        idx_flat = topk_idx.reshape(B_N, k)   # [B*N, k] - which experts each patch uses
        w_flat   = weights.reshape(B_N, k)    # [B*N, k] - corresponding weights
        
        patch_indices = torch.arange(B_N, device=x.device).unsqueeze(1).expand(B_N, k).reshape(-1)  # [B*N*k]
        expert_indices = idx_flat.reshape(-1)  # [B*N*k] - which expert each assignment goes to
        assignment_weights = w_flat.reshape(-1)  # [B*N*k] - weight for each assignment

        out = torch.zeros(B_N, D, device=x.device, dtype=x.dtype)

        for expert_id, expert in enumerate(self.expert_heads):
            # Find which patches are routed to this expert
            mask = (expert_indices == expert_id)
            if not mask.any():
                continue
                
            assigned_patch_ids = patch_indices[mask]  # which patches
            assigned_weights = assignment_weights[mask]  # their weights
            assigned_patches = patches_spatial[assigned_patch_ids]  # [M, C, p, p]
            
            # Run expert only on assigned patches
            expert_output = expert(assigned_patches)  # [M, D]

            weighted_output = expert_output * assigned_weights.unsqueeze(1)  # [M, D]
            out.index_add_(0, assigned_patch_ids, weighted_output)

        del patches_spatial, patch_indices, expert_indices, assignment_weights

        if self.training:
            expert_usage = F.one_hot(topk_idx.reshape(-1), num_classes=E).float().mean(0)
            target_usage = torch.ones(E, device=x.device) / E  
            load_balance_loss = F.mse_loss(expert_usage, target_usage) * 0.0001  
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)

        features = out.reshape(B, h_p, w_p, D).permute(0,3,1,2).contiguous()
        return features, load_balance_loss


class CNNMoE(nn.Module):
    def __init__(self, in_channels=3, patch_size=16,
                 emb_dim=512, num_experts=16, k=2,  
                 out_channels=1):
        super().__init__()
        self.patch_size = patch_size
        
        self.encoder = SparseMoEEncoder(in_channels, patch_size,
                                       emb_dim, num_experts, k)

        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(256 // patch_size)  
        )

        self.decoder_pre = nn.Sequential(
            nn.Conv2d(emb_dim + emb_dim // 4, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 16x16 -> 32x32
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 2, emb_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 32x32 -> 64x64
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 64x64 -> 128x128
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim // 8, emb_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 128x128 -> 256x256
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(emb_dim // 16, emb_dim // 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        skip = self.skip_conv(x)  # [B, emb_dim//4, 16, 16]
        
        encoded, aux_loss = self.encoder(x)  # [B, emb_dim, 16, 16], scalar
        
        combined = torch.cat([encoded, skip], dim=1)  # [B, emb_dim + emb_dim//4, 16, 16]
        
        x = self.decoder_pre(combined)   # [B, emb_dim, 16, 16]
        x = self.upsample1(x)           # [B, emb_dim//2, 32, 32]
        x = self.upsample2(x)           # [B, emb_dim//4, 64, 64]
        x = self.upsample3(x)           # [B, emb_dim//8, 128, 128]  
        x = self.upsample4(x)           # [B, emb_dim//16, 256, 256]
        x = self.final_conv(x)          # [B, 1, 256, 256]
        
        return x, aux_loss