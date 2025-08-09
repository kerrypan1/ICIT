import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, num_patches):
        super().__init__()
        self.proj = nn.Linear(in_channels, emb_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))

    def forward(self, x): 
        x = self.proj(x) + self.pos_embed
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=512, depth=6, heads=8, mlp_dim=3072):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, num_classes=1, emb_dim=512):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.input_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.encoder1 = resnet.layer1  
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3 
        self.encoder4 = resnet.layer4 

        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(512, emb_dim, num_patches)
        self.transformer = TransformerEncoder(emb_dim=emb_dim)

        self.decoder4 = DecoderBlock(emb_dim, 256, 128)
        self.decoder3 = DecoderBlock(128, 128, 64)
        self.decoder2 = DecoderBlock(64, 64, 32)
        self.decoder1 = DecoderBlock(32, 3, 32)     

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):         
        B = x.size(0)
        x0 = x
        x1 = self.input_conv(x)      
        x2 = self.encoder1(x1)       
        x3 = self.encoder2(x2)       
        x4 = self.encoder3(x3)      
        x5 = self.encoder4(x4)    

        patches = x5.flatten(2).transpose(1, 2) 
        tokens = self.patch_embed(patches)      
        tokens = self.transformer(tokens)       
        x_trans = tokens.transpose(1, 2).reshape(B, -1, 16, 16)  

        x = self.decoder4(x_trans, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x0)

        out = self.final_conv(x)  
        return out
