import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)