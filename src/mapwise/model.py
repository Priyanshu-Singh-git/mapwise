"""U-Net for land-cover segmentation: pretrained ResNet-34 encoder, decoder written from scratch.

The encoder is torchvision's ResNet-34 used as a feature pyramid; the decoder is ours —
four upsampling blocks, each fusing the skip connection from the matching encoder stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_bn_relu(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """Upsample x2, concatenate the encoder skip, then two conv-bn-relu."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # guard against odd input sizes drifting by a pixel
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetResNet34(nn.Module):
    """ResNet encoder (ImageNet) + from-scratch U-Net decoder -> per-pixel class logits.

    Keeps its historical name so existing checkpoints load, but `encoder` selects the torso:
    resnet34 (skip channels 256/128/64/64) or resnet50 (1024/512/256/64, bottleneck blocks).
    """

    def __init__(self, n_classes=7, pretrained=True, encoder="resnet34"):
        super().__init__()
        self.encoder_name = encoder
        if encoder == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            r = torchvision.models.resnet50(weights=w)
            c4, c3, c2, c1 = 2048, 1024, 512, 256
        else:
            w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet34(weights=w)
            c4, c3, c2, c1 = 512, 256, 128, 64

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # /2   64
        self.pool = r.maxpool                               # /4
        self.layer1 = r.layer1                              # /4   c1
        self.layer2 = r.layer2                              # /8   c2
        self.layer3 = r.layer3                              # /16  c3
        self.layer4 = r.layer4                              # /32  c4

        self.dec4 = DecoderBlock(c4, c3, 256)               # /16
        self.dec3 = DecoderBlock(256, c2, 128)              # /8
        self.dec2 = DecoderBlock(128, c1, 64)               # /4
        self.dec1 = DecoderBlock(64, 64, 48)                # /2
        self.dec0 = DecoderBlock(48, 0, 32)                 # /1
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)          # /2  64
        s1 = self.layer1(self.pool(s0))   # /4  64
        s2 = self.layer2(s1)       # /8  128
        s3 = self.layer3(s2)       # /16 256
        s4 = self.layer4(s3)       # /32 512

        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)
        return self.head(d)


if __name__ == "__main__":
    net = UNetResNet34(pretrained=False)
    y = net(torch.zeros(2, 3, 512, 512))
    n = sum(p.numel() for p in net.parameters())
    print("output", tuple(y.shape), "| params", f"{n/1e6:.1f}M")
