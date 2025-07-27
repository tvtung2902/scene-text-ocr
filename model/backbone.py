import torch
import torch.nn as nn
from torchvision.models import resnet34

class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = resnet34(pretrained=pretrained)

        old_conv1 = base_model.conv1

        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                new_conv1.weight = nn.Parameter(old_conv1.weight.mean(dim=1, keepdim=True))

        base_model.conv1 = new_conv1

        self.feature_extractor = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 512, H', W')
        x = self.pool(x)               # (B, 512, 1, W)
        return x
