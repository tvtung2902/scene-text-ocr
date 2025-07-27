import torch
import torch.nn as nn
from backbone import ResNet34Backbone
from sequence_head import BiLSTM

class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.3, pretrained_backbone=True):
        super(OCRModel, self).__init__()
        self.backbone = ResNet34Backbone(pretrained=pretrained_backbone)
        self.lstm = BiLSTM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.lstm(x)
        return x

if __name__ == '__main__':
    model = OCRModel(vocab_size=38, hidden_size=512, n_layers=2)
    dummy_input = torch.randn(1, 1, 100, 420)  # (B, C, H, W)
    output = model(dummy_input)
    print("Final output shape:", output.shape)
