import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, W, C, 1)
        x = x.view(x.size(0), x.size(1), -1)  # (B, W, 768)
        x, _ = self.lstm(x)  # (B, W, 2*hidden)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # (W, B, Class)
        return x
