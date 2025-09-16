import torch
import torch.nn as nn


class totalModel(nn.Module):
    def __init__(self, num_classes, RSNet_hidden_size, RSNet_heads, RSNet_layers, RSNet_ffn_size,
                 RSNet_dt_rank=None):
        super(totalModel, self).__init__()
        self.num_classes = num_classes
        self.RSNet_hidden_size = RSNet_hidden_size
        self.RSNet_heads = RSNet_heads
        self.RSNet_layers = RSNet_layers
        self.RSNet_ffn_size = RSNet_ffn_size
        self.RSNet_dt_rank = RSNet_dt_rank

        self.BRNet = RSNet(layers=RSNet_layers, hidden_dim=RSNet_hidden_size, ffn_size=RSNet_ffn_size,
                            heads=RSNet_heads, dt_rank=self.RSNet_dt_rank, double_v_dim=False).to(DEVICE.get_device())

        self.norm = nn.LayerNorm(RSNet_hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(RSNet_hidden_size, int(RSNet_hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(RSNet_hidden_size / 2), int(RSNet_hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(RSNet_hidden_size / 4), num_classes),
        ).to(DEVICE.get_device())

    def forward(self, X, chunkwise_size):
        y, r = self.BRNet.forward_chunkwise(X, chunkwise_size)
        pooled = y[:, -1, :]
        pooled = self.classifier(pooled)
        return pooled