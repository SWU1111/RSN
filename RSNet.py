import torch
import torch.nn as nn


class RSNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, dt_rank=None, double_v_dim=False):
        super(RSNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.ResilientSSLs = nn.ModuleList([
            MultiScaleResilientSSL(hidden_dim, heads, dt_rank, double_v_dim).to(DEVICE.get_device())
            for _ in range(layers)
        ]).to(DEVICE.get_device())

        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ]).to(DEVICE.get_device())

        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ]).to(DEVICE.get_device())

    def forward(self, *args, **kwargs):
        return self.forward_chunkwise(*args, **kwargs)

    def forward_chunkwise(self, x, chunkwise_size):

        x = x.to(DEVICE.get_device())

        layer_r_s = []

        for j in range(self.layers):
            o, layer_r_i = self.ResilientSSLs[j].forward_chunkwise(self.layer_norms_1[j](x), chunkwise_size)

            y = o + x

            layer_r_s.append(layer_r_i)

            x = self.layer_norms_2[j](y) + y

        return x, layer_r_s

    def forward_parallel(self, x):
        x = x.to(DEVICE.get_device())

        for i in range(self.layers):
            Y = self.ResilientSSLs[i].forward_parallel(self.layer_norms_1[i](x)) + x

            x = self.ffns[i](self.layer_norms_2[i](Y).reshape(-1, Y.shape[-1]),
                             update_grid=DEVICE.judg_epoch()).reshape(Y.shape) + Y

        return x

    def forward_recurrent(self):
        return 0
