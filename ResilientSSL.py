import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResilientSSL(nn.Module):

    def __init__(self, hidden_size, gamma, head_size=None, dt_rank=None, double_v_dim=False):

        super(SimpleResilientSSL, self).__init__()
        self.hidden_size = hidden_size

        if head_size is None:
            self.head_size = self.hidden_size
        else:
            self.head_size = head_size

        if dt_rank is None:
            self.dt_rank = self.head_size
        else:
            self.dt_rank = dt_rank

        self.proj = KAR([self.hidden_size, self.dt_rank + 2 * self.hidden_size],
                        grid_size=10,
                        spline_order=3,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        base_activation=nn.Identity,
                        grid_eps=0.2,
                        grid_range=[-3, 3],
                        ).to(DEVICE.get_device())

        self.v_dim = self.head_size * 2 if double_v_dim else self.head_size

        self.W_V = nn.Parameter(torch.randn(self.hidden_size, self.v_dim) / self.dt_rank).to(DEVICE.get_device())

        nn.init.xavier_uniform_(self.W_V)

        self.W_V.data = self.W_V.data / self.v_dim

        self.gamma = gamma

        self.xpos = XPOS(self.dt_rank)

    def discretize(self, delta, Q, K):

        Q_bar = torch.einsum('bln,bld->blnd', Q, delta)
        K_bar = torch.einsum('bln,bld->blnd', K, delta)
        return Q_bar, K_bar

    def selective_scan(self, x):
        batch_size, seq_len, hid_size = x.shape

        proj_out = self.proj(x.reshape(-1, hid_size), update_grid=DEVICE.judg_epoch()).reshape(batch_size, seq_len, -1)
        delta, Q, K = torch.split(proj_out, [self.dt_rank, self.hidden_size, self.hidden_size], dim=-1)

        delta = F.softplus(delta).to(DEVICE.get_device())

        Q_bar, K_bar = self.discretize(delta, Q, K)

        return delta, Q_bar, K_bar

    def product(self, x, y):

        x = x.unsqueeze(2)
        return (x @ y).squeeze(2)

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1).to(DEVICE.get_device())

        m = torch.arange(sequence_length).unsqueeze(0).to(DEVICE.get_device())

        D = (self.gamma ** (n - m)) * (n >= m).float()

        D[D != D] = 0

        return D

    def _get_D2(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)

        m = torch.arange(sequence_length).unsqueeze(0)

        D = (self.gamma ** (m - n)) * (m > n).float()

        D[D != D] = 0

        return D

    def forward(self, x_i, r_i_1, i):

        x_i = x_i.to(DEVICE.get_device())
        r_i_1 = r_i_1.to(DEVICE.get_device())
        delta, Q_bar, K_bar = self.selective_scan(x_i)

        batch, chunk_size, _ = x_i.shape

        D = self._get_D(chunk_size).to(DEVICE.get_device())

        Qx = self.product(x_i, Q_bar)
        Kx = self.product(x_i, K_bar)

        Q = self.xpos(Qx, delta, i * chunk_size)
        K = self.xpos(Kx, delta, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        e = torch.zeros(batch, chunk_size, 1).to(DEVICE.get_device())

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def forward_chunkwise(self, x, chunkwise_size):

        x = x.to(DEVICE.get_device())

        batch_size, sequence_length, _ = x.shape

        r_n_1 = torch.zeros(self.dt_rank, self.v_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE.get_device())

        r_s = []

        Y_chunkwise = []

        for i in range(sequence_length // chunkwise_size):
            y_i, r_i = self.forward(
                x[:, i * chunkwise_size: (i + 1) * chunkwise_size, :], r_n_1, i
            )
            r_n_1 = r_i

            Y_chunkwise.append(y_i)

            r_s.append(r_i)

        return torch.cat(Y_chunkwise, dim=1), r_s

    def forward_parallel(self, x):
        x = x.to(DEVICE.get_device())

        delta, Q_bar, K_bar = self.selective_scan(x)

        _, sequence_length, _ = x.shape

        D = self._get_D(sequence_length).to(DEVICE.get_device())

        Qx = self.product(x, Q_bar)
        Kx = self.product(x, K_bar)

        Q = self.xpos(Qx, delta)
        K = self.xpos(Kx, delta, downscale=True)

        V = x @ self.W_V
        ret = ((Q @ K.permute(0, 2, 1)) * D.unsqueeze(0))

        return ret @ V

    def forward_recurrent(self):
        return 0

    def forward_b(self, x_i, r_i_1, i):

        x_i = x_i.to(DEVICE.get_device())
        r_i_1 = r_i_1.to(DEVICE.get_device())
        delta, Q_bar, K_bar = self.selective_scan(x_i)

        batch, chunk_size, _ = x_i.shape

        D = self._get_D2(chunk_size).to(DEVICE.get_device())

        Q = self.product(x_i, Q_bar)
        K = self.product(x_i, K_bar)

        Q = self.xpos(Q, delta, i * chunk_size)
        K = self.xpos(K, delta, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        e = torch.zeros(batch, chunk_size, 1).to(DEVICE.get_device())

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def forward_chunkwise_b(self, x, chunkwise_size):

        x = x.to(DEVICE.get_device())

        batch_size, sequence_length, _ = x.shape

        r_n_1 = torch.zeros(self.dt_rank, self.v_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(
            DEVICE.get_device())

        r_s = []

        Y_chunkwise = []

        for i in range(sequence_length // chunkwise_size):
            y_i, r_i = self.forward_b(
                x[:, i * chunkwise_size: (i + 1) * chunkwise_size, :], r_n_1, i
            )
            r_n_1 = r_i

            Y_chunkwise.append(y_i)
            r_s.append(r_i)

        return torch.cat(Y_chunkwise, dim=1), r_s

    def forward_parallel_b(self, x):
        x = x.to(DEVICE.get_device())

        self.delta, self.Q_bar, self.K_bar = self.selective_scan(x)

        _, sequence_length, _ = x.shape

        D = self._get_D(sequence_length).to(DEVICE.get_device())

        D2 = self._get_D2(sequence_length).to(DEVICE.get_device())

        Q = self.product(x, self.Q_bar)
        K = self.product(x, self.K_bar)

        Q = self.xpos(Q, self.delta).to(DEVICE.get_device())
        K = self.xpos(K, self.delta, downscale=True).to(DEVICE.get_device())

        V = x @ self.W_V
        ret = ((Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)) + ((Q @ K.permute(0, 2, 1)) * D2.unsqueeze(0))

        return ret @ V

    def forward_recurrent_b(self):
        return 0


class MultiScaleResilientSSL(nn.Module):
    def __init__(self, hidden_size, heads, dt_rank=None, double_v_dim=False):

        super(MultiScaleResilientSSL, self).__init__()

        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (
                1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)

        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size).to(DEVICE.get_device())

        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size).to(DEVICE.get_device())
        self.group_norm = nn.GroupNorm(heads, self.v_dim).to(DEVICE.get_device())

        self.ResilientSSLs = nn.ModuleList([
            SimpleResilientSSL(self.hidden_size, gamma, self.head_size, dt_rank, double_v_dim).to(DEVICE.get_device()) for
            gamma in
            self.gammas

        ]).to(DEVICE.get_device())

    def forward_chunkwise(self, x, chunkwise_size):

        x = x.to(DEVICE.get_device())

        Y = []
        head_r_is = []

        for j in range(self.heads):
            y, head_r_i = self.ResilientSSLs[j].forward_chunkwise(
                x, chunkwise_size)
            Y.append(y)
            head_r_is.append(head_r_i)

        Y = torch.cat(Y, dim=2)

        r_states = [torch.cat([head_r_is[head][state_idx] for head in range(len(head_r_is))], dim=2) for state_idx in
                    range(len(head_r_is[0]))]

        return Y, r_states

    def forward_parallel(self, x):

        x = x.to(DEVICE.get_device())

        Y = []
        for i in range(self.heads):
            Y.append(self.ResilientSSLs[i].forward_parallel(x))

        Y = torch.cat(Y, dim=2)

        return Y

    def forward_recurrent(self):
        return 0

    def forward_chunkwise_b(self, x, chunkwise_size):

        x = x.to(DEVICE.get_device())

        Y = []
        head_r_is = []

        for j in range(self.heads):
            y, head_r_i = self.ResilientSSLs[j].forward_chunkwise(
                x, chunkwise_size)
            Y.append(y)
            head_r_is.append(head_r_i)

        Y = torch.cat(Y, dim=2)

        r_states = [torch.cat([head_r_is[head][state_idx] for head in range(len(head_r_is))], dim=2) for state_idx in
                    range(len(head_r_is[0]))]

        return Y, r_states

    def forward_parallel_b(self, x):

        x = x.to(DEVICE.get_device())

        Y = []
        for i in range(self.heads):
            Y.append(self.ResilientSSLs[i].forward_parallel_b(x))

        Y = torch.cat(Y, dim=2)

        return Y

    def forward_recurrent_b(self):
        return 0

