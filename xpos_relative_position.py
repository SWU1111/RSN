import torch
import torch.nn as nn


def fixed_pos_embedding(x, delta, batch_size, offset):
    seq_len, dim = x.shape

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim).to(DEVICE.get_device()) / dim))

    sinusoid_inp = (
            torch.einsum("i , j -> i j",
                         torch.arange(offset, seq_len + offset, dtype=torch.float).to(DEVICE.get_device()),
                         inv_freq).unsqueeze(0).expand(batch_size, -1, -1) * delta.view(batch_size, seq_len, dim,
                                                                                        2).mean(dim=-1)
    ).to(DEVICE.get_device())

    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = torch.stack((-x2, x1), dim=-1).to(DEVICE.get_device())

    return x.flatten(-2)


def duplicate_interleave(m):
    dim0 = m.shape[0]
    dim1 = m.shape[1]

    m = m.view(dim0, -1, 1)

    m = m.repeat(1, 1, 2)

    m = m.view(dim0, dim1, -1)

    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))

    return (x * cos[:, :, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]]


class XPOS(nn.Module):
    def __init__(
            self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim

        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2).to(DEVICE.get_device()) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, delta, offset=0, downscale=False):

        batch_size, length, _ = x.shape

        min_pos = offset
        max_pos = length + offset

        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]

        sin, cos = fixed_pos_embedding(scale, delta, batch_size, offset)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(DEVICE.get_device()).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x
