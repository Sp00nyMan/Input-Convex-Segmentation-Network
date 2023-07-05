import torch
import torch.nn as nn

from enum import IntEnum


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """

    def __init__(self, in_channels, mid_channels, reverse_mask, device):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        self.st = nn.Sequential(nn.Linear(in_channels//2, mid_channels,
                                          dtype=float, device=device),
                                nn.LeakyReLU(),
                                nn.Linear(mid_channels, mid_channels,
                                          dtype=float, device=device),
                                nn.LeakyReLU(),
                                nn.Linear(mid_channels, 2,
                                          dtype=float, device=device))

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels//2))
        self.rescale.to(device)
        self.st.to(device)

    def forward(self, x):
        x_change, x_id = x[:, ::2], x[:, 1::2]
        if self.reverse_mask:
            x_id, x_change = x_change, x_id

        st = self.st(x_change)
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        # Scale and translate
        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_id = x_id * exp_s + t

        # Add log-determinant of the Jacobian
        sldj = torch.sum(s, dim=1)

        if self.reverse_mask:
            x_id, x_change = x_change, x_id

        x = torch.cat((x_change, x_id), dim=1)

        return x, sldj

    def backward(self, z):
        z_change, z_id = z[:, ::2], z[1:, ::2]

        if self.reverse_mask:
            z_change, z_id = z_id, z_change

        s, t = self.st(z_change)

        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        z_id = (z_id - t) * inv_exp_s

        sldj = torch.sum(-s, dim=1)

        if self.reverse_mask:
            z_change, z_id = z_id, z_change

        x = torch.cat((z_change, z_id), dim=1)

        return x, sldj


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        x = self.weight * x
        return x
