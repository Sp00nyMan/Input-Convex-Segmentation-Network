import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupling_layer import CouplingLayer


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """

    def __init__(self, in_channels=2, mid_channels=32):
        super(RealNVP, self).__init__()

        self.flows = _RealNVP(in_channels, mid_channels)

    def forward(self, x):
        # Expect inputs in [0, 1]
        if x.min() < 0 or x.max() > 1:
            raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                             .format(x.min(), x.max()))

        x, sldj = self.flows(x)

        return x, sldj

    def backward(self, z):
        x, sldj = self.flows.backward(z)

        return x, sldj


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """

    def __init__(self, in_channels, mid_channels):
        super(_RealNVP, self).__init__()

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, reverse_mask=i % 2) for i in range(5)
        ])

    def forward(self, x):
        for coupling in self.in_couplings:
            x, sldj = coupling(x)

        return x, sldj

    def backward(self, z):
        for coupling in reversed(self.in_couplings):
            z, sldj = coupling.backward(z)
        return z, sldj
