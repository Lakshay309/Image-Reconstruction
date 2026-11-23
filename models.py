"""DIP generator: skip-connected encoder/decoder with fixed, unlearnt upsampling.

This follows the architectural family used by Ulyanov et al. (Deep Image Prior,
CVPR 2018) and Liu et al. (DIP-TV, 2018): a randomly initialized convolutional
generator maps fixed noise to the image; only generator weights are optimized
for the current image. No pretrained weights or external training data are used.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, n_conv: int = 2):
        super().__init__()
        layers = []
        c = cin
        for _ in range(n_conv):
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(c, cout, 3), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]
            c = cout
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """DIP-style skip U-Net; nearest-neighbour upsampling is deliberately unlearnt."""
    def __init__(self, in_ch: int = 16, out_ch: int = 3, base: int = 16,
                 depth: int = 4, skip_ch: int | None = None):
        super().__init__()
        if min(in_ch, out_ch, base, depth) <= 0:
            raise ValueError("in_ch, out_ch, base and depth must be positive")
        self.depth = depth
        skip_ch = max(4, base // 4) if skip_ch is None else int(skip_ch)
        self.enc = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.down = nn.ModuleList()
        channels = [base * min(2 ** i, 8) for i in range(depth + 1)]
        c = in_ch
        for i in range(depth):
            self.enc.append(ConvBlock(c, channels[i]))
            self.skip.append(nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(channels[i], skip_ch, 3),
                nn.BatchNorm2d(skip_ch), nn.LeakyReLU(0.2, inplace=True)))
            self.down.append(nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(channels[i], channels[i + 1], 3, stride=2),
                nn.BatchNorm2d(channels[i + 1]), nn.LeakyReLU(0.2, inplace=True)))
            c = channels[i + 1]
        self.bottleneck = ConvBlock(channels[-1], channels[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = channels[-1]
        for i in reversed(range(depth)):
            self.up.append(nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(c, channels[i], 3),
                nn.BatchNorm2d(channels[i]), nn.LeakyReLU(0.2, inplace=True)))
            self.dec.append(ConvBlock(channels[i] + skip_ch, channels[i]))
            c = channels[i]
        self.head = nn.Sequential(nn.Conv2d(channels[0], out_ch, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        div = 2 ** self.depth
        if z.shape[-2] % div or z.shape[-1] % div:
            raise ValueError(f"Image dimensions must be divisible by 2**depth ({div})")
        skips = []
        h = z
        for enc, skip, down in zip(self.enc, self.skip, self.down):
            h = enc(h)
            skips.append(skip(h))
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up(h)
            h = dec(torch.cat([h, s], dim=1))
        return self.head(h)
