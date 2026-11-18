"""models.py -- lightweight U-Net used as a Deep Image Prior (DIP) generator.

x_theta = f_theta(z), z = fixed random code.  The *architecture* is the prior: convolutional
nets fit smooth structure before noise (spectral bias), so early stopping / TV regularisation
yields a clean image estimate without any training data.
Input spatial size must be divisible by 2**depth (64 and 128 are fine for depth<=4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    """(reflect-pad, conv3x3, BN, LeakyReLU) x 2."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        
        self.net = nn.Sequential(
            # First convolution
            nn.ReflectionPad2d(1), 
            nn.Conv2d(cin, cout, 3), 
            nn.BatchNorm2d(cout), 
            nn.LeakyReLU(0.2, True),
        
            # second convolution
            nn.ReflectionPad2d(1), 
            nn.Conv2d(cout, cout, 3), 
            nn.BatchNorm2d(cout), 
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):

    def __init__(
        self,
        in_ch=8,
        out_ch=1,
        base=16,
        depth=3
    ):
        super().__init__()

        self.depth = depth
        ch = [base * 2 ** i for i in range(depth + 1)] # e.g. [16, 32, 64, 128]
        
        self.enc = nn.ModuleList(
            [ _Block(in_ch, ch[0])] +
            [
                _Block(ch[i - 1], ch[i])
                for i in range(1, depth + 1)
            ]
        )
        
        self.dec = nn.ModuleList(
            [
                _Block(ch[i] + ch[i - 1], ch[i - 1])
                for i in range(depth, 0, -1)
            ]
        )
        
        self.pool = nn.AvgPool2d(2)
        self.out = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.shape[-1] % 2 ** self.depth == 0 and z.shape[-2] % 2 ** self.depth == 0, \
            "H and W must be divisible by 2**depth"
            
        skips = []
        h = z
        
        for i, blk in enumerate(self.enc):
            h = blk(h)
            
            if i < self.depth:
                skips.append(h)
                h = self.pool(h)
        
        for blk in self.dec:
            
            h = F.interpolate(
                h,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
            
            h = blk(
                torch.cat(
                    [h, skips.pop()], 
                    dim=1
                )
            )
        return torch.sigmoid(self.out(h)) # image in (0, 1)
