"""degradation.py -- forward model  y = M (.) x + eta.

x   : clean image                       (B, C, H, W) in [0, 1]
M   : binary sampling mask, 1=observed  (B, 1, H, W)   (shared across channels)
eta : i.i.d. Gaussian noise N(0, sigma^2)

Implementation note: we generate  y = M * (x + sigma * randn).  The data term of the
inverse problem only ever touches the observed entries (M=1), so this is equivalent to
y = M (.) x + eta on the observed set, and y is exactly 0 on the missing set.
"""
from __future__ import annotations

import torch


def make_mask(shape, missing_ratio: float = 0.5, kind: str = "random",
              block_size: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
    """Binary mask (B,1,H,W). kind: 'random' (i.i.d. pixels) | 'block' (square holes) | 'mixed'."""
    B, _, H, W = shape
    assert 0.0 <= missing_ratio <= 0.9, "missing_ratio must be in [0, 0.9]"
    if kind == "random":
        return (torch.rand(B, 1, H, W, generator=generator) >= missing_ratio).float()
    if kind == "block":
        M = torch.ones(B, 1, H, W)
        for b in range(B):
            while 1.0 - M[b].mean().item() < missing_ratio:
                i = int(torch.randint(0, H - block_size + 1, (1,), generator=generator))
                j = int(torch.randint(0, W - block_size + 1, (1,), generator=generator))
                M[b, :, i:i + block_size, j:j + block_size] = 0.0
        return M
    if kind == "mixed":                          # half the budget random pixels, half blocks
        a = make_mask(shape, missing_ratio / 2, "random", block_size, generator)
        b = make_mask(shape, missing_ratio / 2, "block", block_size, generator)
        return a * b
    raise ValueError(f"unknown mask kind '{kind}'")


def degrade(x: torch.Tensor, missing_ratio: float = 0.5, sigma: float = 0.05,
            kind: str = "random", block_size: int = 8, seed: int = 0):
    """Return (y, M). Deterministic given `seed`."""
    g = torch.Generator().manual_seed(seed)
    M = make_mask(x.shape, missing_ratio, kind, block_size, g)
    eta = sigma * torch.randn(x.shape, generator=g)
    y = M * (x + eta)
    return y, M
