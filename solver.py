
"""solver.py -- variational solvers for   min_x  1/2 ||M (.) x - y||^2 + R(x).

Fixed TV (FTV)       R(x) = lam * sum_ij ||grad x_ij||_2
Adaptive TV (ATV)    R(x) = sum_ij phi(||grad x_ij||_2),  phi(t) = lam0*kappa*log(1 + t/kappa)
                    solved by Majorize-Minimize = iteratively reweighted TV with weights
                    lam_ij^{k} = phi'(t_ij^{k}) = lam0 / (1 + ||grad x^{k}_ij|| / kappa).
Hybrid (ATV + DIP)   same penalty, but x = f_theta(z) is a U-Net output optimised with Adam.

All TV-type problems are solved by Chambolle-Pock primal-dual (PDHG); see README.md.
Tensors are (B, C, H, W); gradients are (B, C, 2, H, W); TV is *vectorial* (the 2-norm runs
over channels and both directions) so colour images do not bleed across channels.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from models import SmallUNet


# --------------------------------------------------------------------------------------
# Discrete operators: forward differences with Neumann boundary, and div = -grad^T
# --------------------------------------------------------------------------------------
def grad(x: torch.Tensor) -> torch.Tensor:
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.stack((gy, gx), dim=2)                                    # (B,C,2,H,W)


def div(p: torch.Tensor) -> torch.Tensor:
    """Negative adjoint of grad:  <grad x, p> = -<x, div p>."""
    qy, qx = p[:, :, 0, :-1, :], p[:, :, 1, :, :-1]
    dx = F.pad(qx, (0, 1, 0, 0)) - F.pad(qx, (1, 0, 0, 0))
    dy = F.pad(qy, (0, 0, 0, 1)) - F.pad(qy, (0, 0, 1, 0))
    return dx + dy


def grad_mag(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Pixelwise ||grad x_ij||_2 (over channels and directions) -> (B,1,H,W)."""
    return torch.sqrt((grad(x) ** 2).sum(dim=(1, 2)) + eps ** 2).unsqueeze(1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=C)


# --------------------------------------------------------------------------------------
# Initialisation: fill missing pixels with the nearest observed value (scipy EDT)
# --------------------------------------------------------------------------------------
def nearest_fill(y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    yn, Mn = y.detach().cpu().numpy(), M.detach().cpu().numpy()
    out = yn.copy()
    for b in range(yn.shape[0]):
        missing = Mn[b, 0] < 0.5
        if not missing.any() or missing.all():
            continue
        idx = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
        for c in range(yn.shape[1]):
            out[b, c] = yn[b, c][idx[0], idx[1]]
    return torch.from_numpy(out).to(y)


# --------------------------------------------------------------------------------------
# Objectives (used for monitoring / proving monotone decrease)
# --------------------------------------------------------------------------------------
def data_term(x, y, M):
    return 0.5 * ((M * x - y) ** 2).sum(dim=(1, 2, 3))


def ftv_objective(x, y, M, lam):
    return data_term(x, y, M) + lam * grad_mag(x).sum(dim=(1, 2, 3))


def adaptive_objective(x, y, M, lam0, kappa):
    t = grad_mag(x)
    return data_term(x, y, M) + (lam0 * kappa * torch.log1p(t / kappa)).sum(dim=(1, 2, 3))


# --------------------------------------------------------------------------------------
# Chambolle-Pock primal-dual for  min_x 1/2||Mx-y||^2 + sum_ij lam_ij ||(grad x)_ij||_2
# --------------------------------------------------------------------------------------
@torch.no_grad()
def pdhg_tv(y, M, lam, n_iter: int = 300, x0: torch.Tensor | None = None) -> torch.Tensor:
    """lam: float or tensor (B,1,H,W) of per-pixel weights."""
    tau = sigma = 0.99 / math.sqrt(8.0)               # tau*sigma*||grad||^2 < 1, ||grad||^2 <= 8
    lam_t = torch.as_tensor(lam, dtype=y.dtype, device=y.device)
    lam_t = lam_t.clamp_min(1e-12) if lam_t.dim() == 0 else lam_t.clamp_min(1e-12).unsqueeze(2)  # (B,1,1,H,W)

    x = nearest_fill(y, M) if x0 is None else x0.clone()
    x_bar = x.clone()
    p = torch.zeros_like(grad(x))
    for _ in range(n_iter):
        # dual ascent + projection onto the ball {||p_ij|| <= lam_ij}  (prox of conjugate of weighted TV)
        p = p + sigma * grad(x_bar)
        nrm = torch.sqrt((p ** 2).sum(dim=(1, 2), keepdim=True))
        p = p / torch.clamp(nrm / lam_t, min=1.0)
        # primal descent: x <- prox_{tau G}(x + tau div p), closed form for the masked quadratic
        x_old = x
        v = x + tau * div(p)
        x = (v + tau * M * y) / (1.0 + tau * M)
        x_bar = 2.0 * x - x_old                       # over-relaxation theta = 1
    return x


# --------------------------------------------------------------------------------------
# Baseline: fixed TV
# --------------------------------------------------------------------------------------
def fixed_tv(y, M, lam: float, n_iter: int = 400) -> torch.Tensor:
    return pdhg_tv(y, M, lam, n_iter).clamp(0, 1)


# --------------------------------------------------------------------------------------
# Spatially adaptive TV  (iteratively reweighted TV / MM)
# --------------------------------------------------------------------------------------
def adaptive_weights(x, lam0: float, kappa: float, grad_sigma: float = 0.0) -> torch.Tensor:
    """lam_ij = lam0 / (1 + ||grad x_ij|| / kappa)  -> (B,1,H,W).
    Large in flat regions (strong smoothing), small on edges (weak smoothing)."""
    return lam0 / (1.0 + grad_mag(gaussian_blur(x, grad_sigma)) / kappa)


@torch.no_grad()
def adaptive_tv(y, M, lam0: float, kappa: float, n_outer: int = 6, n_inner: int = 150,
                grad_sigma: float = 0.0, pilot_lam: float | None = None, pilot_iters: int = 300):
    """Returns (x_hat, final_weights, history of the nonconvex objective per outer iteration)."""
    x = pdhg_tv(y, M, lam0 if pilot_lam is None else pilot_lam, pilot_iters)   # pilot estimate
    hist = []
    for _ in range(n_outer):
        W = adaptive_weights(x, lam0, kappa, grad_sigma)                       # MM surrogate weights
        x = pdhg_tv(y, M, W, n_inner, x0=x)                                    # solve weighted-TV problem
        hist.append(adaptive_objective(x, y, M, lam0, kappa).mean().item())
    W = adaptive_weights(x, lam0, kappa, grad_sigma)
    return x.clamp(0, 1), W, hist


# --------------------------------------------------------------------------------------
# Hybrid: Deep Image Prior generator + adaptive-TV penalty (Adam on network weights)
# --------------------------------------------------------------------------------------
def _dip_single(y, M, x_pilot, lam0, kappa, n_iter, lr, reweight_every, ema, grad_sigma,
                z_ch, base, noise_std, dev, seed):
    torch.manual_seed(seed)
    y, M, x_pilot = y.to(dev), M.to(dev), x_pilot.to(dev)
    _, C, H, W_ = y.shape
    net = SmallUNet(z_ch, C, base).to(dev)
    z = 0.1 * torch.rand(1, z_ch, H, W_, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    W = adaptive_weights(x_pilot, lam0, kappa, grad_sigma)
    avg = None
    for it in range(n_iter):
        opt.zero_grad()
        out = net(z + noise_std * torch.randn_like(z))                         # input jitter regularises
        loss = data_term(out, y, M).sum() + (W * grad_mag(out, eps=1e-3)).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            o = out.detach()
            avg = o if avg is None else ema * avg + (1 - ema) * o               # exp. moving average output
            if (it + 1) % reweight_every == 0:
                W = adaptive_weights(avg, lam0, kappa, grad_sigma)              # refresh lam_ij from current estimate
    return avg.clamp(0, 1).cpu()


def dip_adaptive_tv(y, M, x_pilot, lam0, kappa, n_iter: int = 800, lr: float = 1e-2,
                    reweight_every: int = 100, ema: float = 0.95, grad_sigma: float = 0.0,
                    z_ch: int = 8, base: int = 16, noise_std: float = 1 / 30.0,
                    device: str | torch.device | None = None, seed: int = 0) -> torch.Tensor:
    """One fresh network per image (DIP is per-image, no training set).

    ema controls exponential moving average of network outputs:
    0.95 = strong averaging, 0.80 = less averaging, 0.50 = more responsive,
    0.00 = use the latest network output without averaging.
    """
    if not 0.0 <= ema < 1.0:
        raise ValueError("ema must be in the range [0.0, 1.0).")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outs = [_dip_single(y[b:b + 1], M[b:b + 1], x_pilot[b:b + 1], lam0, kappa, n_iter, lr,
                        reweight_every, ema, grad_sigma, z_ch, base, noise_std, dev, seed + b)
            for b in range(y.shape[0])]
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------------------
# Tiny self-test (instant):  python solver.py
# --------------------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    x, p = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 2, 16, 16)
    lhs, rhs = (grad(x) * p).sum(), -(x * div(p)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-3), (lhs.item(), rhs.item())
    M = (torch.rand(1, 1, 16, 16) > 0.5).float()
    y = M * torch.rand(1, 1, 16, 16)
    xf = fixed_tv(y, M, 0.1, n_iter=10)
    xa, W, hist = adaptive_tv(y, M, 0.1, 0.05, n_outer=2, n_inner=5, pilot_iters=5)
    assert xf.shape == xa.shape == y.shape and W.shape == y.shape
    print("solver self-test passed: grad/div adjoint OK, FTV/ATV shapes OK, hist =", hist)


if __name__ == "__main__":
    _self_test()
