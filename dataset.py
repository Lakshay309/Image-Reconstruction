"""dataset.py -- small-image loaders (CIFAR-10, STL-10, scikit-image crops).

Everything is returned as a float32 tensor of shape (N, C, size, size) in [0, 1],
so that experiments run in seconds on a laptop CPU.

Sources
-------
cifar10 : CIFAR-10 test split (32x32, bicubic/anti-aliased resize to `size`).
stl10   : STL-10 test split (96x96 -> `size`; keeps more fine detail than CIFAR).
skimage : crops of bundled scikit-image photos (no download needed; richest textures).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from skimage import color, data as skdata, transform


def _prep(img: np.ndarray, size: int, gray: bool) -> np.ndarray:
    """Any HxW / HxWx{1,3,4} image -> float32 HxWxC in [0,1], resized to size x size."""
    img = np.asarray(img)
    scale = 255.0 if img.dtype == np.uint8 else 65535.0 if img.dtype == np.uint16 else 1.0
    img = img.astype(np.float32) / scale
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 4:                      # drop alpha
        img = img[..., :3]
    if gray and img.shape[2] == 3:
        img = color.rgb2gray(img)[..., None]
    if (not gray) and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[0] != size or img.shape[1] != size:
        img = transform.resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _to_tensor(arrs: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrs, 0)).permute(0, 3, 1, 2).contiguous()


def _skimage_images(n: int, size: int, gray: bool) -> torch.Tensor:
    """Center-square crops, then quadrant crops, of several bundled skimage photos."""
    names = ["camera", "astronaut", "chelsea", "coffee", "coins", "moon",
            "rocket", "brick", "grass", "gravel", "horse", "page"]
    full, quads = [], []
    for name in names:
        try:
            img = np.asarray(getattr(skdata, name)())
        except Exception:                       # some images need a download; just skip them
            continue
        h, w = img.shape[:2]
        s = min(h, w)
        r0, c0 = (h - s) // 2, (w - s) // 2
        sq = img[r0:r0 + s, c0:c0 + s]
        full.append(sq)
        q = s // 2
        quads += [sq[:q, :q], sq[:q, q:2 * q], sq[q:2 * q, :q], sq[q:2 * q, q:2 * q]]
    pool = full + quads
    if not pool:
        raise RuntimeError("No scikit-image sample images could be loaded.")
    return _to_tensor([_prep(p, size, gray) for p in pool[:n]])


def load_images(name: str = "cifar10", n: int = 8, size: int = 64, gray: bool = True,
                root: str = "./data") -> torch.Tensor:
    """Return the first `n` images of the chosen source as (n, C, size, size)."""
    name = name.lower()
    if name == "skimage":
        return _skimage_images(n, size, gray)
    if name == "cifar10":
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=root, train=False, download=True)
        raw = [ds.data[i] for i in range(n)]                      # uint8 HxWx3
    elif name == "stl10":
        from torchvision.datasets import STL10
        ds = STL10(root=root, split="test", download=True)
        raw = [np.transpose(ds.data[i], (1, 2, 0)) for i in range(n)]  # CHW -> HWC
    else:
        raise ValueError(f"unknown dataset '{name}' (use cifar10 | stl10 | skimage)")
    return _to_tensor([_prep(r, size, gray) for r in raw])


# --------------------------------------------------------------------------------------
# Your own images:  a single file or a folder  ->  square, size x size, in [0, 1]
# --------------------------------------------------------------------------------------
_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _square(img: np.ndarray, fit: str) -> np.ndarray:
    """Make an image square. fit='crop': centre crop (no distortion, loses borders).
    fit='pad': edge-pad to a square (keeps the whole picture)."""
    h, w = img.shape[:2]
    if fit == "crop":
        s = min(h, w)
        r0, c0 = (h - s) // 2, (w - s) // 2
        return img[r0:r0 + s, c0:c0 + s]
    s = max(h, w)
    ph, pw = s - h, s - w
    pad = [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="edge")


def load_user_images(path: str, n: int | None = None, size: int = 64, gray: bool = True,
                    fit: str = "crop"):
    """Load one image file, or every image in a folder (sorted by name, at most `n`).
    Each is made square (`fit`), resized to size x size, converted to gray/RGB in [0,1].
    Returns (tensor (N,C,size,size), list of file stems)."""
    from skimage import io
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                        if os.path.splitext(f)[1].lower() in _EXTS)
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"image path not found: {path}")
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(sorted(_EXTS))}) found in {path}")
    if n:
        files = files[:n]
    arrs, names = [], []
    for f in files:
        arrs.append(_prep(_square(io.imread(f), fit), size, gray))
        names.append(os.path.splitext(os.path.basename(f))[0])
    return _to_tensor(arrs), names
