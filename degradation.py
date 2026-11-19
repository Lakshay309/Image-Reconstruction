"""
degradation.py

Forward model:
    y = M * (x + eta)

x:
    Original image, shape (B, C, H, W), values in [0, 1].

M:
    Binary sampling mask, shape (B, 1, H, W).
    1 = observed, 0 = missing.
    Shared across image channels.

eta:
    Gaussian noise with standard deviation sigma.

Default degradation:
    Missing ratio = 0.20
    Noise sigma   = 0.01
    Mask type     = random
    Block ratio   = 0.02
"""

from __future__ import annotations

import torch


# 1. AUTOMATIC OR EXPLICIT BLOCK SIZE

def _get_block_size(
    height: int,
    width: int,
    block_ratio: float,
    block_size: int | None = None,
) -> int:
    """
    Calculate block dimensions relative to image dimensions,
    unless an explicit block_size is provided.
    """

    if block_size is not None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2.")

        return min(block_size, height, width)

    if not 0 < block_ratio <= 1:
        raise ValueError("block_ratio must be between 0 and 1.")

    block_size = max(
        2,
        round(min(height, width) * block_ratio),
    )

    return min(block_size, height, width)


# 2. RANDOM MISSING PIXELS

def _add_random_missing(
    missing: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> None:
    """
    Randomly mark exactly 'count' additional pixels as missing.
    Existing missing pixels are preserved.
    """

    available = torch.where(~missing.flatten())[0]

    count = min(count, len(available))

    if count == 0:
        return

    selected = torch.randperm(
        len(available),
        generator=generator,
    )[:count]

    missing.view(-1)[available[selected]] = True


# 3. BLOCK MISSING PIXELS

def _add_block_missing(
    missing: torch.Tensor,
    count: int,
    block_size: int,
    generator: torch.Generator,
) -> None:
    """
    Add missing pixels in small square regions.

    Existing missing pixels are not counted twice.
    The requested number of missing pixels is never exceeded.
    """

    height, width = missing.shape

    remaining = count
    attempts = 0

    max_attempts = max(100, height * width * 2)

    while remaining > 0 and attempts < max_attempts:

        top = int(
            torch.randint(
                0,
                height - block_size + 1,
                (1,),
                generator=generator,
            ).item()
        )

        left = int(
            torch.randint(
                0,
                width - block_size + 1,
                (1,),
                generator=generator,
            ).item()
        )

        region = missing[
            top:top + block_size,
            left:left + block_size,
        ]

        available_rows, available_cols = torch.where(~region)
        available_count = len(available_rows)

        if available_count > 0:

            selected_count = min(
                remaining,
                available_count,
            )

            selected = torch.randperm(
                available_count,
                generator=generator,
            )[:selected_count]

            # Coordinate indexing works for non-contiguous region views.
            region[
                available_rows[selected],
                available_cols[selected],
            ] = True

            remaining -= selected_count

        attempts += 1

    # Fill any remaining pixels randomly.
    # This guarantees the requested missing count.
    if remaining > 0:
        _add_random_missing(
            missing,
            remaining,
            generator,
        )


# 4. CREATE SAMPLING MASK

def make_mask(
    shape,
    missing_ratio: float = 0.20,
    kind: str = "random",
    block_ratio: float = 0.02,
    generator: torch.Generator | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """
    Generate a binary sampling mask.

    Supported mask types:
        random:
            Independently selected random missing pixels.

        block:
            Missing pixels arranged in small square regions.

        mixed:
            Half the missing-pixel budget randomly distributed
            and half arranged in blocks.

    block_size:
        Explicit square block side length in pixels.
        If None, block_ratio determines the size.

    Returns:
        Tensor of shape (B, 1, H, W).
    """

    if len(shape) != 4:
        raise ValueError("Expected image shape (B, C, H, W).")

    batch_size, _, height, width = shape

    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("Batch size and image dimensions must be positive.")

    if not 0.0 <= missing_ratio <= 0.9:
        raise ValueError("missing_ratio must be between 0 and 0.9.")

    if kind not in ("random", "block", "mixed"):
        raise ValueError(
            f"Unknown mask kind '{kind}'. Use random, block or mixed."
        )

    if generator is None:
        generator = torch.Generator().manual_seed(0)

    total_pixels = height * width

    # Exact missing-pixel count per image.
    missing_count = round(total_pixels * missing_ratio)

    actual_block_size = _get_block_size(
        height,
        width,
        block_ratio,
        block_size,
    )

    masks = []

    for _ in range(batch_size):

        missing = torch.zeros(
            height,
            width,
            dtype=torch.bool,
        )

        if kind == "random":

            _add_random_missing(
                missing,
                missing_count,
                generator,
            )

        elif kind == "block":

            _add_block_missing(
                missing,
                missing_count,
                actual_block_size,
                generator,
            )

        elif kind == "mixed":

            random_count = missing_count // 2
            block_count = missing_count - random_count

            _add_random_missing(
                missing,
                random_count,
                generator,
            )

            _add_block_missing(
                missing,
                block_count,
                actual_block_size,
                generator,
            )

        masks.append((~missing).float())

    return torch.stack(masks, dim=0).unsqueeze(1)


# 5. DEGRADE IMAGES

def degrade(
    x: torch.Tensor,
    missing_ratio: float = 0.20,
    sigma: float = 0.01,
    kind: str = "random",
    seed: int = 0,
    block_ratio: float = 0.02,
    block_size: int | None = None,
):
    """
    Degrade images by removing pixels and adding Gaussian noise.

    Args:
        x:
            Original images (B, C, H, W).

        missing_ratio:
            Fraction of pixels to remove.

        sigma:
            Standard deviation of Gaussian noise.

        kind:
            random, block or mixed.

        seed:
            Random seed for reproducibility.

        block_ratio:
            Automatic block side length relative to image dimensions.

        block_size:
            Explicit square block side length in pixels.
            Overrides automatic block sizing when provided.

    Returns:
        y:
            Degraded observations.

        M:
            Binary sampling mask.
    """

    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("x must be a tensor of shape (B, C, H, W).")

    if not x.is_floating_point():
        raise TypeError("Input images must have a floating-point dtype.")

    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    # Generate random values on CPU for reproducibility.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    M = make_mask(
        x.shape,
        missing_ratio=missing_ratio,
        kind=kind,
        block_ratio=block_ratio,
        generator=generator,
        block_size=block_size,
    )

    M = M.to(
        device=x.device,
        dtype=x.dtype,
    )

    noise = torch.randn(
        x.shape,
        generator=generator,
        dtype=torch.float32,
    ).to(
        device=x.device,
        dtype=x.dtype,
    )

    eta = sigma * noise

    # Missing pixels are exactly zero.
    y = M * (x + eta)

    return y, M
