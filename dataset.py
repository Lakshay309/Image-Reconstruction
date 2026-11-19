
"""
dataset.py

Loads user-provided images and prepares them for reconstruction.

Output:
    PyTorch float32 tensor
    Shape: (N, C, size, size)
    Pixel range: [0, 1]

Supported formats:
    PNG, JPG, JPEG, BMP, TIFF, WEBP
"""

import os

import numpy as np
import torch

from skimage import color, io, transform
from skimage.util import img_as_float32


# 1. IMAGE CONFIGURATION

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# 2. IMAGE PREPROCESSING

def _prep(img: np.ndarray, size: int, gray: bool) -> np.ndarray:
    """
    Convert an image to float32 HWC format in [0, 1]
    and resize it to (size, size).
    """

    img = np.asarray(img)

    if img.size == 0:
        raise ValueError("Image is empty.")

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image dimensions: {img.shape}"
        )

    # Convert integer pixel values to float32 [0, 1].
    img = img_as_float32(img)

    # Grayscale: H x W -> H x W x 1
    if img.ndim == 2:
        img = img[..., None]

    channels = img.shape[2]

    # Remove alpha channel.
    if channels == 4:
        img = img[..., :3]

    elif channels not in (1, 3):
        raise ValueError(
            f"Unsupported number of image channels: {channels}"
        )

    # Convert RGB to grayscale.
    if gray and img.shape[2] == 3:
        img = color.rgb2gray(img)[..., None]

    # Convert grayscale to RGB.
    elif not gray and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # Resize while preserving the channel dimension.
    if img.shape[:2] != (size, size):

        img = transform.resize(
            img,
            (size, size, img.shape[2]),
            anti_aliasing=True,
            preserve_range=True,
        )

    return np.clip(img, 0.0, 1.0).astype(np.float32)



# 3. CONVERT IMAGES TO PYTORCH TENSOR

def _to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of HWC images into a BCHW tensor.
    """

    if not images:
        raise ValueError("No images available for tensor conversion.")

    stacked = np.stack(images, axis=0)

    return torch.from_numpy(
        stacked
    ).permute(0, 3, 1, 2).contiguous()



# 4. MAKE IMAGES SQUARE

def _square(img: np.ndarray, fit: str) -> np.ndarray:
    """
    Make an image square without distorting its aspect ratio.

    crop:
        Center-crop the image. Some borders may be lost.

    pad:
        Add edge pixels to preserve the complete image.
    """

    if fit not in ("crop", "pad"):
        raise ValueError(
            "fit must be either 'crop' or 'pad'."
        )

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image dimensions: {img.shape}"
        )

    h, w = img.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Image has invalid dimensions.")

    if h == w:
        return img

    if fit == "crop":

        side = min(h, w)

        top = (h - side) // 2
        left = (w - side) // 2

        return img[
            top:top + side,
            left:left + side
        ]

    # Edge padding
    side = max(h, w)

    pad_h = side - h
    pad_w = side - w

    padding = [
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    ]

    if img.ndim == 3:
        padding.append((0, 0))

    return np.pad(
        img,
        padding,
        mode="edge",
    )


# 5. LOAD USER IMAGES

def load_user_images(
    path: str,
    n: int | None = None,
    size: int = 64,
    gray: bool = True,
    fit: str = "crop",
):
    """
    Load one image or multiple images from a directory.

    Parameters:
        path:
            Image file or directory containing images.

        n:
            Maximum number of images to load.
            None means load all supported images.

        size:
            Output image dimensions (size x size).

        gray:
            True for grayscale, False for RGB.

        fit:
            'crop' or 'pad'.

    Returns:
        images:
            Float32 tensor of shape (N, C, size, size).

        names:
            List of image filenames without extensions.
    """

    if size <= 0:
        raise ValueError("Image size must be positive.")

    if n is not None and n <= 0:
        raise ValueError("Number of images must be positive.")

    if fit not in ("crop", "pad"):
        raise ValueError("fit must be either 'crop' or 'pad'.")

    path = os.path.abspath(path)

    # Find input files.
    if os.path.isdir(path):

        files = sorted(
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if os.path.splitext(filename)[1].lower()
            in SUPPORTED_EXTENSIONS
        )

    elif os.path.isfile(path):

        extension = os.path.splitext(path)[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}"
            )

        files = [path]

    else:
        raise FileNotFoundError(
            f"Image path not found: {path}"
        )

    if not files:
        raise FileNotFoundError(
            f"No supported images found in: {path}"
        )

    # Limit the number of images if requested.
    if n is not None:
        files = files[:n]

    images = []
    names = []

    # Load and preprocess each image.
    for filename in files:

        try:
            img = io.imread(filename)

            img = _square(img, fit)

            img = _prep(
                img,
                size=size,
                gray=gray,
            )

            images.append(img)

            names.append(
                os.path.splitext(
                    os.path.basename(filename)
                )[0]
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to process image '{filename}': {error}"
            ) from error

    return _to_tensor(images), names
