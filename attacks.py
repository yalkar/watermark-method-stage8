
"""
Laboratory image attacks used in the common experiment protocol.

All attacks return an image with the same width, height, dtype and channel
count as the input image, so the extractor receives a consistent shape.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


Attack = Callable[[np.ndarray], np.ndarray]


def attack_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be from 1 to 100")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise RuntimeError("JPEG encoding failed")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def attack_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=image.shape)
    attacked = image.astype(np.float64) + noise
    return np.clip(np.rint(attacked), 0, 255).astype(np.uint8)


def attack_salt_pepper(
    image: np.ndarray,
    amount: float = 0.01,
    salt_ratio: float = 0.5,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Replace `amount` fraction of pixels by black/white values.

    Default: 1% corrupted pixels, half salt and half pepper.
    """
    if not 0 <= amount <= 1:
        raise ValueError("amount must be from 0 to 1")
    if not 0 <= salt_ratio <= 1:
        raise ValueError("salt_ratio must be from 0 to 1")

    attacked = image.copy()
    height, width = image.shape[:2]
    count = int(round(height * width * amount))

    if count == 0:
        return attacked

    indices = rng.choice(height * width, size=count, replace=False)
    rows, cols = np.unravel_index(indices, (height, width))

    salt_count = int(round(count * salt_ratio))
    attacked[rows[:salt_count], cols[:salt_count]] = 255
    attacked[rows[salt_count:], cols[salt_count:]] = 0
    return attacked


def attack_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Blur kernel size must be a positive odd number")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def attack_median(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Median kernel size must be odd and at least 3")
    return cv2.medianBlur(image, kernel_size)


def attack_rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate around the center while keeping the original canvas size.

    Empty regions are filled by reflection to avoid introducing a large
    artificial black border.
    """
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def attack_scale(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale by the requested factor and return to the original dimensions.

    This models resampling damage while keeping the extractor input shape
    compatible with the embedded bit grid.
    """
    if factor <= 0:
        raise ValueError("Scale factor must be positive")

    height, width = image.shape[:2]
    scaled_width = max(1, int(round(width * factor)))
    scaled_height = max(1, int(round(height * factor)))

    first_interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    scaled = cv2.resize(
        image,
        (scaled_width, scaled_height),
        interpolation=first_interpolation,
    )
    return cv2.resize(
        scaled,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def attack_random_crop_10pct(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Remove one random rectangular region containing approximately 10% of
    image pixels and fill it with black.

    The output size is unchanged. The rectangle aspect ratio follows the
    source image, so its width and height are approximately sqrt(0.1) of the
    full dimensions.
    """
    attacked = image.copy()
    height, width = image.shape[:2]

    side_fraction = np.sqrt(0.10)
    crop_height = max(1, int(round(height * side_fraction)))
    crop_width = max(1, int(round(width * side_fraction)))

    max_row = height - crop_height
    max_col = width - crop_width
    row = int(rng.integers(0, max_row + 1))
    col = int(rng.integers(0, max_col + 1))

    attacked[row : row + crop_height, col : col + crop_width] = 0
    return attacked


def build_attacks(seed: int = 20260720) -> list[tuple[str, Attack]]:
    """
    Return the common laboratory attack set.

    Separate deterministic generators are created for stochastic attacks so
    the experiment is reproducible and attack order does not change results.
    """
    return [
        ("no_attack", lambda image: image.copy()),
        ("jpeg_q90", lambda image: attack_jpeg(image, 90)),
        ("jpeg_q70", lambda image: attack_jpeg(image, 70)),
        ("jpeg_q50", lambda image: attack_jpeg(image, 50)),
        (
            "noise_sigma5",
            lambda image: attack_gaussian_noise(
                image,
                5.0,
                rng=np.random.default_rng(seed + 1),
            ),
        ),
        (
            "noise_sigma10",
            lambda image: attack_gaussian_noise(
                image,
                10.0,
                rng=np.random.default_rng(seed + 2),
            ),
        ),
        (
            "salt_pepper_1pct",
            lambda image: attack_salt_pepper(
                image,
                amount=0.01,
                salt_ratio=0.5,
                rng=np.random.default_rng(seed + 3),
            ),
        ),
        ("blur_k3", lambda image: attack_blur(image, 3)),
        ("blur_k5", lambda image: attack_blur(image, 5)),
        ("median_k3", lambda image: attack_median(image, 3)),
        ("median_k5", lambda image: attack_median(image, 5)),
        ("rot_+5deg", lambda image: attack_rotate(image, 5.0)),
        ("rot_-5deg", lambda image: attack_rotate(image, -5.0)),
        ("scale_0.5", lambda image: attack_scale(image, 0.5)),
        ("scale_1.5", lambda image: attack_scale(image, 1.5)),
        (
            "crop_random_10pct",
            lambda image: attack_random_crop_10pct(
                image,
                rng=np.random.default_rng(seed + 4),
            ),
        ),
    ]
