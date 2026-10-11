
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_bits(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    return np.fromiter(
        (int(ch) for ch in text if ch in {"0", "1"}),
        dtype=np.uint8,
    )


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    tst = test.astype(np.float64)
    mse = float(np.mean((ref - tst) ** 2))
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    tst_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    return float(
        structural_similarity(
            ref_rgb,
            tst_rgb,
            channel_axis=2,
            data_range=255,
        )
    )


def ber(original: np.ndarray, extracted: np.ndarray) -> float:
    denominator = max(len(original), len(extracted))
    if denominator == 0:
        raise ValueError("Sequences are empty")

    common = min(len(original), len(extracted))
    errors = int(np.count_nonzero(original[:common] != extracted[:common]))
    errors += abs(len(original) - len(extracted))
    return float(errors / denominator)
