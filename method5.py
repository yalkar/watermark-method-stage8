
"""
Stage 4 — optimized unified implementation.

Changes from Stage 3:
- vectorized 3x3 predictor using NumPy sliding windows;
- optional explicit expected_length for extraction;
- no mandatory embedded 32-bit header;
- full payload can be used for message bits;
- optional metadata JSON stores message length and parameters;
- same project-facing interface.

Examples:
    method5("embed", "image.png")
    method5("embed", "image.png", "watermark.txt")
    method5("extract", "image_watermarked.png")
    method5("extract", "image_watermarked.png", "extracted.txt")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import convolve


BF = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 0, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.float64,
)

G_FILTERS = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [1, 3, 8, 3, 1],
            [0, 0, 0, 0, 0],
            [-1, -3, -8, -3, -1],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 8, 3, 0, 0],
            [1, 3, 0, -3, -1],
            [0, 0, -3, -8, 0],
            [0, 0, -1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 3, 8, 0],
            [-1, -3, 0, 3, 1],
            [0, -8, -3, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        [
            [0, 1, 0, -1, 0],
            [0, 3, 0, -3, 0],
            [0, 8, 0, -8, 0],
            [0, 3, 0, -3, 0],
            [0, 1, 0, -1, 0],
        ],
    ],
    dtype=np.float64,
)

DEFAULT_KEY = "method5-default-key"
DEFAULT_STRENGTH = 1.4
DEFAULT_SIGMA = 0.5


def key_to_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def read_color_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a three-channel raster color image")
    return image


def save_color_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    if path.suffix.lower() not in {".png", ".tif", ".tiff", ".bmp"}:
        raise ValueError("Use PNG, TIFF, or BMP for watermarked output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def read_bits(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    allowed = {"0", "1", " ", "\n", "\r", "\t"}
    invalid = sorted({ch for ch in text if ch not in allowed})
    if invalid:
        raise ValueError(f"Invalid symbols in watermark file: {invalid}")
    bits = np.fromiter((int(ch) for ch in text if ch in {"0", "1"}), dtype=np.uint8)
    if bits.size == 0:
        raise ValueError("Watermark bit sequence is empty")
    return bits


def write_bits(path: str | Path, bits: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(bits.astype(str).tolist()), encoding="utf-8")


def calculate_spatial_jnd(image_bgr: np.ndarray) -> np.ndarray:
    y = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)

    bg = convolve(y, BF / 32.0, mode="reflect")

    tb = np.where(
        bg <= 127.0,
        20.0 - 1.5085 * np.sqrt(np.maximum(bg, 0.0)),
        0.0234 * (bg - 127.0) + 3.0,
    )

    responses = [
        np.abs(convolve(y, kernel / 16.0, mode="reflect"))
        for kernel in G_FILTERS
    ]
    wl = np.maximum.reduce(responses)

    ts = wl * (0.0001 * bg + 0.115) + (0.5 - 0.01 * bg)
    return np.maximum(np.maximum(tb, ts), 0.0)


def gaussian_kernel_3x3(sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coords = np.arange(-1, 2, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def calculate_gaussian_jnd(jnd: np.ndarray, sigma: float) -> np.ndarray:
    return convolve(jnd, gaussian_kernel_3x3(sigma), mode="reflect")


def make_balanced_rx(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    total = height * width
    order = rng.permutation(total)
    rx_flat = np.full(total, 255, dtype=np.uint8)

    for flat_index in order:
        row, col = divmod(int(flat_index), width)
        r0, r1 = max(0, row - 1), min(height, row + 2)
        c0, c1 = max(0, col - 1), min(width, col + 2)

        idx = [
            rr * width + cc
            for rr in range(r0, r1)
            for cc in range(c0, c1)
            if not (rr == row and cc == col)
        ]
        assigned = rx_flat[idx]
        assigned = assigned[assigned != 255]
        n0 = int(np.count_nonzero(assigned == 0))
        n1 = int(np.count_nonzero(assigned == 1))

        if n1 >= 5:
            value = 0
        elif n0 >= 5:
            value = 1
        else:
            value = int(rng.integers(0, 2))

        rx_flat[flat_index] = value

    return rx_flat.reshape(shape)


def prepare_payload(
    message_bits: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(message_bits) > capacity:
        raise ValueError(
            f"Message contains {len(message_bits)} bits, capacity is {capacity}"
        )
    if len(message_bits) == capacity:
        return message_bits.copy()

    padding = rng.integers(0, 2, size=capacity - len(message_bits), dtype=np.uint8)
    return np.concatenate([message_bits, padding])


def secure_payload(
    payload: np.ndarray,
    shape: tuple[int, int],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = key_to_seed(key)
    rng = np.random.default_rng(seed)

    permutation = rng.permutation(payload.size)
    diffused = np.empty_like(payload)
    diffused[permutation] = payload

    rx = make_balanced_rx(shape, rng)
    hidden = np.bitwise_xor(diffused.reshape(shape), rx)
    secured = np.where(hidden == 1, 1.0, -1.0)
    return secured, permutation, rx


def regenerate_security_data(
    shape: tuple[int, int],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    seed = key_to_seed(key)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(shape[0] * shape[1])
    rx = make_balanced_rx(shape, rng)
    return permutation, rx


def optimized_predict_blue(blue: np.ndarray) -> np.ndarray:
    """
    Vectorized version of the Stage-3 3x3 predictor.

    For every 3x3 window:
    - determine local mean;
    - determine majority sign relative to the mean;
    - remove the majority-sign sample farthest from mean;
    - average the remaining eight samples.

    This avoids Python loops over pixels.
    """
    blue = blue.astype(np.float64)
    padded = np.pad(blue, 1, mode="reflect")
    windows = sliding_window_view(padded, (3, 3))
    flat = windows.reshape(blue.shape[0], blue.shape[1], 9)

    mu = flat.mean(axis=2, keepdims=True)
    signs_positive = flat >= mu
    positive_count = signs_positive.sum(axis=2)
    majority_positive = positive_count >= 5

    majority_mask = np.where(
        majority_positive[:, :, None],
        signs_positive,
        ~signs_positive,
    )

    distances = np.abs(flat - mu)
    masked_distances = np.where(majority_mask, distances, -1.0)
    remove_index = masked_distances.argmax(axis=2)

    total_sum = flat.sum(axis=2)
    removed_value = np.take_along_axis(
        flat,
        remove_index[:, :, None],
        axis=2,
    )[:, :, 0]

    return (total_sum - removed_value) / 8.0


def recover_payload(hidden_binary: np.ndarray, key: str) -> np.ndarray:
    permutation, rx = regenerate_security_data(hidden_binary.shape, key)
    diffused = np.bitwise_xor(hidden_binary, rx).reshape(-1)
    return diffused[permutation]


def metadata_default_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    return path.with_suffix(".watermark.json")


def embed(
    image_path: str | Path,
    watermark_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    generated_watermark_path: str | Path = "generated_watermark.txt",
    metadata_path: str | Path | None = None,
    key: str = DEFAULT_KEY,
    strength: float = DEFAULT_STRENGTH,
    sigma: float = DEFAULT_SIGMA,
    random_length: int | None = None,
) -> dict:
    image_path = Path(image_path)
    host = read_color_image(image_path)
    height, width = host.shape[:2]
    capacity = height * width

    if watermark_path is None:
        if random_length is None:
            random_length = capacity
        if random_length < 1 or random_length > capacity:
            raise ValueError(f"random_length must be from 1 to {capacity}")
        bits = np.random.default_rng().integers(
            0, 2, size=random_length, dtype=np.uint8
        )
        write_bits(generated_watermark_path, bits)
        watermark_file = str(generated_watermark_path)
    else:
        bits = read_bits(watermark_path)
        watermark_file = str(watermark_path)

    payload_rng = np.random.default_rng(key_to_seed(key) ^ 0xA5A5A5A5)
    payload = prepare_payload(bits, capacity, payload_rng)

    jnd = calculate_spatial_jnd(host)
    jnd_g = calculate_gaussian_jnd(jnd, sigma)
    secured, _, _ = secure_payload(payload, (height, width), key)

    result = host.astype(np.float64).copy()
    result[:, :, 0] = np.clip(
        np.rint(result[:, :, 0] + strength * secured * jnd_g),
        0,
        255,
    )
    watermarked = result.astype(np.uint8)

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_watermarked.png")
    save_color_image(output_path, watermarked)

    if metadata_path is None:
        metadata_path = metadata_default_path(output_path)

    metadata = {
        "method": "method5",
        "message_length": int(len(bits)),
        "capacity_bits": int(capacity),
        "image_width": int(width),
        "image_height": int(height),
        "key_fingerprint": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
        "strength": float(strength),
        "sigma": float(sigma),
        "watermark_file": watermark_file,
    }
    Path(metadata_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "mode": "embed",
        "input_image": str(image_path),
        "output_image": str(output_path),
        "metadata_file": str(metadata_path),
        "watermark_file": watermark_file,
        "image_width": width,
        "image_height": height,
        "capacity_bits": capacity,
        "message_bits": int(len(bits)),
        "bpp": float(len(bits) / capacity),
        "strength": float(strength),
        "sigma": float(sigma),
    }


def extract(
    image_path: str | Path,
    output_path: str | Path = "extracted_watermark.txt",
    *,
    key: str = DEFAULT_KEY,
    expected_length: int | None = None,
    metadata_path: str | Path | None = None,
) -> dict:
    image = read_color_image(image_path)
    height, width = image.shape[:2]
    capacity = height * width

    if expected_length is None:
        if metadata_path is None:
            candidate = metadata_default_path(image_path)
            metadata_path = candidate if candidate.exists() else None

        if metadata_path is not None and Path(metadata_path).exists():
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            expected_length = int(metadata["message_length"])

    if expected_length is None:
        expected_length = capacity

    if expected_length < 1 or expected_length > capacity:
        raise ValueError(
            f"expected_length must be from 1 to {capacity}"
        )

    blue = image[:, :, 0].astype(np.float64)
    predicted = optimized_predict_blue(blue)
    residual = blue - predicted
    hidden_binary = (residual >= 0).astype(np.uint8)

    payload = recover_payload(hidden_binary, key)
    extracted_bits = payload[:expected_length]
    write_bits(output_path, extracted_bits)

    return {
        "mode": "extract",
        "input_image": str(image_path),
        "output_watermark": str(output_path),
        "extracted_bits": int(len(extracted_bits)),
        "expected_length": int(expected_length),
        "metadata_used": str(metadata_path) if metadata_path is not None else None,
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
    }


def method5(
    action: Literal["embed", "extract"],
    image_path: str | Path,
    watermark_path: str | Path | None = None,
    **kwargs,
) -> dict:
    action = action.lower().strip()

    if action == "embed":
        return embed(image_path, watermark_path, **kwargs)

    if action == "extract":
        output_path = (
            watermark_path
            if watermark_path is not None
            else "extracted_watermark.txt"
        )
        return extract(image_path, output_path=output_path, **kwargs)

    raise ValueError("action must be 'embed' or 'extract'")


if __name__ == "__main__":
    print("Import method5 and call method5('embed', ...) or method5('extract', ...)")
