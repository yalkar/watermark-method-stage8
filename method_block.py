"""
Stage 8 — block differential DCT watermarking.

This module does not overwrite method5.py. It adds a more robust experimental
embedding scheme intended for JPEG, filtering, geometric resampling, and
print-camera tests.

Main idea:
- one logical bit is embedded into one or more 8x8/4x4 blocks;
- each block carries the bit through the ordering of two low-frequency DCT
  coefficients;
- the same logical bit is repeated in several pseudorandom blocks;
- extraction uses majority voting;
- optional JND adaptation changes the embedding margin without changing the
  detector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from method5 import (
    DEFAULT_KEY,
    calculate_gaussian_jnd,
    calculate_spatial_jnd,
    key_to_seed,
    read_bits,
    read_color_image,
    save_color_image,
    write_bits,
)


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    # Higher payload, mainly for median filtering, blur and small rotations.
    # JPEG robustness is limited compared with the 8x8 profiles.
    "block4_capacity": {
        "block_size": 4,
        "repetition": 3,
        "margin": 18.0,
        "coefficient_a": (0, 1),
        "coefficient_b": (1, 0),
        "jnd_adaptive": True,
        "jnd_min_factor": 0.80,
        "jnd_max_factor": 1.25,
    },
    # Default compromise between visual quality, capacity and robustness.
    "block8_balanced": {
        "block_size": 8,
        "repetition": 5,
        "margin": 40.0,
        "coefficient_a": (0, 1),
        "coefficient_b": (1, 0),
        "jnd_adaptive": True,
        "jnd_min_factor": 0.85,
        "jnd_max_factor": 1.25,
    },
    # Lower capacity and stronger modification for print-camera experiments.
    "block8_robust": {
        "block_size": 8,
        "repetition": 7,
        "margin": 64.0,
        "coefficient_a": (0, 1),
        "coefficient_b": (1, 0),
        "jnd_adaptive": True,
        "jnd_min_factor": 0.90,
        "jnd_max_factor": 1.20,
    },
}


def metadata_default_path(image_path: str | Path) -> Path:
    return Path(image_path).with_suffix(".block_watermark.json")


def get_profile(profile: str) -> dict[str, Any]:
    try:
        return dict(PROFILE_PRESETS[profile])
    except KeyError as exc:
        available = ", ".join(sorted(PROFILE_PRESETS))
        raise ValueError(
            f"Unknown profile {profile!r}. Available profiles: {available}"
        ) from exc


def _validate_parameters(
    block_size: int,
    repetition: int,
    coefficient_a: tuple[int, int],
    coefficient_b: tuple[int, int],
    margin: float,
) -> None:
    if block_size < 4:
        raise ValueError("block_size must be at least 4")
    if repetition < 1 or repetition % 2 == 0:
        raise ValueError("repetition must be a positive odd integer")
    if margin <= 0:
        raise ValueError("margin must be positive")
    for coefficient in (coefficient_a, coefficient_b):
        if len(coefficient) != 2:
            raise ValueError("DCT coefficient must contain two indices")
        if not all(0 <= value < block_size for value in coefficient):
            raise ValueError(
                f"DCT coefficient {coefficient} is outside a "
                f"{block_size}x{block_size} block"
            )
        if coefficient == (0, 0):
            raise ValueError("The DC coefficient (0, 0) must not be used")
    if coefficient_a == coefficient_b:
        raise ValueError("The two DCT coefficients must be different")


def capacity_for_image(
    image_shape: tuple[int, ...],
    *,
    block_size: int,
    repetition: int,
) -> dict[str, int]:
    height, width = image_shape[:2]
    block_rows = height // block_size
    block_cols = width // block_size
    raw_blocks = block_rows * block_cols
    logical_capacity = raw_blocks // repetition
    return {
        "block_rows": int(block_rows),
        "block_cols": int(block_cols),
        "raw_blocks": int(raw_blocks),
        "logical_capacity_bits": int(logical_capacity),
        "used_height": int(block_rows * block_size),
        "used_width": int(block_cols * block_size),
    }


def _security_maps(
    logical_capacity: int,
    raw_blocks: int,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = key_to_seed(key) ^ 0xB10CD7C7A5E8F913
    rng = np.random.default_rng(seed)

    payload_permutation = rng.permutation(logical_capacity)
    xor_mask = rng.integers(
        0,
        2,
        size=logical_capacity,
        dtype=np.uint8,
    )
    block_permutation = rng.permutation(raw_blocks)

    return payload_permutation, xor_mask, block_permutation


def _prepare_payload(
    message_bits: np.ndarray,
    logical_capacity: int,
    key: str,
) -> np.ndarray:
    if len(message_bits) > logical_capacity:
        raise ValueError(
            f"Watermark contains {len(message_bits)} bits, while the "
            f"logical capacity is {logical_capacity} bits"
        )

    if len(message_bits) == logical_capacity:
        return message_bits.copy()

    rng = np.random.default_rng(key_to_seed(key) ^ 0xA55A19F0C3D27741)
    padding = rng.integers(
        0,
        2,
        size=logical_capacity - len(message_bits),
        dtype=np.uint8,
    )
    return np.concatenate([message_bits, padding])


def _secure_payload(
    payload: np.ndarray,
    raw_blocks: int,
    repetition: int,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    logical_capacity = len(payload)
    payload_permutation, xor_mask, block_permutation = _security_maps(
        logical_capacity,
        raw_blocks,
        key,
    )

    diffused = np.empty_like(payload)
    diffused[payload_permutation] = payload
    secured = np.bitwise_xor(diffused, xor_mask)

    repeated = np.repeat(secured, repetition)
    used_block_indices = block_permutation[: len(repeated)]

    return repeated, used_block_indices


def _recover_payload(
    repeated_bits: np.ndarray,
    raw_blocks: int,
    repetition: int,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    logical_capacity = len(repeated_bits) // repetition
    payload_permutation, xor_mask, _ = _security_maps(
        logical_capacity,
        raw_blocks,
        key,
    )

    groups = repeated_bits.reshape(logical_capacity, repetition)
    vote_count = groups.sum(axis=1)
    secured = (
        vote_count >= (repetition // 2 + 1)
    ).astype(np.uint8)

    diffused = np.bitwise_xor(secured, xor_mask)
    payload = diffused[payload_permutation]

    agreement = np.maximum(vote_count, repetition - vote_count) / repetition
    return payload, agreement


def _block_margin_map(
    host: np.ndarray,
    *,
    block_size: int,
    block_rows: int,
    block_cols: int,
    base_margin: float,
    sigma: float,
    adaptive: bool,
    min_factor: float,
    max_factor: float,
) -> np.ndarray:
    if not adaptive:
        return np.full(
            (block_rows, block_cols),
            base_margin,
            dtype=np.float64,
        )

    jnd = calculate_spatial_jnd(host)
    jnd_g = calculate_gaussian_jnd(jnd, sigma)

    used = jnd_g[
        : block_rows * block_size,
        : block_cols * block_size,
    ]
    reshaped = used.reshape(
        block_rows,
        block_size,
        block_cols,
        block_size,
    )
    block_jnd = reshaped.mean(axis=(1, 3))
    reference = float(np.median(block_jnd))
    reference = max(reference, 1e-6)

    factors = np.clip(
        block_jnd / reference,
        min_factor,
        max_factor,
    )
    return base_margin * factors


def _embed_bit_in_block(
    block: np.ndarray,
    bit: int,
    *,
    coefficient_a: tuple[int, int],
    coefficient_b: tuple[int, int],
    margin: float,
) -> np.ndarray:
    transformed = cv2.dct(block.astype(np.float32))

    value_a = float(transformed[coefficient_a])
    value_b = float(transformed[coefficient_b])
    sign = 1.0 if bit == 1 else -1.0

    signed_difference = sign * (value_a - value_b)
    if signed_difference < margin:
        adjustment = (margin - signed_difference) / 2.0
        transformed[coefficient_a] = value_a + sign * adjustment
        transformed[coefficient_b] = value_b - sign * adjustment

    return cv2.idct(transformed)


def embed(
    image_path: str | Path,
    watermark_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    generated_watermark_path: str | Path = "generated_block_watermark.txt",
    metadata_path: str | Path | None = None,
    key: str = DEFAULT_KEY,
    profile: str = "block8_balanced",
    random_length: int | None = None,
    block_size: int | None = None,
    repetition: int | None = None,
    margin: float | None = None,
    coefficient_a: tuple[int, int] | None = None,
    coefficient_b: tuple[int, int] | None = None,
    jnd_adaptive: bool | None = None,
    jnd_sigma: float = 0.5,
    jnd_min_factor: float | None = None,
    jnd_max_factor: float | None = None,
) -> dict[str, Any]:
    parameters = get_profile(profile)

    block_size = int(
        parameters["block_size"] if block_size is None else block_size
    )
    repetition = int(
        parameters["repetition"] if repetition is None else repetition
    )
    margin = float(parameters["margin"] if margin is None else margin)
    coefficient_a = tuple(
        parameters["coefficient_a"]
        if coefficient_a is None
        else coefficient_a
    )
    coefficient_b = tuple(
        parameters["coefficient_b"]
        if coefficient_b is None
        else coefficient_b
    )
    jnd_adaptive = bool(
        parameters["jnd_adaptive"]
        if jnd_adaptive is None
        else jnd_adaptive
    )
    jnd_min_factor = float(
        parameters["jnd_min_factor"]
        if jnd_min_factor is None
        else jnd_min_factor
    )
    jnd_max_factor = float(
        parameters["jnd_max_factor"]
        if jnd_max_factor is None
        else jnd_max_factor
    )

    _validate_parameters(
        block_size,
        repetition,
        coefficient_a,
        coefficient_b,
        margin,
    )
    if jnd_sigma <= 0:
        raise ValueError("jnd_sigma must be positive")
    if not 0 < jnd_min_factor <= jnd_max_factor:
        raise ValueError("Invalid JND factor range")

    image_path = Path(image_path)
    host = read_color_image(image_path)
    capacity = capacity_for_image(
        host.shape,
        block_size=block_size,
        repetition=repetition,
    )
    logical_capacity = capacity["logical_capacity_bits"]
    if logical_capacity < 1:
        raise ValueError(
            "The image is too small for the chosen block size and repetition"
        )

    if watermark_path is None:
        if random_length is None:
            random_length = logical_capacity
        if not 1 <= random_length <= logical_capacity:
            raise ValueError(
                f"random_length must be from 1 to {logical_capacity}"
            )
        bits = np.random.default_rng().integers(
            0,
            2,
            size=random_length,
            dtype=np.uint8,
        )
        write_bits(generated_watermark_path, bits)
        watermark_file = str(generated_watermark_path)
    else:
        bits = read_bits(watermark_path)
        watermark_file = str(watermark_path)

    payload = _prepare_payload(bits, logical_capacity, key)
    repeated, used_block_indices = _secure_payload(
        payload,
        capacity["raw_blocks"],
        repetition,
        key,
    )

    margins = _block_margin_map(
        host,
        block_size=block_size,
        block_rows=capacity["block_rows"],
        block_cols=capacity["block_cols"],
        base_margin=margin,
        sigma=jnd_sigma,
        adaptive=jnd_adaptive,
        min_factor=jnd_min_factor,
        max_factor=jnd_max_factor,
    )

    result = host.astype(np.float32).copy()
    blue = result[:, :, 0]
    block_cols = capacity["block_cols"]

    for encoded_index, block_index in enumerate(used_block_indices):
        block_index = int(block_index)
        row = block_index // block_cols
        col = block_index % block_cols

        y0 = row * block_size
        x0 = col * block_size
        y1 = y0 + block_size
        x1 = x0 + block_size

        blue[y0:y1, x0:x1] = _embed_bit_in_block(
            blue[y0:y1, x0:x1],
            int(repeated[encoded_index]),
            coefficient_a=coefficient_a,
            coefficient_b=coefficient_b,
            margin=float(margins[row, col]),
        )

    watermarked = np.clip(
        np.rint(result),
        0,
        255,
    ).astype(np.uint8)

    if output_path is None:
        output_path = image_path.with_name(
            f"{image_path.stem}_{profile}_watermarked.png"
        )
    output_path = Path(output_path)
    save_color_image(output_path, watermarked)

    if metadata_path is None:
        metadata_path = metadata_default_path(output_path)
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "method": "block_dct_differential_v1",
        "profile": profile,
        "message_length": int(len(bits)),
        "logical_capacity_bits": int(logical_capacity),
        "raw_block_capacity_bits": int(capacity["raw_blocks"]),
        "image_width": int(host.shape[1]),
        "image_height": int(host.shape[0]),
        "used_width": int(capacity["used_width"]),
        "used_height": int(capacity["used_height"]),
        "block_size": int(block_size),
        "repetition": int(repetition),
        "margin": float(margin),
        "coefficient_a": list(coefficient_a),
        "coefficient_b": list(coefficient_b),
        "jnd_adaptive": bool(jnd_adaptive),
        "jnd_sigma": float(jnd_sigma),
        "jnd_min_factor": float(jnd_min_factor),
        "jnd_max_factor": float(jnd_max_factor),
        "key_fingerprint": hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()[:16],
        "watermark_file": watermark_file,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "mode": "embed",
        "input_image": str(image_path),
        "output_image": str(output_path),
        "metadata_file": str(metadata_path),
        "watermark_file": watermark_file,
        "profile": profile,
        "message_bits": int(len(bits)),
        "logical_capacity_bits": int(logical_capacity),
        "raw_block_capacity_bits": int(capacity["raw_blocks"]),
        "bpp": float(len(bits) / (host.shape[0] * host.shape[1])),
        "block_size": int(block_size),
        "repetition": int(repetition),
        "margin": float(margin),
    }


def _raw_block_decisions(
    blue: np.ndarray,
    *,
    block_size: int,
    block_rows: int,
    block_cols: int,
    coefficient_a: tuple[int, int],
    coefficient_b: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    raw_blocks = block_rows * block_cols
    decisions = np.zeros(raw_blocks, dtype=np.uint8)
    confidence = np.zeros(raw_blocks, dtype=np.float64)

    index = 0
    for row in range(block_rows):
        y0 = row * block_size
        y1 = y0 + block_size
        for col in range(block_cols):
            x0 = col * block_size
            x1 = x0 + block_size

            transformed = cv2.dct(
                blue[y0:y1, x0:x1].astype(np.float32)
            )
            difference = float(
                transformed[coefficient_a]
                - transformed[coefficient_b]
            )
            decisions[index] = 1 if difference >= 0 else 0
            confidence[index] = abs(difference)
            index += 1

    return decisions, confidence


def _decode_candidate(
    blue: np.ndarray,
    *,
    block_size: int,
    repetition: int,
    block_rows: int,
    block_cols: int,
    coefficient_a: tuple[int, int],
    coefficient_b: tuple[int, int],
    key: str,
) -> dict[str, Any]:
    raw_blocks = block_rows * block_cols
    logical_capacity = raw_blocks // repetition
    _, _, block_permutation = _security_maps(
        logical_capacity,
        raw_blocks,
        key,
    )
    used_count = logical_capacity * repetition
    used_block_indices = block_permutation[:used_count]

    raw_decisions, raw_confidence = _raw_block_decisions(
        blue,
        block_size=block_size,
        block_rows=block_rows,
        block_cols=block_cols,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
    )
    repeated_bits = raw_decisions[used_block_indices]
    repeated_confidence = raw_confidence[used_block_indices]

    payload, agreement = _recover_payload(
        repeated_bits,
        raw_blocks,
        repetition,
        key,
    )

    return {
        "payload": payload,
        "agreement_mean": float(agreement.mean()),
        "agreement_min": float(agreement.min()),
        "coefficient_confidence_median": float(
            np.median(repeated_confidence)
        ),
    }


def extract(
    image_path: str | Path,
    output_path: str | Path = "extracted_block_watermark.txt",
    *,
    key: str = DEFAULT_KEY,
    metadata_path: str | Path | None = None,
    expected_length: int | None = None,
    offset_search_radius: int = 0,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = read_color_image(image_path)

    if metadata_path is None:
        candidate = metadata_default_path(image_path)
        metadata_path = candidate if candidate.exists() else None

    if metadata_path is None or not Path(metadata_path).exists():
        raise FileNotFoundError(
            "Metadata is required for block extraction. "
            "Pass metadata_path explicitly."
        )

    metadata = json.loads(
        Path(metadata_path).read_text(encoding="utf-8")
    )
    if metadata.get("method") != "block_dct_differential_v1":
        raise ValueError("Metadata belongs to another watermark method")

    expected_width = int(metadata["image_width"])
    expected_height = int(metadata["image_height"])
    if image.shape[1] != expected_width or image.shape[0] != expected_height:
        raise ValueError(
            "Image dimensions do not match metadata: "
            f"got {image.shape[1]}x{image.shape[0]}, expected "
            f"{expected_width}x{expected_height}. Run geometric "
            "synchronization first."
        )

    block_size = int(metadata["block_size"])
    repetition = int(metadata["repetition"])
    coefficient_a = tuple(int(v) for v in metadata["coefficient_a"])
    coefficient_b = tuple(int(v) for v in metadata["coefficient_b"])

    if expected_length is None:
        expected_length = int(metadata["message_length"])

    capacity = capacity_for_image(
        image.shape,
        block_size=block_size,
        repetition=repetition,
    )
    if not 1 <= expected_length <= capacity["logical_capacity_bits"]:
        raise ValueError(
            f"expected_length must be from 1 to "
            f"{capacity['logical_capacity_bits']}"
        )

    if offset_search_radius < 0:
        raise ValueError("offset_search_radius must not be negative")
    if offset_search_radius >= block_size:
        raise ValueError(
            "offset_search_radius must be smaller than block_size"
        )

    blue = image[:, :, 0].astype(np.float32)
    candidates: list[dict[str, Any]] = []

    for dy in range(-offset_search_radius, offset_search_radius + 1):
        for dx in range(-offset_search_radius, offset_search_radius + 1):
            shifted = np.roll(blue, shift=(dy, dx), axis=(0, 1))
            candidate = _decode_candidate(
                shifted,
                block_size=block_size,
                repetition=repetition,
                block_rows=capacity["block_rows"],
                block_cols=capacity["block_cols"],
                coefficient_a=coefficient_a,
                coefficient_b=coefficient_b,
                key=key,
            )
            candidate["offset_x"] = int(dx)
            candidate["offset_y"] = int(dy)
            candidates.append(candidate)

    best = max(
        candidates,
        key=lambda item: (
            item["agreement_mean"],
            item["coefficient_confidence_median"],
        ),
    )

    extracted_bits = best["payload"][:expected_length]
    write_bits(output_path, extracted_bits)

    return {
        "mode": "extract",
        "input_image": str(image_path),
        "output_watermark": str(output_path),
        "metadata_used": str(metadata_path),
        "extracted_bits": int(len(extracted_bits)),
        "expected_length": int(expected_length),
        "best_offset_x": int(best["offset_x"]),
        "best_offset_y": int(best["offset_y"]),
        "repetition_agreement_mean": float(best["agreement_mean"]),
        "repetition_agreement_min": float(best["agreement_min"]),
        "coefficient_confidence_median": float(
            best["coefficient_confidence_median"]
        ),
    }


def method_block(
    action: Literal["embed", "extract"],
    image_path: str | Path,
    watermark_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    action = action.lower().strip()

    if action == "embed":
        return embed(
            image_path,
            watermark_path,
            **kwargs,
        )

    if action == "extract":
        output_path = (
            watermark_path
            if watermark_path is not None
            else "extracted_block_watermark.txt"
        )
        return extract(
            image_path,
            output_path=output_path,
            **kwargs,
        )

    raise ValueError("action must be 'embed' or 'extract'")


if __name__ == "__main__":
    print(
        "Import method_block and call "
        "method_block('embed', ...) or method_block('extract', ...)"
    )
