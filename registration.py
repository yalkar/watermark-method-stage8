"""Geometric synchronization for photographed or digitally transformed images.

The moving/captured image is aligned to a reference watermarked image by:
1. local feature detection (SIFT by default, ORB fallback),
2. descriptor matching,
3. RANSAC homography estimation,
4. perspective warping to the reference dimensions.

This module does not perform color correction. It only restores geometry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


DetectorName = Literal["auto", "sift", "orb"]


def read_color(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def _create_detector(name: DetectorName, max_features: int):
    requested = name.lower()
    if requested not in {"auto", "sift", "orb"}:
        raise ValueError("detector must be 'auto', 'sift', or 'orb'")

    if requested in {"auto", "sift"} and hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=max_features)

    if requested == "sift":
        raise RuntimeError("SIFT is unavailable in this OpenCV build")

    return "orb", cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=8,
    )


def _ratio_matches(desc_moving, desc_reference, detector_name: str, ratio: float):
    if detector_name == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pairs = matcher.knnMatch(desc_moving, desc_reference, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def align_to_reference(
    moving_image: np.ndarray,
    reference_image: np.ndarray,
    *,
    detector: DetectorName = "auto",
    max_features: int = 8000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
    min_matches: int = 12,
    min_inliers: int = 8,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Align ``moving_image`` to ``reference_image``.

    Returns ``(aligned_image, diagnostics, visualization)``.
    The homography maps moving-image coordinates into reference coordinates.
    """
    if moving_image.ndim != 3 or reference_image.ndim != 3:
        raise ValueError("Both images must be color images")

    detector_name, feature_detector = _create_detector(detector, max_features)

    moving_gray = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    keypoints_moving, desc_moving = feature_detector.detectAndCompute(moving_gray, None)
    keypoints_reference, desc_reference = feature_detector.detectAndCompute(reference_gray, None)

    if desc_moving is None or desc_reference is None:
        raise RuntimeError("Not enough image texture to compute feature descriptors")

    good_matches = _ratio_matches(
        desc_moving,
        desc_reference,
        detector_name,
        ratio_test,
    )

    if len(good_matches) < min_matches:
        raise RuntimeError(
            f"Only {len(good_matches)} reliable matches found; at least {min_matches} required"
        )

    source_points = np.float32(
        [keypoints_moving[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [keypoints_reference[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        ransac_threshold,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed")

    inlier_mask_flat = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask_flat.sum())
    if inliers < min_inliers:
        raise RuntimeError(
            f"Homography has only {inliers} inliers; at least {min_inliers} required"
        )

    ref_height, ref_width = reference_image.shape[:2]
    aligned = cv2.warpPerspective(
        moving_image,
        homography,
        (ref_width, ref_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    projected = cv2.perspectiveTransform(source_points, homography)
    reprojection_errors = np.linalg.norm(
        projected.reshape(-1, 2) - destination_points.reshape(-1, 2),
        axis=1,
    )
    inlier_errors = reprojection_errors[inlier_mask_flat]

    visualization = cv2.drawMatches(
        moving_image,
        keypoints_moving,
        reference_image,
        keypoints_reference,
        good_matches,
        None,
        matchesMask=inlier_mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    diagnostics = {
        "detector": detector_name,
        "moving_keypoints": int(len(keypoints_moving)),
        "reference_keypoints": int(len(keypoints_reference)),
        "good_matches": int(len(good_matches)),
        "inliers": inliers,
        "inlier_ratio": float(inliers / len(good_matches)),
        "reprojection_error_mean": float(inlier_errors.mean()),
        "reprojection_error_median": float(np.median(inlier_errors)),
        "homography": homography.tolist(),
        "reference_width": int(ref_width),
        "reference_height": int(ref_height),
    }
    return aligned, diagnostics, visualization


def synchronize_image(
    captured_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path = "synchronized.png",
    *,
    diagnostics_path: str | Path | None = None,
    matches_path: str | Path | None = None,
    detector: DetectorName = "auto",
    max_features: int = 8000,
    ratio_test: float = 0.75,
    ransac_threshold: float = 4.0,
) -> dict:
    captured = read_color(captured_path)
    reference = read_color(reference_path)

    aligned, diagnostics, visualization = align_to_reference(
        captured,
        reference,
        detector=detector,
        max_features=max_features,
        ratio_test=ratio_test,
        ransac_threshold=ransac_threshold,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), aligned):
        raise OSError(f"Cannot save synchronized image: {output_path}")

    if diagnostics_path is None:
        diagnostics_path = output_path.with_suffix(".registration.json")
    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if matches_path is None:
        matches_path = output_path.with_name(f"{output_path.stem}_matches.jpg")
    matches_path = Path(matches_path)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(matches_path), visualization):
        raise OSError(f"Cannot save match visualization: {matches_path}")

    return {
        "captured_image": str(captured_path),
        "reference_image": str(reference_path),
        "synchronized_image": str(output_path),
        "diagnostics_file": str(diagnostics_path),
        "matches_image": str(matches_path),
        **diagnostics,
    }
