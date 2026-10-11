from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError as error:
    raise ImportError(
        "Установи поддержку HEIC: python -m pip install pillow-heif"
    ) from error

register_heif_opener()

from method_block import extract, read_bits
from registration import synchronize_image


# ============================================================================
# ФИНАЛЬНЫЙ PHONE PRINT-CAMERA ПРОГОН
# Автоматическое сопоставление группы из 3 снимков с PRINT_READY
# + геометрическая синхронизация + извлечение ЦВЗ
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PHOTO_DIR = PROJECT_ROOT / "Итоговый список снимков"

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "full_experiment_stage8_256_r15"
    / "20260817_161337"
)

REFERENCE_DIR = EXPERIMENT_ROOT / "PRINT_READY"
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
ORIGINAL_WATERMARK = EXPERIMENT_ROOT / "watermark_256_bits.txt"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "final_phone_stage8_auto_sync_final"
    / RUN_ID
)

PROFILE_NAME = "block8_256_r15"

EXPECTED_PHOTOS = 150
SHOTS_PER_GROUP = 3
EXPECTED_GROUPS = EXPECTED_PHOTOS // SHOTS_PER_GROUP

OFFSET_SEARCH_RADIUS = 3

# Параметры только для определения, какой PRINT_READY изображён на фото.
SIFT_MAX_FEATURES = 3500
MATCH_RATIO = 0.75
TOP_CANDIDATES = 5
MATCH_MAX_SIDE = 1600

SUPPORTED_EXTENSIONS = {
    ".heic", ".heif",
    ".jpg", ".jpeg",
    ".png",
}


def natural_key(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    if numbers:
        return (0, int(numbers[-1]), path.name.lower())
    return (1, 0, path.name.lower())


def calculate_ber(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(
            f"Разная длина битовых последовательностей: "
            f"{len(a)} и {len(b)}"
        )
    return float((a != b).mean())


def safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def make_stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_ber": mean(values),
        "median_ber": median(values),
        "min_ber": min(values),
        "max_ber": max(values),
        "std_ber": safe_std(values),
        "mean_correct_percent": (1.0 - mean(values)) * 100.0,
    }


def collect_phone_photos() -> list[Path]:
    if not PHOTO_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка:\n{PHOTO_DIR}")

    photos = sorted(
        (
            path
            for path in PHOTO_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_key,
    )

    if len(photos) != EXPECTED_PHOTOS:
        raise ValueError(
            f"Ожидалось {EXPECTED_PHOTOS} поддерживаемых снимков, "
            f"но найдено {len(photos)}.\n"
            f"Папка: {PHOTO_DIR}"
        )

    return photos


def collect_references() -> list[dict]:
    refs = []

    pattern = re.compile(
        rf"^(\d{{6}})_{re.escape(PROFILE_NAME)}\.png$",
        re.IGNORECASE,
    )

    for path in sorted(REFERENCE_DIR.glob("*.png")):
        match = pattern.match(path.name)
        if not match:
            continue

        image_id = match.group(1)
        metadata = (
            METADATA_DIR
            / f"{image_id}_{PROFILE_NAME}.block_watermark.json"
        )

        if not metadata.exists():
            raise FileNotFoundError(metadata)

        refs.append(
            {
                "image_id": image_id,
                "reference": path,
                "metadata": metadata,
            }
        )

    if len(refs) != 50:
        raise ValueError(
            f"В PRINT_READY ожидалось 50 эталонов, найдено {len(refs)}."
        )

    return refs


def normalize_to_png(source: Path, destination: Path) -> None:
    """
    HEIC/JPG/PNG -> RGB PNG с учётом EXIF-ориентации.

    Для JPG нового JPEG-сжатия нет: файл только декодируется,
    после чего пиксели сохраняются в lossless PNG.
    Исходные файлы не изменяются.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(
            destination,
            format="PNG",
            compress_level=3,
        )


def read_cv_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Не удалось открыть: {path}")

    return image


def resize_for_matching(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    max_side = max(h, w)

    if max_side <= MATCH_MAX_SIDE:
        return gray

    scale = MATCH_MAX_SIDE / max_side
    return cv2.resize(
        gray,
        (round(w * scale), round(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def compute_features(
    sift,
    image_path: Path,
) -> tuple[list, np.ndarray | None]:
    image = read_cv_unicode(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize_for_matching(gray)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def ratio_matches(
    matcher,
    ref_desc: np.ndarray | None,
    photo_desc: np.ndarray | None,
):
    if ref_desc is None or photo_desc is None:
        return []

    if len(ref_desc) < 2 or len(photo_desc) < 2:
        return []

    pairs = matcher.knnMatch(ref_desc, photo_desc, k=2)

    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < MATCH_RATIO * second.distance:
            good.append(first)

    return good


def homography_inliers(
    ref_kp,
    photo_kp,
    matches,
) -> int:
    if len(matches) < 4:
        return 0

    # queryIdx -> reference, trainIdx -> phone
    phone_points = np.float32(
        [photo_kp[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    ref_points = np.float32(
        [ref_kp[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(
        phone_points,
        ref_points,
        cv2.RANSAC,
        5.0,
    )

    if mask is None:
        return 0

    return int(mask.ravel().sum())


def prepare_inputs(
    photos: list[Path],
) -> list[Path]:
    prepared_dir = OUTPUT_DIR / "input_png"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    prepared = []

    for index, source in enumerate(photos, start=1):
        ext_tag = source.suffix.lower().lstrip(".")
        destination = prepared_dir / f"{source.stem}_{ext_tag}.png"

        print(
            f"INPUT -> PNG [{index:03d}/{len(photos):03d}] "
            f"{source.name}"
        )

        normalize_to_png(source, destination)
        prepared.append(destination)

    return prepared


def build_reference_features(
    sift,
    references: list[dict],
) -> None:
    print("\nВычисляю признаки для 50 PRINT_READY...")

    for index, ref in enumerate(references, start=1):
        kp, desc = compute_features(
            sift,
            ref["reference"],
        )

        ref["kp"] = kp
        ref["desc"] = desc

        print(
            f"  [{index:02d}/50] "
            f"{ref['image_id']} | features={len(kp)}"
        )


def match_group(
    *,
    group_number: int,
    group_paths: list[Path],
    sift,
    matcher,
    references: list[dict],
) -> tuple[dict, list[dict], list[tuple]]:
    """
    Каждая группа = три последовательных кадра одной распечатки.

    Сначала для каждого из 50 эталонов считаем суммарное число good matches
    по всем трём кадрам. Затем у TOP_CANDIDATES считаем RANSAC-inliers.
    Побеждает кандидат с максимальным суммарным числом inliers.
    """
    photo_features = []

    for path in group_paths:
        kp, desc = compute_features(sift, path)
        photo_features.append((path, kp, desc))

    candidates = []

    for ref in references:
        matches_per_shot = []
        good_total = 0

        for _, _, photo_desc in photo_features:
            good = ratio_matches(
                matcher,
                ref["desc"],
                photo_desc,
            )
            matches_per_shot.append(good)
            good_total += len(good)

        candidates.append(
            {
                "ref": ref,
                "good_total": good_total,
                "matches_per_shot": matches_per_shot,
            }
        )

    candidates.sort(
        key=lambda item: item["good_total"],
        reverse=True,
    )

    finalists = candidates[:TOP_CANDIDATES]

    for candidate in finalists:
        inliers_per_shot = []

        for shot_idx, (_, photo_kp, _) in enumerate(photo_features):
            inliers = homography_inliers(
                candidate["ref"]["kp"],
                photo_kp,
                candidate["matches_per_shot"][shot_idx],
            )
            inliers_per_shot.append(inliers)

        candidate["inliers_per_shot"] = inliers_per_shot
        candidate["inliers_total"] = sum(inliers_per_shot)

    finalists.sort(
        key=lambda item: (
            item["inliers_total"],
            item["good_total"],
        ),
        reverse=True,
    )

    best = finalists[0]

    second_inliers = (
        finalists[1]["inliers_total"]
        if len(finalists) > 1
        else 0
    )

    best["second_inliers_total"] = second_inliers
    best["inlier_gap"] = best["inliers_total"] - second_inliers

    print(
        f"[GROUP {group_number:02d}/{EXPECTED_GROUPS:02d}] "
        f"{group_paths[0].stem} .. {group_paths[-1].stem} "
        f"-> {best['ref']['image_id']} | "
        f"inliers={best['inliers_total']} | "
        f"good={best['good_total']} | "
        f"gap={best['inlier_gap']}"
    )

    return best, finalists, photo_features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    photos = collect_phone_photos()
    references = collect_references()

    if not ORIGINAL_WATERMARK.exists():
        raise FileNotFoundError(ORIGINAL_WATERMARK)

    original_bits = read_bits(ORIGINAL_WATERMARK)

    if len(original_bits) != 256:
        raise ValueError(
            f"Ожидалось 256 бит ЦВЗ, найдено {len(original_bits)}."
        )

    prepared_photos = prepare_inputs(photos)

    sift = cv2.SIFT_create(
        nfeatures=SIFT_MAX_FEATURES,
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    build_reference_features(
        sift,
        references,
    )

    groups = [
        prepared_photos[i:i + SHOTS_PER_GROUP]
        for i in range(
            0,
            len(prepared_photos),
            SHOTS_PER_GROUP,
        )
    ]

    print()
    print("=" * 88)
    print("ЭТАП 1 — СОПОСТАВЛЕНИЕ 50 ГРУПП С PRINT_READY")
    print("=" * 88)

    mapping_rows = []
    group_matches = []

    for group_idx, group_paths in enumerate(groups, start=1):
        best, finalists, _ = match_group(
            group_number=group_idx,
            group_paths=group_paths,
            sift=sift,
            matcher=matcher,
            references=references,
        )

        group_matches.append(
            {
                "group": group_idx,
                "paths": group_paths,
                "best": best,
            }
        )

        top_ids = [
            item["ref"]["image_id"]
            for item in finalists
        ]
        top_inliers = [
            item["inliers_total"]
            for item in finalists
        ]

        mapping_rows.append(
            {
                "group": group_idx,
                "phone_file_1": group_paths[0].stem,
                "phone_file_2": group_paths[1].stem,
                "phone_file_3": group_paths[2].stem,
                "matched_image": best["ref"]["image_id"],
                "good_matches_total": best["good_total"],
                "inliers_total": best["inliers_total"],
                "second_inliers_total": best["second_inliers_total"],
                "inlier_gap": best["inlier_gap"],
                "top_candidates": ";".join(top_ids),
                "top_inliers": ";".join(map(str, top_inliers)),
            }
        )

    mapping_csv = OUTPUT_DIR / "auto_mapping.csv"

    with mapping_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=mapping_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    counts = Counter(
        row["matched_image"]
        for row in mapping_rows
    )

    missing_ids = [
        ref["image_id"]
        for ref in references
        if counts[ref["image_id"]] == 0
    ]

    duplicate_ids = {
        image_id: count
        for image_id, count in sorted(counts.items())
        if count > 1
    }

    print()
    print("Сопоставление завершено.")
    print(f"Уникальных PRINT_READY найдено: {len(counts)}/50")
    print(
        "Не найдены: "
        + (", ".join(missing_ids) if missing_ids else "нет")
    )
    print(
        "Повторы: "
        + (
            ", ".join(
                f"{image_id} x{count}"
                for image_id, count in duplicate_ids.items()
            )
            if duplicate_ids
            else "нет"
        )
    )
    print(f"Карта: {mapping_csv}")

    # Финальный набор должен быть строго сбалансирован:
    # 50 уникальных PRINT_READY, каждый встретился ровно один раз как группа из 3 кадров.
    if len(counts) != 50 or missing_ids or duplicate_ids:
        print()
        print("СТОП: карта не является чистой 50 x 3.")
        print("Проверь auto_mapping.csv. BER пока не считаем.")
        raise SystemExit(2)

    print()
    print("Карта корректна: 50 уникальных изображений x 3 кадра.")

    print()
    print("=" * 88)
    print("ЭТАП 2 — ГЕОМЕТРИЧЕСКАЯ СИНХРОНИЗАЦИЯ + ИЗВЛЕЧЕНИЕ")
    print("=" * 88)

    result_rows = []

    for group in group_matches:
        group_idx = group["group"]
        matched_ref = group["best"]["ref"]
        image_id = matched_ref["image_id"]

        for shot_idx, photo_path in enumerate(
            group["paths"],
            start=1,
        ):
            case_dir = (
                OUTPUT_DIR
                / "cases"
                / f"group_{group_idx:02d}_{image_id}"
                / f"shot_{shot_idx:02d}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)

            synchronized = case_dir / "synchronized.png"
            extracted = case_dir / "extracted.txt"

            row = {
                "group": group_idx,
                "shot": shot_idx,
                "phone_file": photo_path.stem,
                "matched_image": image_id,
                "status": "",
                "ber": "",
                "correct_bits_percent": "",
                "good_matches": "",
                "inliers": "",
                "reprojection_error_mean": "",
                "best_offset_x": "",
                "best_offset_y": "",
                "repetition_agreement_mean": "",
                "error": "",
            }

            print(
                f"[G{group_idx:02d} S{shot_idx}] "
                f"{photo_path.stem} -> {image_id}"
            )

            try:
                registration_result = synchronize_image(
                    captured_path=photo_path,
                    reference_path=matched_ref["reference"],
                    output_path=synchronized,
                    diagnostics_path=case_dir / "registration.json",
                    matches_path=case_dir / "matches.jpg",
                )

                extraction_result = extract(
                    image_path=synchronized,
                    output_path=extracted,
                    metadata_path=matched_ref["metadata"],
                    offset_search_radius=OFFSET_SEARCH_RADIUS,
                )

                extracted_bits = read_bits(extracted)
                ber = calculate_ber(
                    original_bits,
                    extracted_bits,
                )
                correct = (1.0 - ber) * 100.0

                row.update(
                    {
                        "status": "PASS",
                        "ber": ber,
                        "correct_bits_percent": correct,
                        "good_matches": registration_result[
                            "good_matches"
                        ],
                        "inliers": registration_result["inliers"],
                        "reprojection_error_mean": registration_result[
                            "reprojection_error_mean"
                        ],
                        "best_offset_x": extraction_result[
                            "best_offset_x"
                        ],
                        "best_offset_y": extraction_result[
                            "best_offset_y"
                        ],
                        "repetition_agreement_mean": extraction_result[
                            "repetition_agreement_mean"
                        ],
                    }
                )

                print(
                    f"    BER={ber:.6f} | "
                    f"correct={correct:.2f}% | "
                    f"inliers={registration_result['inliers']}"
                )

            except Exception as error:
                row["status"] = "ERROR"
                row["error"] = str(error)

                print(f"    ERROR: {error}")

            result_rows.append(row)

    results_csv = OUTPUT_DIR / "sync_results.csv"

    with results_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    valid_rows = [
        row
        for row in result_rows
        if row["status"] == "PASS"
    ]

    summary_rows = []

    for image_id in sorted(counts):
        values = [
            float(row["ber"])
            for row in valid_rows
            if row["matched_image"] == image_id
        ]

        if not values:
            continue

        summary_rows.append(
            {
                "image": image_id,
                **make_stats(values),
            }
        )

    summary_csv = OUTPUT_DIR / "summary_by_matched_image.csv"

    if summary_rows:
        with summary_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=summary_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    all_bers = [
        float(row["ber"])
        for row in valid_rows
    ]

    error_count = sum(
        row["status"] == "ERROR"
        for row in result_rows
    )

    print()
    print("=" * 88)
    print("ИТОГ")
    print("=" * 88)
    print(f"Успешно обработано: {len(valid_rows)}/{len(result_rows)}")
    print(f"Ошибок регистрации/извлечения: {error_count}")
    print(f"Уникальных изображений: {len(counts)}/50")
    print(
        "Не найдены: "
        + (", ".join(missing_ids) if missing_ids else "нет")
    )

    if all_bers:
        overall = make_stats(all_bers)

        print(f"Mean BER:      {overall['mean_ber']:.6f}")
        print(f"Median BER:    {overall['median_ber']:.6f}")
        print(f"Min BER:       {overall['min_ber']:.6f}")
        print(f"Max BER:       {overall['max_ber']:.6f}")
        print(f"Std BER:       {overall['std_ber']:.6f}")
        print(
            f"Mean correct:  "
            f"{overall['mean_correct_percent']:.2f}%"
        )

        overall_row = {
            "mode": "auto_match_plus_geometric_sync",
            "photos_total": len(result_rows),
            "photos_success": len(valid_rows),
            "errors": error_count,
            "unique_matched_images": len(counts),
            "missing_images": ";".join(missing_ids),
            "duplicates": ";".join(
                f"{k}:{v}"
                for k, v in duplicate_ids.items()
            ),
            **overall,
        }

        overall_csv = OUTPUT_DIR / "overall_summary.csv"

        with overall_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=overall_row.keys(),
            )
            writer.writeheader()
            writer.writerow(overall_row)

        print()
        print(f"Карта соответствий: {mapping_csv}")
        print(f"Все BER:            {results_csv}")
        print(f"По изображениям:    {summary_csv}")
        print(f"Общая сводка:       {overall_csv}")


if __name__ == "__main__":
    main()
