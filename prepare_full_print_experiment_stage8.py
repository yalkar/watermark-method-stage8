from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from method_block import (
    PROFILE_PRESETS,
    capacity_for_image,
    embed,
    extract,
    read_bits,
)


# ============================================================================
# НАСТРОЙКИ ПОЛНОГО PRINT-CAMERA ЭКСПЕРИМЕНТА
# ============================================================================

# Сюда положить ровно те исходные изображения, которые войдут в эксперимент.
INPUT_DIR = Path("demo_image")

# Ожидаем 50 изображений. Если хочешь сначала сделать пробный прогон,
# временно поменяй это число.
EXPECTED_IMAGE_COUNT = 1

# Подготовка изображения к квадратной печати 6x6 см.
OUTPUT_SIZE = 512

# Финальный профиль, выбранный после capacity sweep.
PROFILE_NAME = "block8_256_r15"
WATERMARK_BITS = 256
REPETITION = 15

# Остальные параметры полностью наследуем от уже рабочего block8_robust.
_base_robust = dict(PROFILE_PRESETS["block8_robust"])
PROFILE_PRESETS[PROFILE_NAME] = {
    **_base_robust,
    "repetition": REPETITION,
}

# Один и тот же 256-битный ЦВЗ для всех изображений:
# так сравнение BER между изображениями будет чище.
RANDOM_SEED = 20260817

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Каждый запуск создаёт отдельную папку, чтобы старые результаты не смешивались.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path("full_experiment_stage8_256_r15") / RUN_ID

HOST_DIR = OUTPUT_ROOT / "hosts_512"
WORK_DIR = OUTPUT_ROOT / "work_watermarked"
PRINT_DIR = OUTPUT_ROOT / "PRINT_READY"
METADATA_DIR = OUTPUT_ROOT / "metadata"
EXTRACTED_DIR = OUTPUT_ROOT / "digital_extracted"

SUMMARY_CSV = OUTPUT_ROOT / "digital_control.csv"
MANIFEST_JSON = OUTPUT_ROOT / "experiment_manifest.json"
WATERMARK_PATH = OUTPUT_ROOT / "watermark_256_bits.txt"


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Не удалось сохранить изображение: {path}")


def write_bits(path: Path, bits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(bits.astype(str).tolist()),
        encoding="utf-8",
    )


def calculate_ber(original: np.ndarray, extracted: np.ndarray) -> float:
    if original.shape != extracted.shape:
        raise ValueError(
            f"Разная длина последовательностей: "
            f"{len(original)} и {len(extracted)}"
        )
    return float(np.mean(original != extracted))


def image_quality(
    original: np.ndarray,
    watermarked: np.ndarray,
) -> tuple[float, float]:
    psnr = float(cv2.PSNR(original, watermarked))

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(
        watermarked,
        cv2.COLOR_BGR2GRAY,
    )

    ssim = float(
        structural_similarity(
            original_gray,
            watermarked_gray,
            data_range=255,
        )
    )

    return psnr, ssim


def prepare_square(image: np.ndarray, size: int) -> np.ndarray:
    """
    Центр-кроп до квадрата и resize до 512x512.
    Если изображение уже 512x512, пиксели не меняются.
    """
    height, width = image.shape[:2]

    if height == size and width == size:
        return image.copy()

    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2

    cropped = image[y0:y0 + side, x0:x0 + side]

    interpolation = (
        cv2.INTER_AREA
        if side >= size
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        cropped,
        (size, size),
        interpolation=interpolation,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_input_images() -> list[Path]:
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"Создана папка '{INPUT_DIR}'. "
            f"Положи туда {EXPECTED_IMAGE_COUNT} изображений "
            f"и запусти скрипт ещё раз."
        )

    images = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if len(images) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"В '{INPUT_DIR}' найдено {len(images)} изображений, "
            f"а для полного эксперимента ожидается "
            f"{EXPECTED_IMAGE_COUNT}. "
            f"Проверь набор перед подготовкой к печати."
        )

    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "В наборе есть файлы с одинаковыми именами без расширения. "
            "Переименуй их, чтобы выходные PNG не перезаписывались."
        )

    return images


# ============================================================================
# ОСНОВНОЙ ПРОГОН
# ============================================================================

def main() -> None:
    input_images = get_input_images()

    for directory in (
        OUTPUT_ROOT,
        HOST_DIR,
        WORK_DIR,
        PRINT_DIR,
        METADATA_DIR,
        EXTRACTED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    params = PROFILE_PRESETS[PROFILE_NAME]

    # Проверяем ёмкость на эталонном 512x512 кадре.
    capacity = capacity_for_image(
        (OUTPUT_SIZE, OUTPUT_SIZE, 3),
        block_size=int(params["block_size"]),
        repetition=int(params["repetition"]),
    )
    logical_capacity = int(capacity["logical_capacity_bits"])

    if WATERMARK_BITS > logical_capacity:
        raise ValueError(
            f"ЦВЗ {WATERMARK_BITS} бит не помещается: "
            f"ёмкость профиля только {logical_capacity} бит."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    watermark_bits = rng.integers(
        0,
        2,
        size=WATERMARK_BITS,
        dtype=np.uint8,
    )
    write_bits(WATERMARK_PATH, watermark_bits)


    print(f"Изображений:       {len(input_images)}")
    print(f"Профиль:           {PROFILE_NAME}")
    print(f"Размер:            {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"Payload:           {WATERMARK_BITS} бит")
    print(f"Block size:        {params['block_size']}x{params['block_size']}")
    print(f"Repetition:        {params['repetition']}")
    print(f"Margin:            {params['margin']}")
    print(f"Logical capacity:  {logical_capacity} бит")
    print(f"Результаты:        {OUTPUT_ROOT}")
    print()

    rows: list[dict] = []
    failures: list[str] = []

    for index, input_path in enumerate(input_images, start=1):
        print(
            f"[{index:02d}/{len(input_images):02d}] "
            f"{input_path.name}"
        )

        stem = input_path.stem
        host_path = HOST_DIR / f"{stem}_host_512.png"
        work_watermarked_path = (
            WORK_DIR / f"{stem}_{PROFILE_NAME}.png"
        )
        print_ready_path = (
            PRINT_DIR / f"{stem}_{PROFILE_NAME}.png"
        )
        metadata_path = (
            METADATA_DIR
            / f"{stem}_{PROFILE_NAME}.block_watermark.json"
        )
        extracted_path = (
            EXTRACTED_DIR / f"{stem}_extracted.txt"
        )

        row = {
            "index": index,
            "source_file": input_path.name,
            "print_file": print_ready_path.name,
            "profile": PROFILE_NAME,
            "message_bits": WATERMARK_BITS,
            "block_size": params["block_size"],
            "repetition": params["repetition"],
            "margin": params["margin"],
            "logical_capacity_bits": logical_capacity,
            "digital_ber": "",
            "psnr_db": "",
            "ssim": "",
            "repetition_agreement_mean": "",
            "sha256": "",
            "status": "ERROR",
            "error": "",
        }

        try:
            source = read_image(input_path)
            host = prepare_square(source, OUTPUT_SIZE)
            save_image(host_path, host)

            embed_result = embed(
                image_path=host_path,
                watermark_path=WATERMARK_PATH,
                output_path=work_watermarked_path,
                metadata_path=metadata_path,
                profile=PROFILE_NAME,
            )

            watermarked = read_image(work_watermarked_path)
            psnr, ssim = image_quality(host, watermarked)

            diagnostics = extract(
                image_path=work_watermarked_path,
                output_path=extracted_path,
                metadata_path=metadata_path,
                offset_search_radius=0,
            )

            extracted_bits = read_bits(extracted_path)
            ber = calculate_ber(watermark_bits, extracted_bits)

            row["digital_ber"] = ber
            row["psnr_db"] = psnr
            row["ssim"] = ssim
            row["repetition_agreement_mean"] = diagnostics.get(
                "repetition_agreement_mean",
                "",
            )

            # В папку PRINT_READY попадают ТОЛЬКО файлы,
            # прошедшие цифровую проверку без единой ошибки.
            if ber == 0.0:
                shutil.copy2(
                    work_watermarked_path,
                    print_ready_path,
                )
                row["sha256"] = sha256_file(print_ready_path)
                row["status"] = "PASS"
                print(
                    f"    PASS | BER={ber:.6f} | "
                    f"PSNR={psnr:.3f} dB | "
                    f"SSIM={ssim:.6f}"
                )
            else:
                failures.append(
                    f"{input_path.name}: BER={ber:.6f}"
                )
                row["status"] = "FAIL_BER"
                print(
                    f"    FAIL | BER={ber:.6f} | "
                    "файл НЕ добавлен в PRINT_READY"
                )

            # Берём реальные значения из embed_result, чтобы зафиксировать
            # фактическую конфигурацию в CSV.
            row["block_size"] = embed_result["block_size"]
            row["repetition"] = embed_result["repetition"]
            row["margin"] = embed_result["margin"]
            row["logical_capacity_bits"] = (
                embed_result["logical_capacity_bits"]
            )

        except Exception as error:
            row["error"] = str(error)
            failures.append(
                f"{input_path.name}: ERROR: {error}"
            )
            print(f"    ERROR: {error}")

        rows.append(row)

    # ------------------------------------------------------------------------
    # CSV со всеми 50 изображениями
    # ------------------------------------------------------------------------
    fieldnames = [
        "index",
        "source_file",
        "print_file",
        "profile",
        "message_bits",
        "block_size",
        "repetition",
        "margin",
        "logical_capacity_bits",
        "digital_ber",
        "psnr_db",
        "ssim",
        "repetition_agreement_mean",
        "sha256",
        "status",
        "error",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    pass_rows = [
        row for row in rows
        if row["status"] == "PASS"
    ]

    psnr_values = [
        float(row["psnr_db"])
        for row in pass_rows
    ]
    ssim_values = [
        float(row["ssim"])
        for row in pass_rows
    ]

    manifest = {
        "run_id": RUN_ID,
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_ROOT),
        "expected_images": EXPECTED_IMAGE_COUNT,
        "processed_images": len(rows),
        "print_ready_images": len(pass_rows),
        "profile": PROFILE_NAME,
        "output_size_px": OUTPUT_SIZE,
        "watermark_bits": WATERMARK_BITS,
        "random_seed": RANDOM_SEED,
        "same_watermark_for_all_images": True,
        "block_size": int(params["block_size"]),
        "repetition": int(params["repetition"]),
        "margin": float(params["margin"]),
        "logical_capacity_bits": logical_capacity,
        "digital_control_requirement": "BER == 0",
        "mean_psnr_db": (
            float(np.mean(psnr_values))
            if psnr_values else None
        ),
        "min_psnr_db": (
            float(np.min(psnr_values))
            if psnr_values else None
        ),
        "mean_ssim": (
            float(np.mean(ssim_values))
            if ssim_values else None
        ),
        "min_ssim": (
            float(np.min(ssim_values))
            if ssim_values else None
        ),
        "all_passed": len(pass_rows) == EXPECTED_IMAGE_COUNT,
        "failures": failures,
    }

    MANIFEST_JSON.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Цифровую проверку прошли: "
        f"{len(pass_rows)}/{EXPECTED_IMAGE_COUNT}"
    )

    if psnr_values:
        print(
            f"Средний PSNR:              "
            f"{np.mean(psnr_values):.3f} dB"
        )
        print(
            f"Минимальный PSNR:          "
            f"{np.min(psnr_values):.3f} dB"
        )
        print(
            f"Средний SSIM:              "
            f"{np.mean(ssim_values):.6f}"
        )
        print(
            f"Минимальный SSIM:          "
            f"{np.min(ssim_values):.6f}"
        )

    print(f"CSV:                       {SUMMARY_CSV}")
    print(f"Manifest:                  {MANIFEST_JSON}")
    print(f"ЦВЗ:                       {WATERMARK_PATH}")
    print(f"Готовые PNG:               {PRINT_DIR}")

    if len(pass_rows) == EXPECTED_IMAGE_COUNT:
        print()
        print("ВСЕ 50 ИЗОБРАЖЕНИЙ ПРОШЛИ ПРОВЕРКУ.")
        print(
            "На печать отправляй ТОЛЬКО содержимое папки PRINT_READY."
        )
        print(
            "Эту же папку можно отправить руководителю как точные "
            "цифровые оригиналы файлов, переданных на печать."
        )
    else:
        print()
        print("НЕ ОТПРАВЛЯЙ НА ПЕЧАТЬ: есть ошибки.")
        print("Проблемные изображения:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
