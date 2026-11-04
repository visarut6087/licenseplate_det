
from pathlib import Path
import random
import shutil
import json
import re

import cv2
from ultralytics import YOLO


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "datasets" / "test" / "images"
RUNS_DIR = ROOT / "runs" / "detect"
RESULTS = ROOT / "results"

# จำนวนภาพที่ต้องการสุ่มทดสอบ
TEST_IMAGE_COUNT = 20

# Model configuration
PLATE_CLASS_ID = 0
CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = 0

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =========================================================
# FIND LATEST MODEL
# =========================================================

def find_latest_best():
    candidates = list(
        RUNS_DIR.glob("v*_train/weights/best.pt")
    )

    candidates = [
        path for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]

    if not candidates:
        raise FileNotFoundError(
            f"ไม่พบ best.pt ใน {RUNS_DIR}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime
    )


WEIGHTS = find_latest_best()


# =========================================================
# SAVE CROP
# =========================================================

def save_crop(image, box, destination):
    """Crop ตาม bounding box โดยไม่แก้ไขภาพต้นฉบับ"""

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return cv2.imwrite(
        str(destination),
        crop
    )


# =========================================================
# SAFE FILENAME
# =========================================================

def safe_filename(value):
    return re.sub(
        r'[<>:"/\\|?*]',
        "_",
        str(value)
    )


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(selected, model, model_names):

    print("\n" + "-" * 60)
    print(f"Input: {selected.name}")

    # Output folders
    run_dir = RESULTS / selected.stem

    input_dir = run_dir / "input"
    plate_dir = run_dir / "plate_crop"
    chars_dir = run_dir / "characters"

    for folder in (
        input_dir,
        plate_dir,
        chars_dir
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # Copy original image
    original_copy = input_dir / selected.name

    shutil.copy2(
        selected,
        original_copy
    )

    # Read original image
    image = cv2.imread(str(selected))

    if image is None:
        raise RuntimeError(
            f"อ่านภาพไม่ได้: {selected}"
        )

    # Inference
    predictions = model.predict(
        source=str(selected),
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    result = predictions[0]

    detections = []

    # Extract detections
    if result.boxes is not None:

        for i, box in enumerate(result.boxes):

            xyxy = box.xyxy[0].cpu().tolist()

            class_id = int(box.cls[0].item())

            confidence = float(
                box.conf[0].item()
            )

            class_name = model_names.get(
                class_id,
                str(class_id)
            )

            detections.append({
                "index": i,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": round(confidence, 6),
                "xyxy": [
                    round(float(value), 2)
                    for value in xyxy
                ],
            })

    # Find plate detections
    plate_detections = [
        detection
        for detection in detections
        if detection["class_id"] == PLATE_CLASS_ID
    ]

    # Metadata
    metadata = {
        "source_image": str(
            selected.relative_to(ROOT)
        ),
        "weights": str(
            WEIGHTS.relative_to(ROOT)
        ),
        "plate_class_id": PLATE_CLASS_ID,
        "confidence_threshold": CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "summary": {
            "total_detections": len(detections),
            "plate_detections": len(plate_detections),
            "character_detections": 0,
            "plate_found": len(plate_detections) > 0,
        },
        "detections": detections,
        "selected_plate": None,
        "characters": [],
    }

    # Save preview
    preview = result.plot()

    preview_path = run_dir / "preview.jpg"

    if not cv2.imwrite(
        str(preview_path),
        preview
    ):
        raise IOError(
            f"บันทึก Preview ไม่สำเร็จ: {preview_path}"
        )

    # No plate found
    if not plate_detections:

        (run_dir / "detections.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8",
        )

        print(
            f"ไม่พบป้ายทะเบียน "
            f"(class_id={PLATE_CLASS_ID})"
        )

        print(f"ตรวจสอบ Preview: {preview_path}")

        return {
            "image": selected.name,
            "detections": len(detections),
            "plate_found": False,
            "characters": 0,
        }

    # Select largest plate
    plate = max(
        plate_detections,
        key=lambda detection: (
            (detection["xyxy"][2] - detection["xyxy"][0])
            * (detection["xyxy"][3] - detection["xyxy"][1])
        ),
    )

    plate_box = plate["xyxy"]

    # Crop plate
    plate_output = (
        plate_dir /
        f"{selected.stem}_plate.jpg"
    )

    plate_saved = save_crop(
        image,
        plate_box,
        plate_output
    )

    # Get plate coordinates
    plate_x1, plate_y1, plate_x2, plate_y2 = plate_box

    # Select characters inside plate
    char_detections = [
        detection
        for detection in detections
        if detection["class_id"] != PLATE_CLASS_ID
        and detection["xyxy"][0] >= plate_x1
        and detection["xyxy"][2] <= plate_x2
        and detection["xyxy"][1] >= plate_y1
        and detection["xyxy"][3] <= plate_y2
    ]

    # Sort top-to-bottom and left-to-right
    char_detections.sort(
        key=lambda detection: (
            round(
                (
                    detection["xyxy"][1]
                    + detection["xyxy"][3]
                ) / 2 / 20
            ),
            detection["xyxy"][0],
        )
    )

    saved_characters = []

    # Crop characters
    for number, detection in enumerate(
        char_detections,
        start=1
    ):

        class_id = detection["class_id"]
        class_name = safe_filename(
            detection["class_name"]
        )
        confidence = detection["confidence"]

        filename = (
            f"{number:02d}_"
            f"class-{class_id}_"
            f"{class_name}_"
            f"conf-{confidence:.2f}.jpg"
        )

        crop_path = chars_dir / filename

        saved = save_crop(
            image,
            detection["xyxy"],
            crop_path
        )

        if saved:

            saved_characters.append({
                "order": number,
                "class_id": class_id,
                "class_name": detection["class_name"],
                "confidence": confidence,
                "crop_path": str(
                    crop_path.relative_to(ROOT)
                ),
            })

    # Update metadata
    metadata["summary"]["character_detections"] = len(
        saved_characters
    )

    metadata["selected_plate"] = {
        "class_id": plate["class_id"],
        "class_name": plate["class_name"],
        "confidence": plate["confidence"],
        "bbox": plate_box,
        "crop_path": (
            str(plate_output.relative_to(ROOT))
            if plate_saved else None
        ),
    }

    metadata["characters"] = saved_characters

    # Save JSON
    json_path = run_dir / "detections.json"

    json_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    print(f"Detection ทั้งหมด: {len(detections)}")
    print(f"พบป้ายทะเบียน: {len(plate_detections)}")
    print(f"Crop ป้ายสำเร็จ: {plate_saved}")
    print(f"Crop อักขระสำเร็จ: {len(saved_characters)}")

    print(f"Preview: {preview_path}")
    print(f"Plate: {plate_dir}")
    print(f"Characters: {chars_dir}")
    print(f"JSON: {json_path}")

    return {
        "image": selected.name,
        "detections": len(detections),
        "plate_found": True,
        "characters": len(saved_characters),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    # Check image directory
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ภาพ: {IMAGE_DIR}"
        )

    # Check model
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"ไม่พบโมเดล: {WEIGHTS}"
        )

    # Collect images
    images = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError(
            f"ไม่พบภาพใน {IMAGE_DIR}"
        )

    # Random sampling without replacement
    sample_count = min(
        TEST_IMAGE_COUNT,
        len(images)
    )

    selected_images = random.sample(
        images,
        sample_count
    )

    print("=" * 60)
    print("MOTORCYCLE LICENSE PLATE DETECTION")
    print("=" * 60)

    print(f"Model: {WEIGHTS}")
    print(f"Input directory: {IMAGE_DIR}")
    print(f"Requested images: {TEST_IMAGE_COUNT}")
    print(f"Selected images: {sample_count}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")

    # Load model once
    print("\nLoading YOLO model...")

    model = YOLO(str(WEIGHTS))

    model_names = model.names

    print("Model loaded successfully")

    # Process images
    summary = []

    for index, selected in enumerate(
        selected_images,
        start=1
    ):

        print(
            f"\n[{index}/{sample_count}]"
        )

        try:

            result = process_image(
                selected,
                model,
                model_names
            )

            if result is not None:
                summary.append(result)

        except Exception as error:

            print(
                f"ERROR: {selected.name}: {error}"
            )

            summary.append({
                "image": selected.name,
                "error": str(error),
            })

    # Save overall summary
    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = RESULTS / "test_summary.json"

    summary_data = {
        "model": str(WEIGHTS.relative_to(ROOT)),
        "requested_images": TEST_IMAGE_COUNT,
        "selected_images": sample_count,
        "processed_images": len(summary),
        "successful_detections": sum(
            1 for item in summary
            if item.get("plate_found") is True
        ),
        "results": summary,
    }

    summary_path.write_text(
        json.dumps(
            summary_data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    # Final report
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)

    print(f"Selected images: {sample_count}")
    print(f"Processed images: {len(summary)}")

    print(
        "Plate detected:",
        summary_data["successful_detections"]
    )

    print(f"Summary: {summary_path}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
