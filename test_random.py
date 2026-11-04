
from pathlib import Path
import random
import shutil
import json

import cv2
from ultralytics import YOLO


# =========================
# Project configuration
# =========================
ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "datasets" / "test" / "images"
RUNS_DIR = ROOT / "runs" / "detect"
RESULTS = ROOT / "results"

# จาก label ตัวอย่างที่ส่งมา class 2 มีพิกัดครอบคลุมแผ่นป้าย
# กรุณาตรวจสอบกับผลการฝึกก่อนใช้งานจริง
PLATE_CLASS_ID = 2
CONFIDENCE = 0.25
DEVICE = 0  

def find_latest_best():
    candidates = list(RUNS_DIR.glob("v*_train/weights/best.pt"))

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
    return cv2.imwrite(str(destination), crop)


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {IMAGE_DIR}")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"ไม่พบโมเดล: {WEIGHTS}\n"
            "ตรวจสอบว่า train เสร็จและมีไฟล์ best.pt แล้ว"
        )

    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        raise RuntimeError(f"ไม่พบภาพใน {IMAGE_DIR}")

    # สุ่มภาพหนึ่งภาพ
    selected = random.choice(images)

    # สร้างโฟลเดอร์ผลลัพธ์สำหรับรอบนี้
    run_dir = RESULTS / selected.stem
    input_dir = run_dir / "input"
    plate_dir = run_dir / "plate_crop"
    chars_dir = run_dir / "characters"

    for folder in (input_dir, plate_dir, chars_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # คัดลอกไฟล์ต้นฉบับโดยไม่ re-encode หรือเปลี่ยนพิกเซล
    original_copy = input_dir / selected.name
    shutil.copy2(selected, original_copy)

    # อ่านภาพสำหรับ inference และ crop
    image = cv2.imread(str(selected))
    if image is None:
        raise RuntimeError(f"อ่านภาพไม่ได้: {selected}")

    print(f"Input: {selected}")
    print(f"Output: {run_dir}")

    model = YOLO(str(WEIGHTS))

    predictions = model.predict(
        source=str(selected),
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False,
    )

    result = predictions[0]
    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names.get(
                class_id, str(class_id)
            )

            detections.append({
                "index": i,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": confidence,
                "xyxy": xyxy,
            })

    # เก็บข้อมูลผลตรวจจับไว้ตรวจสอบย้อนหลัง
    metadata = {
        "source_image": str(selected.relative_to(ROOT)),
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "plate_class_id": PLATE_CLASS_ID,
        "confidence_threshold": CONFIDENCE,
        "detections": detections,
    }

    (run_dir / "detections.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # วาดภาพ preview แยกต่างหาก ไม่แตะ input ต้นฉบับ
    preview = result.plot()
    cv2.imwrite(str(run_dir / "preview.jpg"), preview)

    # เลือกกรอบที่เป็นป้ายทะเบียน
    plate_detections = [
        d for d in detections
        if d["class_id"] == PLATE_CLASS_ID
    ]

    if not plate_detections:
        print(
            "\nไม่พบ class ป้ายทะเบียน "
            f"(class_id={PLATE_CLASS_ID})"
        )
        print("โปรดตรวจ detections.json และ preview.jpg")
        print("อาจต้องแก้ PLATE_CLASS_ID ให้ตรงกับโมเดล")
        return

    # ถ้าพบหลายกรอบ เลือกกรอบที่มีพื้นที่มากที่สุด
    plate = max(
        plate_detections,
        key=lambda d: (
            (d["xyxy"][2] - d["xyxy"][0])
            * (d["xyxy"][3] - d["xyxy"][1])
        ),
    )

    plate_box = plate["xyxy"]
    save_crop(
        image,
        plate_box,
        plate_dir / f"{selected.stem}_plate.jpg",
    )

    # Crop อักขระจาก bounding boxes ที่เหลือ
    # เรียงจากซ้ายไปขวา แยกตามตำแหน่งแนวตั้งโดยประมาณ
    plate_x1, plate_y1, plate_x2, plate_y2 = plate_box

    char_detections = [
        d for d in detections
        if d["class_id"] != PLATE_CLASS_ID
        and (
            d["xyxy"][0] >= plate_x1
            and d["xyxy"][2] <= plate_x2
            and d["xyxy"][1] >= plate_y1
            and d["xyxy"][3] <= plate_y2
        )
    ]

    # เรียงบนลงล่างก่อน แล้วซ้ายไปขวาในแถวเดียวกัน
    char_detections.sort(
        key=lambda d: (
            round((d["xyxy"][1] + d["xyxy"][3]) / 2 / 20),
            d["xyxy"][0],
        )
    )

    for number, detection in enumerate(char_detections, start=1):
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        filename = (
            f"{number:02d}_class-{class_id}_"
            f"{class_name}_conf-{confidence:.2f}.jpg"
        )

        save_crop(
            image,
            detection["xyxy"],
            chars_dir / filename,
        )

    print("\nเสร็จแล้ว")
    print(f"ภาพต้นฉบับ: {original_copy}")
    print(f"ภาพ preview: {run_dir / 'preview.jpg'}")
    print(f"Crop ป้าย: {plate_dir}")
    print(f"Crop อักขระ: {chars_dir}")
    print(f"จำนวนกรอบทั้งหมด: {len(detections)}")
    print(f"จำนวนอักขระที่ crop: {len(char_detections)}")


if __name__ == "__main__":
    main()
