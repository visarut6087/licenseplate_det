
from pathlib import Path
import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "test2.mp4"
RUNS_DIR = ROOT / "runs" / "detect"

OUTPUT_DIR = ROOT / "results" / "video_test"
CROP_DIR = OUTPUT_DIR / "plate_crops"
DEBUG_DIR = OUTPUT_DIR / "debug_all_classes"

CONFIDENCE = 0.25
IMAGE_SIZE = 1080
DEVICE = 0

# ชื่อ Class ของกรอบป้ายทะเบียนตาม data.yaml
PLATE_CLASS_ID = 0

# จับภาพทุก # วินาที
SAMPLE_INTERVAL_SECONDS = 1

CROP_PADDING = 0.05
MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 20
JPEG_QUALITY = 95

DEBUG_ALL_CLASSES = True


# ============================================================
# FIND MODEL
# ============================================================

def find_latest_model():

    candidates = [
        p for p in RUNS_DIR.glob("**/weights/best.pt")
        if p.is_file() and p.stat().st_size > 0
    ]

    if not candidates:
        raise FileNotFoundError(
            f"ไม่พบ best.pt ใน {RUNS_DIR}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ============================================================
# GET PLATE CLASS
# ============================================================

def get_plate_class(model):

    print("\n========== MODEL CLASSES ==========")

    for class_id, class_name in model.names.items():
        print(f"ID: {class_id} | Name: {class_name}")

    if PLATE_CLASS_ID not in model.names:
        raise ValueError(
            f"ไม่พบ Class ID {PLATE_CLASS_ID} ในโมเดล"
        )

    return PLATE_CLASS_ID



# ============================================================
# CROP
# ============================================================

def crop_plate(frame, xyxy):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, xyxy)

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = int(box_width * CROP_PADDING)
    pad_y = int(box_height * CROP_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
        return None

    return crop


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== VIDEO PLATE TEST ==========")
    print("Video:", VIDEO_PATH)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    model_path = find_latest_model()

    print("Model:", model_path)

    model = YOLO(str(model_path))
    plate_class = get_plate_class(model)

    print("Plate class ID:", plate_class)

    CROP_DIR.mkdir(parents=True, exist_ok=True)

    if DEBUG_ALL_CLASSES:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("เปิดวิดีโอไม่ได้")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    frame_step = max(
        1,
        round(fps * SAMPLE_INTERVAL_SECONDS)
    )

    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Sample interval:", SAMPLE_INTERVAL_SECONDS, "seconds")

    # Preview เล่นตามช่วงเวลาที่สุ่มจริง
    preview_path = OUTPUT_DIR / "preview.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / SAMPLE_INTERVAL_SECONDS,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("สร้าง preview.mp4 ไม่สำเร็จ")

    frame_number = 0
    sample_count = 0
    detection_count = 0
    saved_count = 0
    debug_count = 0

    class_counts = {}

    try:

        while frame_number < total_frames:

            # กระโดดไปยังเฟรมตามช่วง 5 วินาที
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()

            if not ret:
                break

            sample_count += 1

            timestamp = frame_number / fps

            print(
                f"\n[SAMPLE {sample_count}] "
                f"Frame={frame_number} "
                f"Time={timestamp:.1f}s"
            )

            # ตรวจจับทุก Class เพื่อดูผล
            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            result = results[0]
            preview = frame.copy()

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes
                xyxys = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for index, (xyxy, class_id, confidence) in enumerate(
                    zip(xyxys, classes, confidences)
                ):

                    detection_count += 1

                    class_counts[class_id] = (
                        class_counts.get(class_id, 0) + 1
                    )

                    x1, y1, x2, y2 = map(int, xyxy)

                    class_name = str(model.names[class_id])

                    is_plate = class_id == plate_class

                    color = (
                        (0, 255, 0)
                        if is_plate
                        else (255, 180, 0)
                    )

                    cv2.rectangle(
                        preview,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        preview,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    print(
                        f"[DETECT] Class={class_id} "
                        f"Name={class_name} "
                        f"Conf={confidence:.3f} "
                        f"Box={x2-x1}x{y2-y1}"
                    )

                    # Debug: เก็บภาพของทุก Class
                    if DEBUG_ALL_CLASSES and debug_count < 300:

                        debug_crop = crop_plate(
                            frame,
                            xyxy
                        )

                        if debug_crop is not None:

                            debug_path = DEBUG_DIR / (
                                f"t{int(timestamp):06d}_"
                                f"class_{class_id}_"
                                f"{index:03d}.jpg"
                            )

                            if cv2.imwrite(
                                str(debug_path),
                                debug_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                            ):
                                debug_count += 1

                    # Crop เฉพาะกรอบป้ายทะเบียน
                    if not is_plate:
                        continue

                    crop = crop_plate(frame, xyxy)

                    if crop is None:
                        print("[CROP REJECTED] กรอบเล็กเกินไป")
                        continue

                    filename = (
                        f"plate_{sample_count:05d}_"
                        f"t{int(timestamp):06d}_"
                        f"{index:03d}.jpg"
                    )

                    output_path = CROP_DIR / filename

                    success = cv2.imwrite(
                        str(output_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )

                    if success:
                        saved_count += 1

                        print(
                            f"[PLATE SAVED] {filename} "
                            f"| {crop.shape[1]}x{crop.shape[0]}"
                        )

            cv2.putText(
                preview,
                f"Time: {timestamp:.1f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            writer.write(preview)

            frame_number += frame_step

    finally:
        cap.release()
        writer.release()

    print("\n========== SUMMARY ==========")
    print("Sampled frames:", sample_count)
    print("Total detections:", detection_count)
    print("Saved plate crops:", saved_count)
    print("Debug crops:", debug_count)

    print("\nDetection by class:")

    for class_id, count in sorted(class_counts.items()):
        print(
            f"Class {class_id} "
            f"({model.names[class_id]}): {count}"
        )

    print("\nCrop directory:", CROP_DIR)
    print("Debug directory:", DEBUG_DIR)
    print("Preview:", preview_path)


if __name__ == "__main__":
    main()
