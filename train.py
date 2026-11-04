
from pathlib import Path
import re

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "data2.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

EPOCHS = 100
IMG_SIZE = 720
BATCH_SIZE = 16
DEVICE = 0


def get_next_version():
    versions = []

    for path in RUNS_DIR.glob("v*_train"):
        match = re.fullmatch(r"v(\d+)_train", path.name)
        if match and path.is_dir():
            versions.append(int(match.group(1)))

    return max(versions, default=0) + 1


def get_previous_last_weight():
    candidates = []

    for path in RUNS_DIR.glob("v*_train/weights/last.pt"):
        match = re.fullmatch(
            r"v(\d+)_train",
            path.parent.parent.name,
        )
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"ไม่พบ Dataset: {DATA_YAML}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    version = get_next_version()
    run_name = f"v{version}_train"

    previous_weight = get_previous_last_weight()

    if previous_weight:
        model_path = previous_weight
        print(f"เริ่มจาก Weight ล่าสุด: {model_path}")
    else:
        model_path = "yolov8n.pt"
        print("เริ่ม Version แรกจาก yolov8n.pt")

    print(f"Version ใหม่: {run_name}")

    model = YOLO(str(model_path))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        workers=4,
        patience=0,
        save=True,
        plots=True,
    )

    print("Training เสร็จแล้ว")
    print(f"Best weight: {RUNS_DIR / run_name / 'weights' / 'best.pt'}")
    print(f"Last weight: {RUNS_DIR / run_name / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
