
from pathlib import Path
import json
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# ==================================================
# CONFIG
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "all_ocr_results.json"

BASE_MODEL = "scb10x/typhoon-ocr1.5-2b"
LORA_MODEL = "Rattatammanoon/hurricane-ocr-tlpr-v1-LoRA"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Image preprocessing
MAX_IMAGE_WIDTH = 400

# Generation
MAX_NEW_TOKENS = 160
NUM_BEAMS = 3

# Resume
SKIP_COMPLETED = True


# ==================================================
# THAI MOTORCYCLE PLATE CONTEXT
# ==================================================

SYSTEM_CONTEXT = """
คุณเป็นระบบ OCR สำหรับอ่านป้ายทะเบียนรถจักรยานยนต์ในประเทศไทย

หน้าที่ของคุณคืออ่านข้อความจากภาพป้ายทะเบียนที่ได้รับเท่านั้น

บริบทเกี่ยวกับป้ายทะเบียนรถจักรยานยนต์ไทย:

1. ป้ายทะเบียนอาจประกอบด้วยตัวอักษรภาษาไทยและตัวเลขอารบิก
2. ตัวอักษรไทยมักอยู่บริเวณด้านบนหรือด้านหน้าของชุดตัวเลข
3. เลขทะเบียนอาจมีตั้งแต่ 1 ถึง 4 หลัก
4. บริเวณด้านล่างของป้ายอาจมีชื่อจังหวัด
5. รูปแบบป้ายอาจแตกต่างกันตามประเภทรถและปีที่จดทะเบียน
6. ภาพอาจมีมุมเอียง แสงสะท้อน ความเบลอ หรือความละเอียดต่ำ

กฎการอ่าน:

- อ่านเฉพาะสิ่งที่มองเห็นจริงในภาพ
- ห้ามเดาตัวอักษรหรือตัวเลขที่มองไม่เห็น
- ห้ามเติมตัวอักษรให้ครบตามรูปแบบทะเบียน
- ห้ามใช้ความรู้เกี่ยวกับทะเบียนที่ถูกต้องมาแทนข้อความที่อ่านไม่ได้
- ต้องแยกตัวอักษรไทย ตัวเลข และจังหวัด
- จังหวัดต้องอ่านจากข้อความบนป้าย ไม่ใช่คาดเดาจากตัวอักษรทะเบียน
- ถ้าอ่านไม่ได้ให้ระบุว่าอ่านไม่ชัด
- ถ้าอ่านได้เพียงบางส่วน ให้เก็บเฉพาะส่วนที่อ่านได้
- ห้ามแปลหรือเปลี่ยนตัวอักษรไทยเป็นภาษาอังกฤษ

ตอบตามรูปแบบที่กำหนดเท่านั้น
"""


OCR_QUERY = """
อ่านข้อความจากป้ายทะเบียนรถจักรยานยนต์ไทยในภาพนี้

ให้ตรวจสอบข้อมูลต่อไปนี้:

1. Plate Number:
   ข้อความทะเบียนทั้งหมดที่อ่านได้

2. Characters:
   ตัวอักษรภาษาไทยที่อยู่บนป้ายทะเบียน

3. Digits:
   ตัวเลขทะเบียนที่มองเห็น

4. Province:
   ชื่อจังหวัดที่พิมพ์อยู่บนป้าย

5. Unreadable:
   ระบุส่วนที่อ่านไม่ชัด หากมี

รูปแบบคำตอบ:

**Plate Number:** ...
**Characters:** ...
**Digits:** ...
**Province:** ...
**Unreadable:** ...

ข้อกำหนด:
- รักษาข้อความตามที่ปรากฏในภาพ
- ห้ามเดาข้อมูลที่ไม่ชัด
- ถ้าไม่สามารถอ่าน Field ใดได้ ให้ตอบว่า อ่านไม่ชัด
- ไม่ต้องอธิบายเพิ่มเติม
"""


# ==================================================
# LOAD MODEL
# ==================================================

def load_ocr_model():

    print("\n========== LOAD OCR MODEL ==========")

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print("Loading base model...")

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL
    )

    model.eval()

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM allocated:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "GB"
        )

    print("Model loaded successfully")

    return processor, model


# ==================================================
# FIND PLATE CROPS
# ==================================================

def find_plate_images():

    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์: {RESULTS_DIR}"
        )

    # รองรับทั้งชื่อเก่าและชื่อจาก Video Detection
    crop_dir_names = {
        "plate_crop",
        "plate_crops",
    }

    crop_dirs = sorted(
        p
        for p in RESULTS_DIR.rglob("*")
        if p.is_dir() and p.name in crop_dir_names
    )

    tasks = []

    for crop_dir in crop_dirs:

        car_id = crop_dir.parent.name

        for image_path in sorted(crop_dir.rglob("*")):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in EXTENSIONS
            ):

                tasks.append({
                    "car_id": car_id,
                    "image_path": image_path,
                })

    print("\n========== INPUT IMAGES ==========")
    print("Crop directories:", len(crop_dirs))
    print("Total images:", len(tasks))

    return tasks


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_path):

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    original_size = image.size

    width, height = image.size

    # ไม่ขยายภาพเล็ก เพราะอาจเพิ่ม noise
    if width > MAX_IMAGE_WIDTH:

        ratio = MAX_IMAGE_WIDTH / width

        new_size = (
            MAX_IMAGE_WIDTH,
            max(1, int(height * ratio))
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    return image, original_size


# ==================================================
# PARSE OCR OUTPUT
# ==================================================

def parse_ocr_output(text):

    fields = {
        "plate_number": None,
        "characters": None,
        "digits": None,
        "province": None,
        "unreadable": None,
    }

    label_mapping = {
        "plate number": "plate_number",
        "characters": "characters",
        "digits": "digits",
        "province": "province",
        "unreadable": "unreadable",
    }

    for line in text.splitlines():

        line = line.strip()

        match = re.match(
            r"\**\s*(Plate Number|Characters|Digits|Province|Unreadable)"
            r"\s*\**\s*:\s*(.*)",
            line,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        label = match.group(1).lower()
        value = match.group(2).strip()

        key = label_mapping.get(label)

        if key and value:
            fields[key] = value

    return fields


# ==================================================
# OCR ONE IMAGE
# ==================================================

def recognize_plate(image_path, processor, model):

    image, original_size = preprocess_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": (
                        SYSTEM_CONTEXT
                        + "\n\n"
                        + OCR_QUERY
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
            early_stopping=True,
        )

    prompt_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, prompt_length:]

    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    structured = parse_ocr_output(raw_text)

    return {
        "raw_text": raw_text,
        "structured": structured,
        "original_size": {
            "width": original_size[0],
            "height": original_size[1],
        },
        "processed_size": {
            "width": image.width,
            "height": image.height,
        },
    }


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    temp_file.replace(OUTPUT_FILE)


# ==================================================
# LOAD PREVIOUS RESULTS
# ==================================================

def load_previous_results():

    if not SKIP_COMPLETED or not OUTPUT_FILE.is_file():
        return [], set()

    try:

        previous = json.loads(
            OUTPUT_FILE.read_text(encoding="utf-8")
        )

        completed = {
            item["image"]
            for item in previous
            if item.get("error") is None
        }

        print("Previous results:", len(previous))
        print("Already completed:", len(completed))

        return previous, completed

    except (json.JSONDecodeError, OSError):

        print("Warning: Previous result file is invalid")
        return [], set()


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n========== THAI MOTORCYCLE OCR ==========")

    tasks = find_plate_images()

    if not tasks:
        print("ไม่พบภาพสำหรับ OCR")
        return

    all_results, completed_images = load_previous_results()

    processor, model = load_ocr_model()

    total = len(tasks)

    success_count = 0
    error_count = 0

    for index, task in enumerate(tasks, start=1):

        image_path = task["image_path"]

        relative_path = str(
            image_path.relative_to(RESULTS_DIR)
        )

        if relative_path in completed_images:

            print(
                f"[{index}/{total}] SKIP | {relative_path}"
            )
            continue

        start = time.perf_counter()

        try:

            result = recognize_plate(
                image_path,
                processor,
                model
            )

            error = None
            success_count += 1

        except Exception as exc:

            result = {
                "raw_text": "",
                "structured": {
                    "plate_number": None,
                    "characters": None,
                    "digits": None,
                    "province": None,
                    "unreadable": None,
                },
                "original_size": None,
                "processed_size": None,
            }

            error = f"{type(exc).__name__}: {exc}"

            error_count += 1

        elapsed = round(
            time.perf_counter() - start,
            3
        )

        record = {
            "car_id": task["car_id"],
            "image": relative_path,
            "ocr_text": result["raw_text"],
            "structured": result["structured"],
            "original_size": result["original_size"],
            "processed_size": result["processed_size"],
            "seconds": elapsed,
            "error": error,
        }

        all_results.append(record)

        # บันทึกทุกภาพ ป้องกันข้อมูลสูญหาย
        save_results(all_results)

        fields = result["structured"]

        print(
            f"\n[{index}/{total}] "
            f"{task['car_id']}"
        )

        print("Plate:", fields["plate_number"])
        print("Characters:", fields["characters"])
        print("Digits:", fields["digits"])
        print("Province:", fields["province"])
        print("Time:", elapsed, "seconds")

        if error:
            print("ERROR:", error)

    print("\n========== SUMMARY ==========")
    print("Total images:", total)
    print("Successful:", success_count)
    print("Errors:", error_count)
    print("Total records:", len(all_results))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
