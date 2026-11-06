
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEVICE = 0
PLATE_CLASS_ID = 0

DATASET_YAML = Path(
    r"C:\patrick\licenseplates_det\datasets\data2.yaml"
)

TRACKER = "bytetrack.yaml"


class PlateTrackingApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Motorcycle License Plate Tracking")
        self.root.geometry("1250x820")

        self.model = None
        self.cap = None

        self.video_path = None
        self.model_path = None

        self.running = False
        self.paused = False

        self.confidence = tk.DoubleVar(
            value=DEFAULT_CONFIDENCE
        )

        self.image_size = tk.IntVar(
            value=DEFAULT_IMAGE_SIZE
        )

        self.total_detections = 0
        self.active_ids = set()
        self.seen_ids = set()

        self.setup_ui()

    # ========================================================
    # UI
    # ========================================================

    def setup_ui(self):

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(
            top,
            text="Select Video",
            command=self.select_video
        ).pack(side="left", padx=5)

        ttk.Button(
            top,
            text="Select YOLO Model",
            command=self.select_model
        ).pack(side="left", padx=5)

        ttk.Button(
            top,
            text="Start Tracking",
            command=self.start_tracking
        ).pack(side="left", padx=5)

        ttk.Button(
            top,
            text="Pause / Resume",
            command=self.toggle_pause
        ).pack(side="left", padx=5)

        ttk.Button(
            top,
            text="Stop",
            command=self.stop_tracking
        ).pack(side="left", padx=5)

        settings = ttk.LabelFrame(
            self.root,
            text="Tracking Settings",
            padding=10
        )
        settings.pack(fill="x", padx=10, pady=5)

        ttk.Label(
            settings,
            text="Confidence"
        ).pack(side="left")

        ttk.Scale(
            settings,
            from_=0.05,
            to=0.95,
            variable=self.confidence,
            orient="horizontal",
            length=180
        ).pack(side="left", padx=10)

        ttk.Label(
            settings,
            text="Image Size"
        ).pack(side="left", padx=10)

        ttk.Combobox(
            settings,
            textvariable=self.image_size,
            values=[320, 480, 640, 960, 1280],
            width=8,
            state="readonly"
        ).pack(side="left")

        self.video_label = ttk.Label(
            self.root,
            text="Select video and YOLO model",
            anchor="center"
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        info = ttk.LabelFrame(
            self.root,
            text="Tracking Information",
            padding=10
        )
        info.pack(fill="x", padx=10, pady=5)

        self.status_label = ttk.Label(
            info,
            text="Status: Waiting"
        )
        self.status_label.pack(anchor="w")

        self.id_label = ttk.Label(
            info,
            text="Active Tracking IDs: 0"
        )
        self.id_label.pack(anchor="w")

        self.total_label = ttk.Label(
            info,
            text="Total Unique IDs: 0"
        )
        self.total_label.pack(anchor="w")

        self.detection_label = ttk.Label(
            info,
            text="Total Detections: 0"
        )
        self.detection_label.pack(anchor="w")

    # ========================================================
    # SELECT FILES
    # ========================================================

    def select_video(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.video_path = path
            self.status_label.config(
                text=f"Video: {Path(path).name}"
            )

    def select_model(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("YOLO weights", "*.pt")
            ]
        )

        if not path:
            return

        try:
            model = YOLO(path)

            if PLATE_CLASS_ID not in model.names:
                raise ValueError(
                    f"ไม่พบ Class ID {PLATE_CLASS_ID}"
                )

            self.model = model
            self.model_path = path

            self.status_label.config(
                text=f"Model: {Path(path).name}"
            )

            messagebox.showinfo(
                "Model Loaded",
                f"Loaded successfully\n{path}"
            )

        except Exception as e:

            messagebox.showerror(
                "Model Error",
                str(e)
            )

    # ========================================================
    # START TRACKING
    # ========================================================

    def start_tracking(self):

        if self.running:
            return

        if not self.video_path:
            messagebox.showwarning(
                "Warning",
                "Please select a video"
            )
            return

        if self.model is None:
            messagebox.showwarning(
                "Warning",
                "Please select YOLO model"
            )
            return

        self.cap = cv2.VideoCapture(
            self.video_path
        )

        if not self.cap.isOpened():
            messagebox.showerror(
                "Error",
                "Cannot open video"
            )
            return

        self.running = True
        self.paused = False

        self.total_detections = 0
        self.active_ids.clear()
        self.seen_ids.clear()

        self.status_label.config(
            text="Status: Tracking"
        )

        self.update_frame()

    # ========================================================
    # TRACKING LOOP
    # ========================================================

    def update_frame(self):

        if not self.running:
            return

        if self.paused:
            self.root.after(100, self.update_frame)
            return

        ret, frame = self.cap.read()

        if not ret:
            self.stop_tracking()
            return

        result = self.model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=float(self.confidence.get()),
            imgsz=int(self.image_size.get()),
            device=DEVICE,
            classes=[PLATE_CLASS_ID],
            verbose=False
        )

        self.active_ids = set()

        if result.boxes is not None:

            boxes = result.boxes

            if boxes.id is not None:

                ids = boxes.id.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().tolist()

                for track_id, xyxy, conf in zip(
                    ids, xyxys, confs
                ):

                    self.active_ids.add(track_id)
                    self.seen_ids.add(track_id)

                    self.total_detections += 1

                    x1, y1, x2, y2 = map(
                        int, xyxy
                    )

                    color = (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = (
                        f"Plate ID: {track_id} "
                        f"| Conf: {conf:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2
                    )

        self.update_information()

        # Resize for GUI display only
        display = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(display)
        image.thumbnail((1150, 620))

        photo = ImageTk.PhotoImage(image)

        self.video_label.configure(
            image=photo,
            text=""
        )

        self.video_label.image = photo

        self.root.after(
            1,
            self.update_frame
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def update_information(self):

        self.id_label.config(
            text=f"Active Tracking IDs: {len(self.active_ids)}"
        )

        self.total_label.config(
            text=f"Total Unique IDs: {len(self.seen_ids)}"
        )

        self.detection_label.config(
            text=f"Total Detections: {self.total_detections}"
        )

    # ========================================================
    # CONTROLS
    # ========================================================

    def toggle_pause(self):

        if not self.running:
            return

        self.paused = not self.paused

        self.status_label.config(
            text="Status: Paused"
            if self.paused
            else "Status: Tracking"
        )

    def stop_tracking(self):

        self.running = False
        self.paused = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.status_label.config(
            text="Status: Stopped"
        )

    def close(self):

        self.stop_tracking()
        self.root.destroy()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PlateTrackingApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close
    )

    root.mainloop()
