import cv2
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ---------- настройки ----------
VIDEO_PATH = "mister-i-missis-smit-2005_1.mkv"          # исходный ролик
OUT_VIDEO_PATH = "outputs/output_skeletons.mp4"
OUT_JSON_PATH = "outputs/skeletons.json"
MODEL_PATH = "models/yolov8n-pose.pt"    # ммодель yolov8s-pose.pt

CONF_THRES = 0.25                 # порог уверенности детекции
# -------------------------------

# 1. загрузка модели
model = YOLO(MODEL_PATH)

# 2. открываем видео
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (w, h))

all_frames = []   # сюда будем складывать скелеты для JSON
frame_idx = 0

# упрощённый «скелет» COCO: пары индексов keypoints,

SKELETON_EDGES = [
    (5, 7), (7, 9),    # левая рука
    (6, 8), (8, 10),   # правая рука
    (11, 13), (13, 15),  # левая нога
    (12, 14), (14, 16),  # правая нога
    (5, 6), (11, 12),    # плечи, бёдра
    (5, 11), (6, 12)     # тело
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. инференс YOLOv8 pose
    results = model(frame, conf=CONF_THRES, verbose=False)[0]

    frame_entry = {
        "frame_idx": int(frame_idx),
        "persons": []
    }

    # 4. обрабатываем каждую детекцию человека
    if results.keypoints is not None:
        kpts = results.keypoints  # ultralytics.engine.results.Keypoints [web:177]
        boxes = results.boxes

        for i in range(len(boxes)):
            box = boxes[i].xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]

            # keypoints в нормированных координатах (0..1) [web:48][web:173]
            kps_xy = kpts.xyn[i].cpu().numpy()  # shape (17,2)
            kps_conf = kpts.conf[i].cpu().numpy()  # shape (17,)
            keypoints = []
            for (x, y), c in zip(kps_xy, kps_conf):
                keypoints.append(
                    {"x": float(x), "y": float(y), "conf": float(c)}
                )

            # добавляем в JSON‑структуру
            frame_entry["persons"].append(
                {
                    "bbox": box,
                    "keypoints": keypoints
                }
            )

            # рисуем bbox
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # рисуем точки и «кости»
            pts_img = np.stack(
                [kps_xy[:, 0] * w, kps_xy[:, 1] * h], axis=1
            ).astype(int)

            for x, y in pts_img:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            for a, b in SKELETON_EDGES:
                xa, ya = pts_img[a]
                xb, yb = pts_img[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                         (255, 0, 0), 2)

    all_frames.append(frame_entry)
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

# 5. сохраняем JSON со скелетами
with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "video_path": str(Path(VIDEO_PATH).absolute()),
            "width": w,
            "height": h,
            "fps": fps,
            "frames": all_frames
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print("Готово:")
print(f"- видео со скелетами: {OUT_VIDEO_PATH}")
print(f"- JSON со скелетами:  {OUT_JSON_PATH}")