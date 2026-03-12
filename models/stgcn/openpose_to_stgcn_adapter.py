import os
import glob
import json
import numpy as np

# BODY_25 -> 18-суставный layout 'openpose'
BODY25_TO_18 = [
    0,  # nose
    1,  # neck
    2,  # right_shoulder
    3,  # right_elbow
    4,  # right_wrist
    5,  # left_shoulder
    6,  # left_elbow
    7,  # left_wrist
    9,  # right_hip
    10, # right_knee
    11, # right_ankle
    12, # left_hip
    13, # left_knee
    14, # left_ankle
    15, # right_eye
    16, # left_eye
    17, # right_ear
    18, # left_ear
]


def load_sequence_from_json_dir(json_dir, min_score=0.1, width=340, height=256):
    """
    Загрузка последовательности поз из папки OpenPose *_keypoints.json
    и преобразование в тензор формата (1, 3, T, 18, 1) для ST-GCN.

    Нормализация как при обучении Kinetics:
      - координаты (x, y) считаются в пикселях кадра;
      - делим на (width, height), получаем [0,1];
      - вычитаем 0.5, центр кадра становится (0, 0).

    По умолчанию width=340, height=256 (как в Kinetics-skeleton), но
    для реальных видео нужно передавать фактический размер кадра.
    """
    files = sorted(glob.glob(os.path.join(json_dir, "*_keypoints.json")))
    frames = []
    wh = np.array([width, height], dtype=np.float32)

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        if not data["people"]:
            continue

        # выбираем человека с максимальной суммарной уверенностью
        best = max(
            data["people"],
            key=lambda p: sum(p["pose_keypoints_2d"][2::3]),
        )
        kps = np.array(best["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)  # (25,3)

        # фильтрация по минимальной уверенности
        conf = kps[:, 2]
        if conf.mean() < min_score:
            continue

        # берём только нужные 18 суставов
        kps = kps[BODY25_TO_18, :]  # (18,3)
        coords = kps[:, :2].copy()  # (18,2) пиксели
        conf_18 = kps[:, 2]

        # Нормализация как в Kinetics (feeder_kinetics): [0,1] по кадру, затем -0.5
        coords = coords / wh - 0.5
        coords[conf_18 < 1e-6] = 0  # обнуляем точки с нулевым score

        kps_norm = np.zeros((18, 3), dtype=np.float32)
        kps_norm[:, :2] = coords
        kps_norm[:, 2] = kps[:, 2]

        frame = kps_norm.T  # (3,18)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No valid skeletons found in {json_dir}")

    data = np.stack(frames, axis=1)  # (3,T,18)
    data = data[:, :, :, None]       # (3,T,18,1)
    data = data[None, ...]           # (1,3,T,18,1)

    print(
        "data_numpy shape:",
        data.shape,
        "min:",
        float(data.min()),
        "max:",
        float(data.max()),
    )
    return data

