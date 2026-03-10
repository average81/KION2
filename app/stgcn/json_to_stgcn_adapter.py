import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from models.pose_format import JOINTS
from app.stgcn.stgcn_wrapper import STGCNWrapper
from tests.video_to_pose import PosePipeline  # если нужно, можно импортнуть из другого места


# Построим имя -> индекс для нашей 30‑точечной системы
NAME2IDX = {name: idx for idx, name in JOINTS.items()}

# Маппинг нашей 30‑точечной разметки → 18‑суставный layout openpose
OUR30_TO_OPENPOSE18 = [
    NAME2IDX["nose"],
    NAME2IDX["neck"],
    NAME2IDX["right_shoulder"],
    NAME2IDX["right_elbow"],
    NAME2IDX["right_wrist"],
    NAME2IDX["left_shoulder"],
    NAME2IDX["left_elbow"],
    NAME2IDX["left_wrist"],
    NAME2IDX["right_hip"],
    NAME2IDX["right_knee"],
    NAME2IDX["right_ankle"],
    NAME2IDX["left_hip"],
    NAME2IDX["left_knee"],
    NAME2IDX["left_ankle"],
    NAME2IDX["right_eye"],
    NAME2IDX["left_eye"],
    NAME2IDX["right_ear"],
    NAME2IDX["left_ear"],
]


def _synthesize_neck_if_missing(
    keypoints: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.1,
) -> None:
    """
    Синтезирует шею, если её нет, а плечи есть.

    keypoints: (J,2)
    conf: (J,)
    Модифицирует keypoints/conf по месту.
    """
    neck_idx = NAME2IDX["neck"]
    ls_idx = NAME2IDX["left_shoulder"]
    rs_idx = NAME2IDX["right_shoulder"]

    neck_c = conf[neck_idx]
    ls_c = conf[ls_idx]
    rs_c = conf[rs_idx]

    # если шея уже есть с нормальной уверенностью — ничего не делаем
    if neck_c >= min_conf:
        return

    # если оба плеча есть — ставим шею в середину между ними
    if ls_c >= min_conf and rs_c >= min_conf:
        neck_xy = (keypoints[ls_idx] + keypoints[rs_idx]) / 2.0
        keypoints[neck_idx] = neck_xy
        conf[neck_idx] = float((ls_c + rs_c) / 2.0)
    # иначе оставляем как есть (0,0,0) — модель сама будет игнорировать по низкому score


def load_sequence_from_our_json(
    json_path: str | Path,
    min_frame_conf: float = 0.1,
) -> np.ndarray:
    """
    Читает JSON, сохранённый PosePipeline / VideoProcessor,
    и преобразует в тензор для ST‑GCN: (1, 3, T, 18, 1).
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_poses: List[Dict[str, Any]] = data.get("raw_poses", [])
    if not raw_poses:
        raise RuntimeError(f"raw_poses пуст в {json_path}")

    # группируем по кадрам
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for p in raw_poses:
        frame_idx = p["frame_idx"]
        by_frame.setdefault(frame_idx, []).append(p)

    frames_arrays: List[np.ndarray] = []

    # обрабатываем кадры в порядке возрастания индекса
    for frame_idx in sorted(by_frame.keys()):
        poses_in_frame = by_frame[frame_idx]
        if not poses_in_frame:
            continue

        # выбираем человека с максимальной суммарной уверенностью
        best_pose = max(
            poses_in_frame,
            key=lambda p: float(sum(p.get("keypoints_conf", []))),
        )

        keypoints = np.array(best_pose["keypoints"], dtype=np.float32)  # (J,2)
        conf = np.array(best_pose["keypoints_conf"], dtype=np.float32)  # (J,)

        if keypoints.shape[0] != len(JOINTS):
            # на всякий случай жёстко проверяем
            raise ValueError(
                f"Ожидалось {len(JOINTS)} точек, но получено {keypoints.shape[0]} "
                f"в кадре {frame_idx}"
            )

        # синтез шеи, если нужно (для YOLO‑кейса)
        _synthesize_neck_if_missing(keypoints, conf)

        # Если кадр совсем шумный — можно выкинуть
        if float(conf.mean()) < min_frame_conf:
            continue

        # Берём только те 18 суставов, которые нужны ST‑GCN (openpose‑layout)
        kps_18_xy = keypoints[OUR30_TO_OPENPOSE18]   # (18,2)
        conf_18 = conf[OUR30_TO_OPENPOSE18]          # (18,)

        # Нормализация как в demo_myjson:
        # центр — mid_hip, масштаб — расстояние между шеей и mid_hip
        # (работаем уже в 18‑суставном пространстве)

        coords = kps_18_xy.copy()               # (18,2)
        neck = coords[1]                        # index 1 = neck
        right_hip = coords[8]                   # index 8 = right_hip
        left_hip = coords[11]                   # index 11 = left_hip
        mid_hip = (right_hip + left_hip) / 2.0  # (2,)

        center = mid_hip
        coords_centered = coords - center

        torso_size = np.linalg.norm(neck - mid_hip) + 1e-6
        coords_norm = coords_centered / torso_size

        # собираем (18,3) = (x,y,score)
        kps_norm = np.zeros((18, 3), dtype=np.float32)
        kps_norm[:, :2] = coords_norm
        kps_norm[:, 2] = conf_18

        # приводим к формату (C,V) = (3,18)
        frame_array = kps_norm.T  # (3,18)
        frames_arrays.append(frame_array)

    if not frames_arrays:
        raise RuntimeError(f"Не удалось собрать ни одного кадра из {json_path}")

    # (3,T,18)
    data = np.stack(frames_arrays, axis=1)
    # (3,T,18,1)
    data = data[:, :, :, None]
    # (1,3,T,18,1)
    data = data[None, ...]

    return data


def recognize_action_from_video(
    video_path: str | Path,
    config_path: str = "config.yml",
    output_dir: str = "outputs",
    device: str = "cpu",
    k: int = 5,
):
    """
    Высокоуровневая функция:
    - прогоняет короткое видео через PosePipeline,
    - конвертирует позы в формат ST‑GCN,
    - возвращает top‑k предсказаний действий.
    """
    video_path = Path(video_path)

    # 1. Видео → позы (JSON)
    pipeline = PosePipeline(
        config_path=config_path,
        output_dir=output_dir,
    )
    res = pipeline.run(
        video_path=str(video_path),
        save_vis=False,   # для задачи классификации действия нам достаточно JSON
        show=False,
        debug_joints=False,
    )
    poses_json_path = res["json_path"]

    # 2. JSON → (1,3,T,18,1)
    data_numpy = load_sequence_from_our_json(poses_json_path)

    # 3. ST‑GCN предобученный
    model = STGCNWrapper(
        weights_path="models/st_gcn.kinetics.pt",
        label_map_path="app/stgcn/kinetics400-id2label.txt",
        device=device,
    )

    topk = model.predict_topk(data_numpy, k=k)
    # topk: список (cls_id, prob, label_str)
    return topk