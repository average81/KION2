"""Сборка массивов NTU 25/18 из .skeleton (colorX/colorY)."""
import math
import pickle
from pathlib import Path

import numpy as np

from .parse_ntu_skeleton import read_skeleton_file, NUM_JOINT

MAX_FRAME = 300
MAX_BODY = 2

# openpose 18 joint index -> ntu 25 joint index (0-based)
OPENPOSE18_TO_NTU25 = [
    3,   # 0: nose        <- head
    2,   # 1: neck        <- neck
    8,   # 2: r_shoulder  <- right_shoulder
    9,   # 3: r_elbow     <- right_elbow
    10,  # 4: r_wrist     <- right_wrist
    4,   # 5: l_shoulder  <- left_shoulder
    5,   # 6: l_elbow     <- left_elbow
    6,   # 7: l_wrist     <- left_wrist
    16,  # 8: r_hip       <- right_hip
    17,  # 9: r_knee      <- right_knee
    18,  # 10: r_ankle    <- right_ankle
    12,  # 11: l_hip      <- left_hip
    13,  # 12: l_knee     <- left_knee
    14,  # 13: l_ankle    <- left_ankle
    -1,  # 14: right_eye  (нет аналога)
    -1,  # 15: left_eye
    -1,  # 16: right_ear
    -1,  # 17: left_ear
]


def build_ntu_2d_25_18(
    skeleton_paths,
    label_dict,
    out_data25_path,
    out_data18_path,
    out_label_path,
    width: float = 1920.0,
    height: float = 1080.0,
):
    """Собирает два массива:
       - (N,3,T,25,2) из colorX/colorY
       - (N,3,T,18,2) через маппинг OPENPOSE18_TO_NTU25.
    Нормализация: x_pix/width - 0.5, y_pix/height - 0.5.
    """
    skeleton_paths = [Path(p) for p in skeleton_paths]
    N = len(skeleton_paths)

    data25 = np.zeros((N, 3, MAX_FRAME, NUM_JOINT, MAX_BODY), dtype=np.float32)
    data18 = np.zeros((N, 3, MAX_FRAME, 18, MAX_BODY), dtype=np.float32)

    sample_name = []
    sample_label = []

    for i, skel_path in enumerate(skeleton_paths):
        name = skel_path.stem  # S002C002P003R002A001
        frames = read_skeleton_file(skel_path)
        T = min(len(frames), MAX_FRAME)

        if name not in label_dict:
            # можно логировать, если нужно строгое совпадение
            continue

        sample_name.append(name)
        sample_label.append(label_dict[name])

        for t in range(T):
            bodies = frames[t]
            for m, body in enumerate(bodies[:MAX_BODY]):
                joints = body["joints"]
                if len(joints) < NUM_JOINT:
                    continue

                # заполняем 25-суставный массив
                for v in range(NUM_JOINT):
                    j = joints[v]
                    x_pix = j["colorX"]
                    y_pix = j["colorY"]

                    # nan: сравнение x<=0 ложно → раньше протекал nan в массив
                    if not (math.isfinite(x_pix) and math.isfinite(y_pix)):
                        continue
                    if x_pix <= 0 or y_pix <= 0:
                        continue

                    x01 = x_pix / width
                    y01 = y_pix / height

                    x_norm = x01 - 0.5
                    y_norm = y01 - 0.5

                    data25[i, 0, t, v, m] = x_norm
                    data25[i, 1, t, v, m] = y_norm
                    data25[i, 2, t, v, m] = 1.0

                # openpose-18 через маппинг
                for j_out, j_ntu in enumerate(OPENPOSE18_TO_NTU25):
                    if j_ntu < 0:
                        continue
                    data18[i, 0, t, j_out, m] = data25[i, 0, t, j_ntu, m]
                    data18[i, 1, t, j_out, m] = data25[i, 1, t, j_ntu, m]
                    data18[i, 2, t, j_out, m] = data25[i, 2, t, j_ntu, m]

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{N} skeletons processed")

    for name_arr in (data25, data18):
        if np.isnan(name_arr).any() or np.isinf(name_arr).any():
            raise RuntimeError(
                "В массиве всё ещё есть nan/inf после сборки — проверьте .skeleton"
            )

    np.save(out_data25_path, data25)
    np.save(out_data18_path, data18)
    with open(out_label_path, "wb") as f:
        pickle.dump((sample_name, sample_label), f)

    print(f"Saved 25-joint data to {out_data25_path}")
    print(f"Saved 18-joint data to {out_data18_path}")
    print(f"Saved labels to {out_label_path}")