import os
import glob
import json
import numpy as np
import torch
import sys

# with open('kinetics400-id2label.txt', 'r') as f:
#     id2label = json.load(f)

# from net.st_gcn import Model  #

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

def load_sequence_from_json_dir(json_dir, min_score=0.1):
    files = sorted(glob.glob(os.path.join(json_dir, '*_keypoints.json')))
    frames = []

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        if not data['people']:
            continue

        # выбираем человека с максимальной суммарной уверенностью
        best = max(
            data['people'],
            key=lambda p: sum(p['pose_keypoints_2d'][2::3])
        )
        kps = np.array(best['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)  # (25,3)

        # фильтрация по минимальной уверенности
        conf = kps[:, 2]
        if conf.mean() < min_score:
            # кадр слишком шумный, можно пропустить
            continue

        # берём только нужные 18 суставов
        kps = kps[BODY25_TO_18, :]  # (18,3)

        # нормализация: центрируем по "mid-hip" (середина тазобедренных),
        # масштаб по расстоянию между шеей (neck=1) и mid-hip
        coords = kps[:, :2]  # (18,2)
        neck = kps[1, :2]
        right_hip = kps[8, :2]   # наш индекс 8 = right_hip
        left_hip  = kps[11, :2]  # наш индекс 11 = left_hip
        mid_hip = (right_hip + left_hip) / 2.0

        center = mid_hip
        coords_centered = coords - center

        # масштаб: расстояние между шеей и mid_hip
        torso_size = np.linalg.norm(neck - mid_hip) + 1e-6
        coords_norm = coords_centered / torso_size

        # собираем обратно (x,y,score)
        kps_norm = np.zeros_like(kps)
        kps_norm[:, :2] = coords_norm
        kps_norm[:, 2] = kps[:, 2]

        # (C,V): C=3
        frame = kps_norm.T  # (3,18)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f'No valid skeletons found in {json_dir}')

    data = np.stack(frames, axis=1)  # (3,T,18)
    data = data[:, :, :, None]       # (3,T,18,1)
    data = data[None, ...]           # (1,3,T,18,1)

    print('data_numpy shape:', data.shape,
          'min:', float(data.min()), 'max:', float(data.max()))
    return data

# def main():
#     if len(sys.argv) < 2:
#         print('Usage: python demo_myjson.py path/to/json_dir')
#         return

#     json_dir = sys.argv[1]
#     data_numpy = load_sequence_from_json_dir(json_dir)
#     data_tensor = torch.from_numpy(data_numpy).float()

#     # создаём модель как в test.yaml
#     model = Model(
#         in_channels=3,
#         num_class=400,
#         edge_importance_weighting=True,
#         graph_args={'layout': 'openpose', 'strategy': 'spatial'}
#     )
#     ckpt = torch.load('./models/st_gcn.kinetics.pt', map_location='cpu')
#     model.load_state_dict(ckpt)
#     model.eval()

#     with torch.no_grad():
#         out = model(data_tensor)  # (1,400)
#         prob = torch.softmax(out, dim=1)[0]
#         topk = torch.topk(prob, k=5)

#         print('Top-5 predictions:')
#         for cls_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
#             label = id2label[str(cls_id)]
#             print(f'{cls_id}: {label} -> {float(p):.4f}')
# if __name__ == '__main__':
#     main()