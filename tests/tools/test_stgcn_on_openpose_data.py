"""
Тестовый скрипт для проверки связки:

  OpenPose JSON (BODY_25) → адаптер `openpose_to_stgcn_adapter` → ST-GCN.

Берёт папку с файлами `*_keypoints.json`, конвертирует их в тензор формата
`(1, 3, T, 18, 1)` с нормализацией как при обучении Kinetics, и прогоняет
через предобученную ST-GCN, выводя Top‑5 действий.

Запуск из корня проекта:
    python tests/tools/test_stgcn_on_openpose_data.py \\
        Open_pose_samples/S001C001P001R002A022_rgb.avi_output_json \\
        --width 1920 --height 1080
"""
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта app
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from models.stgcn.stgcn_wrapper import STGCNWrapper
from models.stgcn.openpose_to_stgcn_adapter import load_sequence_from_json_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Инференс ST-GCN по папке OpenPose *_keypoints.json."
    )
    parser.add_argument(
        "json_dir",
        type=str,
        help="Папка с OpenPose *_keypoints.json для одного ролика.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=340,
        help="Ширина кадра в пикселях. Для 1920x1080 укажите 1920.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Высота кадра в пикселях. Для 1920x1080 укажите 1080.",
    )
    args = parser.parse_args()

    # 1. OpenPose JSON -> тензор для ST-GCN
    data_numpy = load_sequence_from_json_dir(
        args.json_dir, width=args.width, height=args.height
    )
    print("data_numpy shape:", data_numpy.shape)

    # 2. Предобученная ST-GCN на Kinetics
    model = STGCNWrapper(
        weights_path="models/st_gcn.kinetics.pt",
        label_map_path="models/stgcn/kinetics400-id2label.txt"
    )

    # 3. Получаем Top‑5 предсказаний
    top5 = model.predict_topk(data_numpy, k=5)
    print("Top-5:")
    for cls_id, prob, label in top5:
        print(cls_id, label, f"{prob:.4f}")


if __name__ == "__main__":
    main()