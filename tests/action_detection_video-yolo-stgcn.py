"""
Полный пайплайн: видео → позы (YOLO/VideoProcessor) → ST-GCN → Top‑k действий.

По умолчанию пресет **NTU-60** (--dataset ntu60): 60 классов, label_map из
models/stgcn/ntu60-id2label.txt, веса по умолчанию — models/epoch42_model.pt
(смените путь в коде или передайте --stgcn_weights).

Координаты нормализуются по размеру кадра видео (как в json_to_stgcn_adapter).

Примеры из корня проекта:
  python tests/action_detection_video-yolo-stgcn.py path/to/video.mp4
  python tests/action_detection_video-yolo-stgcn.py path/to/video.mp4 --device cuda:0

Другие веса NTU:
  python tests/action_detection_video-yolo-stgcn.py video.mp4 --stgcn_weights models/my_model.pt

Kinetics-400 (400 классов, другие дефолтные веса и label_map):
  python tests/action_detection_video-yolo-stgcn.py video.mp4 --dataset kinetics400

Без --stgcn_weights / --label_map / --num_class подставляются значения из выбранного
пресета --dataset (ntu60 или kinetics400).
"""
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта models.stgcn
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from models.stgcn.json_to_stgcn_adapter import recognize_action_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание действия в коротком видео: видео → позы → ST-GCN."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Путь к видеофайлу (например video_samples/S001C001P001R001A026_rgb.avi).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Путь к config.yml для пайплайна поз.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Каталог для сохранения JSON с позами.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство для ST-GCN (cpu или cuda:0).",
    )
    parser.add_argument(
        "--stgcn_weights",
        type=str,
        default=None,
        help=(
            "Путь к весам ST-GCN (.pt). Если не задан — из пресета --dataset "
            "(ntu60: models/epoch42_model.pt, kinetics400: models/st_gcn.kinetics.pt)."
        ),
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help=(
            "JSON id→label. Если не задан — из пресета --dataset. "
            "Чтобы отключить подписи классов: none"
        ),
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=None,
        help=(
            "Число классов ST-GCN. Если не задано — из пресета (ntu60: 60, kinetics400: 400)."
        ),
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        dest="k",
        help="Сколько предсказаний вывести (Top-k).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kinetics400", "ntu60"],
        default="ntu60",
        help="Пресет: дефолтные веса, num_class, label_map. По умолчанию: ntu60.",
    )

    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    # Дефолты для ntu60 / kinetics400 (см. docstring модуля)
    if args.dataset == "kinetics400":
        default_weights = "models/st_gcn.kinetics.pt"
        default_label_map = "models/stgcn/kinetics400-id2label.txt"
        default_num_class = 400
    elif args.dataset == "ntu60":
        default_weights = "models/epoch42_model.pt"
        default_label_map = "models/stgcn/ntu60-id2label.txt"
        default_num_class = 60
    else:
        raise ValueError(f"Unknown dataset preset: {args.dataset}")

    stgcn_weights = args.stgcn_weights if args.stgcn_weights is not None else default_weights

    if args.label_map is not None and str(args.label_map).lower() == "none":
        label_map = None
    elif args.label_map is not None:
        label_map = args.label_map
    else:
        label_map = default_label_map

    num_class = args.num_class if args.num_class is not None else default_num_class

    topk = recognize_action_from_video(
        video_path=video_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        device=args.device,
        k=args.k,
        stgcn_weights_path=stgcn_weights,
        label_map_path=label_map,
        num_class=num_class,
    )

    print(f"Top-{args.k} действий для ролика:", video_path.name)
    for cls_id, prob, label in topk:
        print(f"{cls_id:3d}  {prob:7.4f}  {label}")

    best_cls_id, best_prob, best_label = topk[0]
    print("\nЛучшее действие:")
    print(best_cls_id, best_prob, best_label)


if __name__ == "__main__":
    main()
