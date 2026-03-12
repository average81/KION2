"""
Полный пайплайн: видео → позы (YOLO/VideoProcessor) → ST-GCN → Top‑5 действий.

Использует PosePipeline для извлечения поз из видео, нормализует координаты
как при обучении Kinetics (по размеру кадра из видео), прогоняет через
предобученную ST-GCN и выводит предсказания.

Запуск из корня проекта:
  python tests/action_detection_video-yolo-stgcn.py path/to/video.mp4
  python tests/action_detection_video-yolo-stgcn.py path/to/video.mp4 --device cuda:0
"""
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта app
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from app.stgcn.json_to_stgcn_adapter import recognize_action_from_video


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
        "-k",
        type=int,
        default=5,
        dest="k",
        help="Сколько предсказаний вывести (Top-k).",
    )
    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    top5 = recognize_action_from_video(
        video_path=video_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        device=args.device,
        k=args.k,
    )

    print("Top‑5 действий для ролика:", video_path.name)
    for cls_id, prob, label in top5:
        print(f"{cls_id:3d}  {prob:7.4f}  {label}")

    best_cls_id, best_prob, best_label = top5[0]
    print("\nЛучшее действие:")
    print(best_cls_id, best_prob, best_label)


if __name__ == "__main__":
    main()
