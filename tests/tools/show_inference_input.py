"""
Отображение того же numpy, что отправляется в модель при инференсе: графы скелета и статистика.

Нормализация как при обучении Kinetics: координаты в [0,1] по размеру кадра (по умолчанию 340×256), затем -0.5.
Источник данных — тот же, что в test_stgcn_in_project.py:
  - либо папка с OpenPose *_keypoints.json (--json_dir),
  - либо наш JSON от VideoProcessor (--json_file).

Полученный массив (1, 3, T, 18, 1) выводится в виде статистики по каналам и визуализации нескольких кадров скелета.

Примеры:
  python show_inference_input.py --json_dir Open_pose_samples/S001C001P002A022_rgb.avi_output_json
  python show_inference_input.py --json_dir ... --width 1920 --height 1080   # если кадр 1920x1080
  python show_inference_input.py --json_file outputs/my_video_poses.json --out inference_input.png

Запуск из корня проекта:
  python tests/tools/show_inference_input.py --json_dir ... --width 1920 --height 1080
"""
import sys
from pathlib import Path

# Корень проекта в sys.path, чтобы находился модуль app при запуске скрипта из любой папки
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import matplotlib.pyplot as plt
import numpy as np

# Рёбра скелета OpenPose 18 (как в app/stgcn/graph.py)
SKELETON_EDGES_18 = [
    (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
    (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
]


def draw_skeleton_18(ax, x, y, score, conf_thr=0.01):
    """Рисует скелет 18 точек: линии по рёбрам и точки с подписью индекса.
    Нулевые (обнулённые по confidence) суставы не соединяем линиями — иначе видна линия из (0,0)."""
    for (i, j) in SKELETON_EDGES_18:
        if i >= len(x) or j >= len(x):
            continue
        if score[i] < conf_thr or score[j] < conf_thr:
            continue
        ax.plot([x[i], x[j]], [y[i], y[j]], "b-", linewidth=1)
    # рисуем только уверенные точки, чтобы не было красной точки в (0,0) от обнулённых суставов
    mask = score >= conf_thr
    if np.any(mask):
        ax.scatter(x[mask], y[mask], c="red", s=15)
    for i in range(len(x)):
        if score[i] >= conf_thr:
            ax.text(x[i], y[i], str(i), fontsize=5)
    ax.invert_yaxis()
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(
        description="Графы и статистика по numpy, который подаётся в ST-GCN при инференсе."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--json_dir",
        type=str,
        help="Папка с OpenPose *_keypoints.json (как в test_stgcn_in_project.py).",
    )
    group.add_argument(
        "--json_file",
        type=str,
        help="Путь к нашему JSON с raw_poses (outputs/*_poses.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="inference_input.png",
        help="Путь к выходному PNG.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=9,
        help="Сколько кадров скелета нарисовать.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=340,
        help="Ширина кадра в пикселях (координаты в JSON в пикселях). По умолчанию 340 (Kinetics). Для 1920x1080 укажите 1920.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Высота кадра в пикселях. По умолчанию 256 (Kinetics). Для 1920x1080 укажите 1080.",
    )
    args = parser.parse_args()

    if args.json_dir:
        from app.stgcn.openpose_to_stgcn_adapter import load_sequence_from_json_dir
        data_numpy = load_sequence_from_json_dir(
            args.json_dir, width=args.width, height=args.height
        )
    else:
        from app.stgcn.json_to_stgcn_adapter import load_sequence_from_our_json
        data_numpy = load_sequence_from_our_json(
            args.json_file, width=args.width, height=args.height
        )

    # Форма как при вызове model.predict_*(data_numpy): (1, 3, T, 18, 1)
    n, c, T, V, m = data_numpy.shape
    assert n == 1 and c == 3 and V == 18 and m == 1, data_numpy.shape

    x_all = data_numpy[0, 0]   # (T, 18)
    y_all = data_numpy[0, 1]
    s_all = data_numpy[0, 2]

    # Статистика (как в validate_stgcn_npy_pkl / inspect_labels)
    print("Форма массива, подаваемого в модель:", data_numpy.shape)
    print("Канал x:  min = {:.4f}, max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
        float(x_all.min()), float(x_all.max()), float(x_all.mean()), float(x_all.std())))
    print("Канал y:  min = {:.4f}, max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
        float(y_all.min()), float(y_all.max()), float(y_all.mean()), float(y_all.std())))
    print("Канал score: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
        float(s_all.min()), float(s_all.max()), float(s_all.mean())))

    # Визуализация кадров
    num_frames = min(args.num_frames, T)
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    cols = 3
    rows = (num_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, t in enumerate(frame_indices):
        ax = axes[idx]
        draw_skeleton_18(ax, x_all[t], y_all[t], s_all[t])
        ax.set_title(f"Кадр {t}")
        ax.axis("on")
        ax.grid(True, alpha=0.3)

    for idx in range(num_frames, len(axes)):
        axes[idx].axis("off")

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    fig.suptitle(
        f"Вход в модель: shape {tuple(data_numpy.shape)} | "
        f"x ∈ [{x_min:.3f}, {x_max:.3f}], y ∈ [{y_min:.3f}, {y_max:.3f}]",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Визуализация сохранена: {args.out}")


if __name__ == "__main__":
    main()
