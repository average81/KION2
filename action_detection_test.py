import argparse
from pathlib import Path
from app.stgcn.json_to_stgcn_adapter import recognize_action_from_video


def main():
    parser = argparse.ArgumentParser(description="Распознавание действия в коротком видео (ST-GCN).")
    parser.add_argument(
        "video",
        type=str,
        help="Путь к видеофайлу (например video_samples/S001C001P001R001A026_rgb.avi)",
    )
    args = parser.parse_args()
    video_path = args.video

    top5 = recognize_action_from_video(
        video_path=video_path,
        config_path="config.yml",
        output_dir="outputs",
        device="cpu",
        k=5,
    )

    print("Top‑5 действий для ролика:", Path(video_path).name)
    for cls_id, prob, label in top5:
        print(f"{cls_id:3d}  {prob:7.4f}  {label}")

    # при желании можно взять top‑1:
    best_cls_id, best_prob, best_label = top5[0]
    print("\nЛучшее действие:")
    print(best_cls_id, best_prob, best_label)


if __name__ == "__main__":
    main()