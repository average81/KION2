"""
Скрипт для извлечения поз человека из видео: запускает VideoProcessor, сохраняет результаты в JSON 
и при необходимости строит видео с наложенным скелетом, используя единую 30‑точечную разметку суставов.
python 

Варианты запуска:
python tests/video_to_pose.py video_samples/video_in.avi
python -m tests.video_to_pose video_samples/video_in.avi

Параметры:
  video              путь к входному видео
  --output_dir DIR   каталог для сохранения JSON и видео с позами (по умолчанию: outputs)
  --config_path CFG  путь к config.yml для VideoProcessor (по умолчанию: config.yml)
  --no_vis           не сохранять видео с наложенным скелетом, только JSON
  --show             показывать окно с визуализацией в реальном времени
  --debug_joints     рисовать номера и имена суставов для отладки

"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import cv2
from app.video_processor import VideoProcessor
import utils.utils as utils
from models.pose_format import JOINTS
from utils.visualize import NAME2IDX, IDX2NAME, SKELETON_EDGES
from utils.visualize import draw_pose, debug_draw_joints, get_color_for_person, _is_zero_point

def load_poses_by_frame(json_path: str | Path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_poses = data["raw_poses"]
    by_frame = {}
    for p in raw_poses:
        frame_idx = p["frame_idx"]
        by_frame.setdefault(frame_idx, []).append(p)
    return by_frame


class PosePipeline:
    def __init__(self, config_path: str = "config.yml", output_dir: str = "outputs"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_paths(self, video_path: Path):
        stem = video_path.stem  # my_video.mp4 -> my_video 
        json_path = self.output_dir / f"{stem}_poses.json"
        vis_path = self.output_dir / f"{stem}_poses.mp4"
        return json_path, vis_path

    def run(self, video_path: str, save_vis: bool = True, show: bool = False,
            debug_joints: bool = False) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # 1. запустить основной пайплайн
        processor = VideoProcessor(
            str(video_path),
            output_dir=str(self.output_dir),
            verbose=False,
            config_path=self.config_path,
        )
        result = processor.process()      # dict с "raw_poses" и др.
        serializable = utils.numpy_to_builtin(result)

        # 2. сохранить JSON с позами
        json_path, vis_path = self._make_paths(video_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        # 3. опционально сделать видео с наложением поз
        if save_vis:
            self._visualize_video(video_path, json_path, vis_path,
                                  show=show, debug_joints=debug_joints)

        return {
            "json_path": str(json_path),
            "video_path": str(vis_path) if save_vis else None,
            "raw_result": serializable,
        }

    def _visualize_video(self, video_path: Path, json_path: Path,
                         out_video_path: Path, show: bool, debug_joints: bool):
        poses_by_frame = load_poses_by_frame(json_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for pose in poses_by_frame.get(frame_idx, []):
                if debug_joints:
                    debug_draw_joints(frame, pose)
                else:
                    draw_pose(frame, pose)

            writer.write(frame)

            if show:
                cv2.imshow("pose_viz", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Видео → позы (JSON) и, опционально, видео с наложенным скелетом."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Путь к входному видеофайлу.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Путь к конфигурационному файлу VideoProcessor.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Каталог для сохранения результатов.",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Не сохранять видео с наложением поз, только JSON.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать окно с видео при обработке.",
    )
    parser.add_argument(
        "--debug_joints",
        action="store_true",
        help="Рисовать отладочные номера суставов вместо обычного скелета.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Включить подробный лог.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    pipeline = PosePipeline(
        config_path=args.config_path,
        output_dir=args.output_dir,
    )

    result = pipeline.run(
        video_path=args.video,
        save_vis=not args.no_vis,
        show=args.show,
        debug_joints=args.debug_joints,
    )

    print(f"JSON:  {result['json_path']}")
    if result["video_path"] is not None:
        print(f"Video: {result['video_path']}")