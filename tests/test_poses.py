import os
import sys
import logging

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(CURRENT_DIR)
# sys.path.append(PARENT_DIR)

from app.video_processor import VideoProcessor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    vp = VideoProcessor(
        input_file="video_samples/potter1 01.mp4",      # путь к короткому видео
        output_dir="outputs",
        verbose=True,
        config_path="config.yml",   # можно не создавать заранее, он сам сохранится
    )

    result = vp.process()            # словарь
    poses = result["raw_poses"]      # список поз

    print(f"Всего поз: {len(poses)}")
    # Выведем первые 3 записи
    for p in poses[:3]:
        print(
            f"frame={p['frame_idx']}, "
            f"person={p['person_id']}, "
            f"kpts_shape={p['keypoints'].shape}"
        )