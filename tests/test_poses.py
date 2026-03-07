# Файл в настоящий момент reference only
# Для запуска в корне проекта выполните python -m tests.test_poses
# В папке outputs появятся файл action_results

import os
import sys
import logging
import json
import numpy as np
from pathlib import Path
import utils.utils as utils

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
    
    # Сохранение результатов в JSON файл
    # Преобразуем результаты
    serializable_result = utils.numpy_to_builtin(result)
    
    # Создаем директорию для сохранения
    output_path = Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем в JSON файл
    json_path = output_path / "action_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в {json_path}")