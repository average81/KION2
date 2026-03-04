import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO  
import torch


pose_models = {
    "YOLOv8-Pose-N": "yolov8n-pose.pt",
    "YOLOv8-Pose-S": "yolov8s-pose.pt",
}


"""
Класс для оценки позы на основе видео.

    Attributes:
    model_name (str): Имя модели для извлечения позы. Должно быть одним из поддерживаемых имен в словаре pose_models.
    frame_sampling_rate (int): Степень прореживания кадров видео. Определяет, через сколько кадров берется кадр для обработки.
    verbose (bool): Флаг для включения подробного логирования.
    model: Загруженная модель для извлечения позы.
    logger: Логгер для записи информационных сообщений.
"""

class PoseEstimator:

    def __init__(self, model_name="YOLOv8-Pose-N", frame_sampling_rate=1, verbose=False):
        self.model_name = model_name
        self.frame_sampling_rate = frame_sampling_rate

        # Проверка имени модели
        if self.model_name not in pose_models:
            raise ValueError(
                f"Model '{self.model_name}' is not supported. "
                f"Supported models: {list(pose_models.keys())}"
            )

        # Проверка шага по кадрам
        if not isinstance(self.frame_sampling_rate, int) or self.frame_sampling_rate < 1:
            raise ValueError("frame_sampling_rate must be a positive integer >= 1")

        # Инициализация логирования
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Загрузка YOLOv8-pose
        weights = pose_models[self.model_name]
        self.model = YOLO(weights)  # загружаем предобученную модель позы

        self.logger.info(
            f"PoseEstimator initialized with model='{self.model_name}' "
            f"({weights}) and frame_sampling_rate={self.frame_sampling_rate}, "
            f"verbose={self.verbose}"
        )
        # выбор устройства
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def estimate_video(self, video_path, conf=0.25):
        """
        Прогоняет видео и возвращает список поз по кадрам.

        Возвращает список словарей:
        [
            {
                "frame_idx": int,
                "person_id": int,               # индекс персоны в кадре (не трек-ID)
                "keypoints": np.ndarray (K, 3)  # x, y, conf
            },
            ...
        ]
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_idx = 0
        poses = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # прореживание по кадрам
            if frame_idx % self.frame_sampling_rate != 0:
                frame_idx += 1
                continue

            # Запуск модели (YOLO сам вернёт объект Results с keypoints)
            results = self.model(frame, conf=conf, verbose=False)

            for r in results:
                if r.keypoints is None:
                    continue

                # r.keypoints.data: tensor (num_persons, K, 3) -> numpy
                kps = r.keypoints.data.cpu().numpy()

                for person_idx, kp in enumerate(kps):
                    poses.append(
                        {
                            "frame_idx": frame_idx,
                            "person_id": person_idx,
                            "keypoints": kp,  # (K, 3): x, y, conf
                        }
                    )

            frame_idx += 1

        cap.release()
        self.logger.info(
            f"Pose estimation finished. Frames processed: {frame_idx}, "
            f"poses collected: {len(poses)}"
        )
        return poses