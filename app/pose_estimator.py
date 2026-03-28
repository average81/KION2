import logging
from pathlib import Path

import cv2
import numpy as np
from models.yolo_models import YoloModel
import torch
import tqdm
from utils.utils import calculate_iou


pose_models = {
    "YOLOv8-Pose-N": {"model": YoloModel,"params":{"weights":"models/yolov8n-pose.pt"}},
    "YOLOv8-Pose-S": {"model": YoloModel,"params":{"weights":"models/yolov8s-pose.pt"}},
    "YOLOv26-Pose-N": {"model": YoloModel,"params":{"weights":"models/yolo26n-pose.pt"}},
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

    def __init__(self, model_name="YOLOv8-Pose-N", frame_sampling_rate=1, verbose=False, threshold = 0.8, batch_size = 128):
        self.model_name = model_name
        self.frame_sampling_rate = frame_sampling_rate
        self.batch_size = batch_size
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
        self.threshold =threshold
        # Загрузка модель

        self.model = pose_models[self.model_name]["model"](pose_models[self.model_name]["params"],
                                                           threshold = self.threshold)  # загружаем предобученную модель позы

        self.logger.info(
            f"PoseEstimator initialized with model='{self.model_name}' "
            f"({pose_models[self.model_name]['params']}) and frame_sampling_rate={self.frame_sampling_rate}, "
            f"verbose={self.verbose}"
        )
        # выбор устройства
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def estimate_video(self, video_path):
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

        poses = []
        batch_cnt = 0
        batch = []
        last_boxes = []
        last_ids = []
        cur_id = 0 # ID текущего объекта
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in tqdm.tqdm(range(int(total_frames)), desc=f"{Path(video_path).__repr__()}"):
            ret, frame = cap.read()
            if not ret:
                break

            # прореживание по кадрам
            if i % self.frame_sampling_rate != 0:
                continue
            # Добавляем кадр в батч
            #frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            #if batch.numel() == 0:
            #    batch = frame_tensor
            #else:
            #    batch = torch.cat([batch, frame_tensor], dim=0)
            batch.append(frame)
            batch_cnt += 1
            
            # Обработка батча при достижении нужного размера или в конце видео
            if batch_cnt == self.batch_size or i == total_frames - 1:
                # Запуск модели
                results = self.model.detect(batch)
                
                # Обработка результатов

                for frame_id,result in enumerate(results):
                    new_boxes = []
                    new_ids = []
                    for frame_result in result:
                        if frame_result.keypoints is None:
                            continue
                        new_boxes.append(frame_result.box)
                        id = -1
                        for j in range(len(last_boxes)):
                            if calculate_iou(frame_result.box,last_boxes[j]) > 0.7 :
                                id = last_ids[j]
                        if id == -1:
                            #Не отслеживается трек, заводим новый id
                            cur_id += 1
                            id =cur_id
                        new_ids.append(id)
                        frame_result.id=id
                        frame_result.frame_idx = i - batch_cnt + frame_id + 1
                        poses.append(frame_result)
                    last_ids =new_ids
                    last_boxes=new_boxes
                
                # Сброс батча
                batch_cnt = 0
                batch = []
                


        cap.release()
        self.logger.info(
            f"Pose estimation finished. Frames processed: {i}, "
            f"poses collected: {len(poses)}"
        )
        return poses