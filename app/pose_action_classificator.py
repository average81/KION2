import logging
from models.LSTM_models import LSTM_model
from models.stgcn_models import STGCN_model,STGN_ACTIONS_MAPPING
from models.conv3DCNN_models import conv3DCNN_model
from models.action_format import *
# Словарь поддерживаемых моделей (можно расширять)
action_models = {
    "LSTMSkeletonNet": {"model":LSTM_model,"params":{"weights":"models/lstm_gcn.pth",
                                                     "num_classes": 60,'fusion':'attention',
                                                     "hidden_size": 256, "num_layers": 2,
                                                     "bodies":2,"dropout": 0.4}},
    "STGCN_model_kinetics": {"model":STGCN_model , "params": {"weights": "models/st_gcn.kinetics.pt","num_classes": 400,
                                                     "label_map_path": "models/stgcn/kinetics400-id2label.txt",
                                                                 'mapping':STGN_ACTIONS_MAPPING }},
    "STGCN_model_rgbd": {"model":STGCN_model , "params": {"weights": "models/st_gcn.ntu60_gleb.pt","num_classes": 60,
                                                              "label_map_path": "models/stgcn/ntu60-id2label.txt"} },
    "Conv3dNet": {"model":conv3DCNN_model,"params":{"weights":"models/conv3dcnn.pth", "num_classes": 60,
                                                     "bodies":4}},
}

"""
Класс для классификации действий на основе оценки позы.

Attributes:
    model_name (str): Имя модели для классификации поз. Должно быть одним из поддерживаемых имен в словаре action_models.
    static_action_period (int): Период в числе кадров поз для определения статических действий. Должен быть положительным целым числом.
    dynamic_action_period (int): Период в числе кадров поз для определения динамических действий. Должен быть положительным целым числом.
    verbose (bool): Флаг для включения подробного логирования.
    model: Загруженная модель для классификации действий.
    logger: Логгер для записи информационных сообщений.
"""

class PoseActionClassificator:
    def __init__(self, model_name="LSTMSkeletonNet", action_period=30,threshold = 0.8, min_pose_frames = 30, min_pair_poses_frames = 30, verbose=False):
        self.model_name = model_name
        self.action_period = action_period
        self.min_pair_poses_frames = min_pair_poses_frames
        self.min_pose_frames = min_pose_frames

        
        # Проверка корректности имени модели
        if self.model_name not in action_models:
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models: {list(action_models.keys())}")
        
        # Проверка корректности периодов
        if not isinstance(self.action_period, int) or self.action_period < 1:
            raise ValueError("action_period must be a positive integer >= 1")
            
        # Инициализация модели
        self.model = action_models[self.model_name]["model"](action_models[self.model_name]["params"],
                                                             threshold=threshold)  # загружаем предобученную модель
        self.threshold = threshold
        # Инициализация логирования
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        self.logger.info(f"PoseActionClassificator initialized with model='{self.model_name}', "
                        f"action_period={self.action_period}, "
                        f"verbose={self.verbose}")
    #Определение одного действия
    def classify_one(self, poses):
        self.logger.info(f"Classifying {len(poses)} poses.")
        action,raw = self.model.predict(poses)
        if raw is None:
            return  None, None
        return  {
                "frame_idx": poses[0].frame_idx,
                "person_id": poses[0].id,
                "action": action.to_dict()
                }, raw
    # Определение всех действий
    def classify(self,poses):
        # Создаем словарь для группировки поз по id
        poses_by_id = {}
        for pose in poses:
            person_id = pose.id
            if person_id not in poses_by_id:
                poses_by_id[person_id] = []
            poses_by_id[person_id].append(pose)
        # Преобразуем в список списков, отсортированный по id
        pose_batches = []
        person_ids = []

        for person_id in sorted(poses_by_id.keys()):
            person_poses = poses_by_id[person_id]
            # Сортируем позы по номеру кадра
            person_poses.sort(key=lambda x: x.frame_idx)
            pose_batches.append(person_poses)
            person_ids.append(person_id)

        self.logger.info(f"Grouped poses into {len(pose_batches)} batches by person_id: {person_ids}")

        # Классифицируем действия для каждой группы поз (индивидуальные действия)
        results = []
        raw_res = []
        for batch in pose_batches:
            # Проверяем, что актер присутствует в достаточном количестве кадров
            if len(batch) >= self.min_pose_frames:
                # Берем последовательность длиной action_period
                for i in range(len(batch) // self.action_period + 1):
                    sequence = batch[i*self.action_period:(i+1)*self.action_period ]
                    result,raw = self.classify_one(sequence)
                    if result is not None:
                        results.append(result)
                        raw_res.append(raw)
            else:
                self.logger.info(f"Person ID {batch[0].id} has only {len(batch)} frames, which is less than min_pose_frames={self.min_pose_frames}. Skipping classification.")

        # Группировка поз по кадрам
        poses_by_frame = {}
        for pose in poses:
            frame_idx = pose.frame_idx
            if frame_idx not in poses_by_frame:
                poses_by_frame[frame_idx] = []
            poses_by_frame[frame_idx].append(pose)

        # Поиск пар актеров в одном кадре и сборка батчей
        pair_batches = {}
        for frame_idx, frame_poses in poses_by_frame.items():
            # Получаем уникальные ID актеров в кадре
            person_ids_in_frame = [pose.id for pose in frame_poses]
            # Генерируем пары
            for i in range(len(person_ids_in_frame)):
                for j in range(i + 1, len(person_ids_in_frame)):
                    pair = tuple(sorted([person_ids_in_frame[i], person_ids_in_frame[j]]))
                    if pair not in pair_batches:
                        pair_batches[pair] = []
                    # Добавляем позы для обоих актеров из этого кадра
                    pair_poses = [pose for pose in frame_poses if pose.id in pair]
                    pair_batches[pair].append(pair_poses)

        # Классификация действий для парных батчей
        for pair, batch in pair_batches.items():
            if len(batch) >= self.min_pair_poses_frames:
                # Сортируем батч по frame_idx
                batch.sort(key=lambda x: x[0].frame_idx)
                # Разбиваем на последовательности длиной action_period
                for i in range(len(batch) // self.action_period + 1):
                    sequence = batch[i*self.action_period:(i+1)*self.action_period]
                    if len(sequence) == self.action_period:
                        # Объединяем позы из пары для классификации
                        flat_sequence = [pose for pair_poses in sequence for pose in pair_poses]
                        result, raw = self.classify_one(flat_sequence)
                        results.append(result)
                        raw_res.append(raw)
            else:
                self.logger.info(f"Pair {pair} has only {len(batch)} frames, which is less than min_pair_poses_frames={self.min_pair_poses_frames}. Skipping classification.")

        return results, raw_res

