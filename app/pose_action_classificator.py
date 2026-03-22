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
    "STGCN_model_kinetics": {"model":STGCN_model , "params": {"weights": "models/st_gcn.kinetics.pt",
                                                     "label_map_path": "models/stgcn/kinetics400-id2label.txt",
                                                                 'mapping':STGN_ACTIONS_MAPPING }},
    "STGCN_model_rgbd": {"model":STGCN_model , "params": {"weights": "models/st_gcn.rgbd.pt",
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
    def __init__(self, model_name="LSTMSkeletonNet", action_period=30,threshold = 0.8,verbose=False):
        self.model_name = model_name
        self.action_period = action_period
        

        
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
            # Берем последовательность длиной action_period
            for i in range(len(batch) // self.action_period + 1):
                sequence = batch[i*self.action_period:(i+1)*self.action_period ]
                result,raw = self.classify_one(sequence)
                results.append(result)
                raw_res.append(raw)
        return results,raw_res

