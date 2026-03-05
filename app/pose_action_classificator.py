import logging
# Словарь поддерживаемых моделей (можно расширять)
action_models = {
    "LSTM_ActionNet": None,
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
    def __init__(self, model_name="LSTM_ActionNet", static_action_period=30, dynamic_action_period=60, verbose=False):
        self.model_name = model_name
        self.static_action_period = static_action_period
        self.dynamic_action_period = dynamic_action_period
        

        
        # Проверка корректности имени модели
        if self.model_name not in action_models:
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models: {list(action_models.keys())}")
        
        # Проверка корректности периодов
        if not isinstance(self.static_action_period, int) or self.static_action_period < 1:
            raise ValueError("static_action_period must be a positive integer >= 1")
            
        if not isinstance(self.dynamic_action_period, int) or self.dynamic_action_period < 1:
            raise ValueError("dynamic_action_period must be a positive integer >= 1")
            
        # Инициализация модели (заглушка)
        self.model = action_models[self.model_name]
        
        # Инициализация логирования
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        self.logger.info(f"PoseActionClassificator initialized with model='{self.model_name}', "
                        f"static_action_period={self.static_action_period}, "
                        f"dynamic_action_period={self.dynamic_action_period}, "
                        f"verbose={self.verbose}")

    def classify(self, poses): #Заглушка
        self.logger.info(f"Classifying {len(poses)} poses (stub).")
        return [
            {
                "frame_idx": p["frame_idx"],
                "person_id": p["person_id"],
                "action": "unknown",
                "score": 0.0,
            }
            for p in poses
        ]
