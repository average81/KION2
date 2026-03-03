import logging

multimodal_models = {
    "TST":None
}

"""
Класс для мультимодальной классификации действий.

Attributes:
    model_name (str): Имя модели для извлечения позы.
    frame_sampling_rate (int): Степень прореживания кадров видео.
    static_action_period (int): Период в числе кадров поз для определения статических действий.
    dynamic_action_period (int): Период в числе кадров поз для определения динамических действий.
    verbose (bool): Флаг для включения подробного логирования.
    logger: Логгер для записи информационных сообщений.
"""

class MultimodalActionClassificator:
    def __init__(self, 
                 model_name,
                 frame_sampling_rate=1, 
                 static_action_period=30, 
                 dynamic_action_period=60, 
                 verbose=False):

        self.model_name = model_name
        self.frame_sampling_rate = frame_sampling_rate
        self.static_action_period = static_action_period
        self.dynamic_action_period = dynamic_action_period
        self.verbose = verbose
        
        # Инициализация логирования
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Проверка корректности имени модели
        if self.model_name not in multimodal_models:
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models: {list(multimodal_models.keys())}")
            
        # Вывод в лог сообщения об успешной инициализации и параметрах
        self.logger.info(f"MultimodalActionClassificator initialized successfully with model='{self.model_name}', "
                        f"frame_sampling_rate={self.frame_sampling_rate}, "
                        f"static_action_period={self.static_action_period}, "
                        f"dynamic_action_period={self.dynamic_action_period}, "
                        f"verbose={self.verbose}")

