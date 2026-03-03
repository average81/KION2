pose_models = {
    "OpenPose": None
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
    def __init__(self, model_name="OpenPose", frame_sampling_rate=1, verbose=False):
        self.model_name = model_name
        self.frame_sampling_rate = frame_sampling_rate
        
        # Проверка корректности имени модели
        if self.model_name not in pose_models:
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models: {list(pose_models.keys())}")
        
        # Проверка корректности степени прореживания кадров
        if not isinstance(self.frame_sampling_rate, int) or self.frame_sampling_rate < 1:
            raise ValueError("frame_sampling_rate must be a positive integer >= 1")
            
        # Инициализация модели (заглушка)
        self.model = pose_models[self.model_name]
        
        # Инициализация логирования
        self.verbose = verbose
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        self.logger.info(f"PoseEstimator initialized with model='{self.model_name}' and frame_sampling_rate={self.frame_sampling_rate}, verbose={self.verbose}")
