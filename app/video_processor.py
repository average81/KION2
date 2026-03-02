import logging
import os
import utils.utils as utils


default_config = {
    "pose_ext_model": None,
    "pose_action_model": None,
    "multimodal_model": None}

class VideoProcessor:
    def __init__(self, input_file, output_dir="output", verbose=False, config_path="config.yml"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.config_path = config_path
        
        # Инициализация логирования без влияния на root logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Добавляем handler, если его нет
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # Проверка существования входного файла
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file {self.input_file} does not exist.")
            raise FileNotFoundError(f"Input file {self.input_file} does not exist.")
            
        # Загрузка или создание конфигурации
        if os.path.exists(self.config_path):
            self.config = utils.open_yaml(self.config_path)
        else:
            self.config = default_config
            try:
                utils.save_yaml(self.config_path, self.config)
                self.logger.info(f"Default configuration saved to {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Could not save default config: {e}")
            
        self.logger.info("VideoProcessor initialized successfully.")
