import logging
import os
import utils.utils as utils
from app.pose_estimator import PoseEstimator
from app.pose_action_classificator import PoseActionClassificator
from app.multimodal_action_classificator import MultimodalActionClassificator

default_config = {
    "pose_ext_model": "OpenPose",
    "pose_action_model": "LSTM_ActionNet",
    "multimodal_model": "TST",
    "video_decimation": 1,
    'static_act_frames': 60,
    'dynamic_act_frames': 60
}

class VideoProcessor:
    def __init__(self, input_file, output_dir="output", verbose=False, config_path="config.yml"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.config_path = config_path
        
        # Инициализация логирования без влияния на root logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        
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
        action_config = {'pose_action_model': self.config['pose_action_model']}
        if "static_act_frames" in self.config.keys():
            action_config['static_act_frames']=self.config['static_act_frames']
        else:
            action_config['static_act_frames']=default_config['static_act_frames']
        if "dynamic_act_frames" in self.config.keys():
            action_config['dynamic_act_frames']=self.config['dynamic_act_frames']
        else:
            action_config['dynamic_act_frames']=default_config['dynamic_act_frames']

        #Создаем объект класса определения поз и действий
        if self.config["pose_ext_model"] != 'None' and self.config['pose_action_model'] != None :
            self.pose_extractor=PoseEstimator(
                model_name=self.config["pose_ext_model"],
                frame_sampling_rate = self.config["video_decimation"],
                verbose = self.verbose)

            self.pose_action_classifier = PoseActionClassificator(
                action_config['pose_action_model'],
                action_config['static_act_frames'],
                action_config['dynamic_act_frames'],
                verbose = self.verbose)
        else:
            self.pose_extractor = None
            self.pose_action_classificator=None
        #Создаем экземпляр мультимодальной модели
        if self.config["multimodal_model"] != None:
            self.mult_modl_clssfr=MultimodalActionClassificator(self.config["multimodal_model"],
                                                       self.config["video_decimation"],
                                                       action_config['static_act_frames'],
                                                       action_config['dynamic_act_frames'],
                                                       verbose=self.verbose)
        else:
            self.mult_modl_clssfr = None
        self.logger.info("VideoProcessor initialized successfully.")
