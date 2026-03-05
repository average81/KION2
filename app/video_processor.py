import logging
import os
import utils.utils as utils
from app.pose_estimator import PoseEstimator
from app.pose_action_classificator import PoseActionClassificator
from app.multimodal_action_classificator import MultimodalActionClassificator
from pathlib import Path
import json
from copy import deepcopy
import numpy as np

default_config = {
    "pose_ext_model": "YOLOv8-Pose-N",  # было "OpenPose"
    "pose_ext_th": 0.8,
    "pose_action_model": "LSTM_ActionNet",
    "multimodal_model": "TST",
    "video_decimation": 1,
    "static_act_frames": 60,
    "dynamic_act_frames": 60
}


class VideoProcessor:
    def __init__(self, input_file, output_dir="output", verbose=False, config_path="config.yml"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.config_path = config_path
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file {self.input_file} does not exist.")
            raise FileNotFoundError(f"Input file {self.input_file} does not exist.")
            
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
        if "static_act_frames" in self.config:
            action_config['static_act_frames'] = self.config['static_act_frames']
        else:
            action_config['static_act_frames'] = default_config['static_act_frames']
        if "dynamic_act_frames" in self.config:
            action_config['dynamic_act_frames'] = self.config['dynamic_act_frames']
        else:
            action_config['dynamic_act_frames'] = default_config['dynamic_act_frames']
        if "pose_ext_th" in self.config:
            action_config['pose_ext_th'] = self.config['pose_ext_th']
        else:
            action_config['pose_ext_th'] = default_config['pose_ext_th']

        if self.config["pose_ext_model"] != 'None' and self.config['pose_action_model'] is not None:
            self.pose_extractor = PoseEstimator(
                model_name=self.config["pose_ext_model"],
                frame_sampling_rate=self.config["video_decimation"],
                verbose=self.verbose,
                threshold = self.config.get("pose_ext_th",0.8)
            )

            self.pose_action_classifier = PoseActionClassificator(
                action_config['pose_action_model'],
                action_config['static_act_frames'],
                action_config['dynamic_act_frames'],
                verbose=self.verbose,
            )
        else:
            self.pose_extractor = None
            self.pose_action_classifier = None

        if self.config["multimodal_model"] is not None:
            self.mult_modl_clssfr = MultimodalActionClassificator(
                self.config["multimodal_model"],
                self.config["video_decimation"],
                action_config['static_act_frames'],
                action_config['dynamic_act_frames'],
                verbose=self.verbose,
            )
        else:
            self.mult_modl_clssfr = None

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("VideoProcessor initialized successfully.")

    def process(self):
        """
        Основной пайплайн:
        1) извлечение поз из видео PoseEstimator-ом,
        2) классификация действий по позам (если есть),
        3) мультимодальная классификация (если есть),
        4) сохранение результатов в JSON.
        """
        self.logger.info("Starting video processing...")

        results = {
            "input_file": self.input_file,
            "pose_actions": None,
            "multimodal_actions": None,
            "raw_poses": None,
        }

        # 1. Извлечение поз из видео
        if self.pose_extractor is not None:
            poses = self.pose_extractor.estimate_video(
                self.input_file
            )
            results["raw_poses"] = poses

            # 2. Классификация действий по позам (если есть классификатор)
            if hasattr(self, "pose_action_classifier") and self.pose_action_classifier is not None:
                pose_actions = self.pose_action_classifier.classify(poses)
                results["pose_actions"] = pose_actions
            else:
                self.logger.info("PoseActionClassificator is None, skipping pose-based actions.")
        else:
            self.logger.info("PoseEstimator is None, skipping pose extraction.")

        # 3. Мультимодальная модель (если есть)
        if self.mult_modl_clssfr is not None:
            multimodal_actions = self.mult_modl_clssfr.classify(self.input_file)
            results["multimodal_actions"] = multimodal_actions
        else:
            self.logger.info("MultimodalActionClassificator is None, skipping multimodal actions.")

        # 4. Сохранение результатов в JSON
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        out_json_path = Path(self.output_dir) / "actions_results.json"

        # делаем копию результата, чтобы не портить оригинальные numpy-массивы
        safe_results = deepcopy(results)

        # конвертируем keypoints: np.ndarray -> list
        if safe_results["raw_poses"] is not None:
            for p in safe_results["raw_poses"]:
                kpts = p.get("keypoints")
                if isinstance(kpts, np.ndarray):
                    p["keypoints"] = kpts.tolist()

        try:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(safe_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved to {out_json_path}")
        except Exception as e:
            self.logger.warning(f"Could not save results JSON: {e}")
        self.logger.info("Video processing finished.")
        
        return results