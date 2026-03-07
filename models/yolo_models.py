from ultralytics import YOLO
from models.pose_format import *
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Сопоставление индексов суставов YOLO с индексами в JOINTS
YOLO_JOINTS_MAPPING = {
    0: 26,  # Nose -> nose
    1: 27,  # Left Eye -> left_eye
    2: 28,  # Right Eye -> right_eye
    3: 29,  # Left Ear -> left_ear
    4: 30,  # Right Ear -> right_ear
    5: 5,   # Left Shoulder -> left_shoulder
    6: 9,   # Right Shoulder -> right_shoulder
    7: 6,   # Left Elbow -> left_elbow
    8: 10,  # Right Elbow -> right_elbow
    9: 7,   # Left Wrist -> left_wrist
    10: 11, # Right Wrist -> right_wrist
    11: 13, # Left Hip -> left_hip
    12: 17, # Right Hip -> right_hip
    13: 14, # Left Knee -> left_knee
    14: 18, # Right Knee -> right_knee
    15: 15, # Left Ankle -> left_ankle
    16: 19  # Right Ankle -> right_ankle
}

class YoloModel:
    def __init__(self, params, threshold):
        self.weights = params["weights"]
        self.threshold = threshold
        self.model=YOLO(self.weights).to(DEVICE)
    def detect(self,image):
        result = self.model(image, conf=self.threshold, verbose=False)[0]
        poses=[]
        if result.keypoints is not None:
            kpts = result.keypoints
            boxes = result.boxes
            pose =Pose()
            for i in range(len(boxes)):
                pose.box = boxes[i].xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                kpts_yolo = kpts.xy[i].cpu().numpy()
                kpts_conf_yolo = kpts.conf[i].cpu().numpy()
                
                # Создаем пустые массивы для keypoints и confidence с учетом размерности JOINTS
                outkpts = np.zeros((len(JOINTS), 2), dtype=np.float32)
                kpts_conf = np.zeros(len(JOINTS), dtype=np.float32)
                
                # Заполняем keypoints и confidence по маппингу
                # for yolo_idx, joint_idx in YOLO_JOINTS_MAPPING.items():
                #     outkpts[joint_idx-1] = kpts_yolo[yolo_idx-1]  # -1 для перехода от 1-based к 0-based индексации
                #     kpts_conf[joint_idx-1] = kpts_conf_yolo[yolo_idx-1]
                for yolo_idx, joint_idx in YOLO_JOINTS_MAPPING.items():
                    # joint_idx (1-based из JOINTS) -> joint_idx-1 в outkpts
                    # yolo_idx уже 0-based, трогать не нужно
                    outkpts[joint_idx - 1] = kpts_yolo[yolo_idx]
                    kpts_conf[joint_idx - 1] = kpts_conf_yolo[yolo_idx]
                pose.keypoints = outkpts
                pose.keypoints_conf = kpts_conf
                pose.box_conf = boxes[i].conf.cpu().numpy()
                pose.id = 0
                poses.append(pose)
        return poses