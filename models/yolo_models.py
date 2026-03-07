from ultralytics import YOLO
from models.pose_format import *
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Сопоставление индексов суставов YOLO с индексами в JOINTS
YOLO_JOINTS_MAPPING = {
    0: 25,  # Nose -> nose
    1: 26,  # Left Eye -> left_eye
    2: 27,  # Right Eye -> right_eye
    3: 28,  # Left Ear -> left_ear
    4: 29,  # Right Ear -> right_ear
    5: 4,   # Left Shoulder -> left_shoulder
    6: 8,   # Right Shoulder -> right_shoulder
    7: 5,   # Left Elbow -> left_elbow
    8: 9,   # Right Elbow -> right_elbow
    9: 6,   # Left Wrist -> left_wrist
    10: 10, # Right Wrist -> right_wrist
    11: 12, # Left Hip -> left_hip
    12: 16, # Right Hip -> right_hip
    13: 13, # Left Knee -> left_knee
    14: 17, # Right Knee -> right_knee
    15: 14, # Left Ankle -> left_ankle
    16: 18  # Right Ankle -> right_ankle
}

class YoloModel:
    def __init__(self, params, threshold):
        self.weights = params["weights"]
        self.threshold = threshold
        self.model=YOLO(self.weights).to(DEVICE)
    def detect(self,images):
        results = self.model.predict(source=images, conf=self.threshold, verbose=False)
        poses=[]
        for result in results:
            if len(result) >0:
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
                    for yolo_idx, joint_idx in YOLO_JOINTS_MAPPING.items():
                        outkpts[joint_idx] = kpts_yolo[yolo_idx]  # теперь индексы уже 0-based
                        kpts_conf[joint_idx] = kpts_conf_yolo[yolo_idx]

                    pose.keypoints = outkpts
                    pose.keypoints_conf = kpts_conf
                    pose.box_conf = boxes[i].conf.cpu().numpy()
                    pose.id = 0
                    poses.append(pose)
        return poses