import numpy as np

# Словарь суставов с их индексами и названиями
JOINTS = {
    0: "base_of_spine",  # основание позвоночника
    1: "middle_of_spine",  # середина позвоночника
    2: "neck",  # шея
    3: "head",  # голова
    4: "left_shoulder",  # левое плечо
    5: "left_elbow",  # левый локоть
    6: "left_wrist",  # левое запястье
    7: "left_hand",  # левая рука
    8: "right_shoulder",  # правое плечо
    9: "right_elbow",  # правый локоть
    10: "right_wrist",  # правое запястье
    11: "right_hand",  # правая рука
    12: "left_hip",  # левое бедро
    13: "left_knee",  # левое колено
    14: "left_ankle",  # левая лодыжка
    15: "left_foot",  # левая ступня
    16: "right_hip",  # правое бедро
    17: "right_knee",  # правое колено
    18: "right_ankle",  # правая лодыжка
    19: "right_foot",  # правая ступня
    20: "spine",  # позвоночник
    21: "tip_of_left_hand",  # кончик левой руки
    22: "left_thumb",  # левый большой палец
    23: "tip_of_right_hand",  # кончик правой руки
    24: "right_thumb",  # правый большой палец
    25: "nose",  # нос
    26: "left_eye",  # левый глаз
    27: "right_eye",  # правый глаз
    28: "left_ear",  # левое ухо
    29: "right_ear",  # правое ухо
}



class Pose:
    def __init__(self):
        self.box = np.array([[]], dtype=np.uint8)
        self.keypoints = np.zeros((len(JOINTS.keys()), len(["x","y"])),dtype=np.float32)
        self.box_conf = 0
        self.id=-1
        self.keypoints_conf= np.zeros((len(self.keypoints)),dtype=np.float32)
        self.frame_idx = 0
