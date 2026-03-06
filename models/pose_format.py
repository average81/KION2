import numpy as np

# Словарь суставов с их индексами и названиями
JOINTS = {
    1: "base_of_spine",  # основание позвоночника
    2: "middle_of_spine",  # середина позвоночника
    3: "neck",  # шея
    4: "head",  # голова
    5: "left_shoulder",  # левое плечо
    6: "left_elbow",  # левый локоть
    7: "left_wrist",  # левое запястье
    8: "left_hand",  # левая рука
    9: "right_shoulder",  # правое плечо
    10: "right_elbow",  # правый локоть
    11: "right_wrist",  # правое запястье
    12: "right_hand",  # правая рука
    13: "left_hip",  # левое бедро
    14: "left_knee",  # левое колено
    15: "left_ankle",  # левая лодыжка
    16: "left_foot",  # левая ступня
    17: "right_hip",  # правое бедро
    18: "right_knee",  # правое колено
    19: "right_ankle",  # правая лодыжка
    20: "right_foot",  # правая ступня
    21: "spine",  # позвоночник
    22: "tip_of_left_hand",  # кончик левой руки
    23: "left_thumb",  # левый большой палец
    24: "tip_of_right_hand",  # кончик правой руки
    25: "right_thumb",  # правый большой палец
    26: "nose",  # нос
    27: "left_eye",  # левый глаз
    28: "right_eye",  # правый глаз
    29: "left_ear",  # левое ухо
    30: "right_ear",  # правое ухо
}



class Pose:
    def __init__(self):
        self.box = np.array([[]], dtype=np.uint8)
        self.keypoints = np.zeros((len(JOINTS.keys()), len(["x","y"])),dtype=np.float32)
        self.box_conf = 0
        self.id=-1
        self.keypoints_conf= np.zeros((len(self.keypoints)),dtype=np.float32)
