from models.pose_format import JOINTS  # твой файл с описанием 30 суставов
import cv2

# 1 → "base_of_spine" → индекс 0 в keypoints
NAME2IDX = {name: idx for idx, name in JOINTS.items()}
IDX2NAME = {idx: name for idx, name in JOINTS.items()}

SKELETON_EDGES = [
    # позвоночник
    (NAME2IDX["base_of_spine"], NAME2IDX["middle_of_spine"]),
    (NAME2IDX["middle_of_spine"], NAME2IDX["spine"]),
    (NAME2IDX["spine"], NAME2IDX["neck"]),
    (NAME2IDX["neck"], NAME2IDX["head"]),

    # левая рука
    (NAME2IDX["spine"], NAME2IDX["left_shoulder"]),
    (NAME2IDX["left_shoulder"], NAME2IDX["left_elbow"]),
    (NAME2IDX["left_elbow"], NAME2IDX["left_wrist"]),
    (NAME2IDX["left_wrist"], NAME2IDX["left_hand"]),
    (NAME2IDX["left_hand"], NAME2IDX["tip_of_left_hand"]),
    (NAME2IDX["left_hand"], NAME2IDX["left_thumb"]),

    # правая рука
    (NAME2IDX["spine"], NAME2IDX["right_shoulder"]),
    (NAME2IDX["right_shoulder"], NAME2IDX["right_elbow"]),
    (NAME2IDX["right_elbow"], NAME2IDX["right_wrist"]),
    (NAME2IDX["right_wrist"], NAME2IDX["right_hand"]),
    (NAME2IDX["right_hand"], NAME2IDX["tip_of_right_hand"]),
    (NAME2IDX["right_hand"], NAME2IDX["right_thumb"]),

    # левая нога
    (NAME2IDX["base_of_spine"], NAME2IDX["left_hip"]),
    (NAME2IDX["left_hip"], NAME2IDX["left_knee"]),
    (NAME2IDX["left_knee"], NAME2IDX["left_ankle"]),
    (NAME2IDX["left_ankle"], NAME2IDX["left_foot"]),

    # правая нога
    (NAME2IDX["base_of_spine"], NAME2IDX["right_hip"]),
    (NAME2IDX["right_hip"], NAME2IDX["right_knee"]),
    (NAME2IDX["right_knee"], NAME2IDX["right_ankle"]),
    (NAME2IDX["right_ankle"], NAME2IDX["right_foot"]),

    # голова / лицо
    (NAME2IDX["head"], NAME2IDX["nose"]),
    (NAME2IDX["nose"], NAME2IDX["left_eye"]),
    (NAME2IDX["nose"], NAME2IDX["right_eye"]),
    (NAME2IDX["left_eye"], NAME2IDX["left_ear"]),
    (NAME2IDX["right_eye"], NAME2IDX["right_ear"]),
]

def get_color_for_person(person_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
    ]
    return palette[person_id % len(palette)]

def _is_zero_point(x, y) -> bool:
    return x == 0.0 and y == 0.0


def draw_pose(frame, pose, conf_thr: float = 0.1):
    keypoints = pose["keypoints"]            # [[x, y], ...] длиной 30
    keypoints_conf = pose["keypoints_conf"]  # [c1, ...] длиной 30
    box = pose.get("box")
    person_id = pose.get("person_id", 0)

    skel_color = get_color_for_person(person_id)
    point_color = (0, 0, 255)

    # линии скелета
    for i, j in SKELETON_EDGES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1 = keypoints[i]
        x2, y2 = keypoints[j]
        c1 = keypoints_conf[i] if i < len(keypoints_conf) else 0.0
        c2 = keypoints_conf[j] if j < len(keypoints_conf) else 0.0

        if c1 <= conf_thr or c2 <= conf_thr:
            continue
        if _is_zero_point(x1, y1) or _is_zero_point(x2, y2):
            continue

        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            skel_color,
            2,
        )

    # точки
    for (x, y), c in zip(keypoints, keypoints_conf):
        if c <= conf_thr or _is_zero_point(x, y):
            continue
        cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)

    # bbox
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            skel_color,
            2,
        )

def debug_draw_joints(frame, pose, conf_thr: float = 0.0):
    keypoints = pose["keypoints"]
    keypoints_conf = pose["keypoints_conf"]

    for idx, ((x, y), c) in enumerate(zip(keypoints, keypoints_conf)):
        if c <= conf_thr or _is_zero_point(x, y):
            continue
        name = IDX2NAME.get(idx, str(idx))

        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{idx}:{name}",
            (int(x) + 3, int(y) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )
