import json
from pathlib import Path
from utils.visualize import draw_pose, debug_draw_joints, draw_actions_on_frame

import cv2



def load_poses_by_frame(json_path: str | Path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_poses = data["raw_poses"]
    by_frame = {}
    for p in raw_poses:
        frame_idx = p["frame_idx"]
        by_frame.setdefault(frame_idx, []).append(p)
    return by_frame
def load_actions_by_frame(json_path: str | Path, max_frame: int | None = None):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("pose_actions", [])
    
    # Проверка на пустой список действий
    if not actions:
        return {}
    
    # Загружаем позы по кадрам для определения продолжительности присутствия актеров
    poses_by_frame = load_poses_by_frame(json_path)
    
    # Определяем диапазон кадров для каждого актера
    actor_frames = {}
    for frame_idx, poses in poses_by_frame.items():
        for pose in poses:
            actor_id = pose.get("person_id", 0)
            if actor_id not in actor_frames:
                actor_frames[actor_id] = []
            actor_frames[actor_id].append(frame_idx)
    # Создаем интервалы действий для каждого актера
    actor_actions = {}
    for action in actions:
        frame_idx = action["frame_idx"]
        actor_id = action.get("person_id", 0)
        
        if actor_id not in actor_actions:
            actor_actions[actor_id] = []
        
        actor_actions[actor_id].append({
            "action": action,
            "start_frame": frame_idx
        })
    
    # Формируем словарь действий по кадрам
    by_frame = {}
    for actor_id, actions_list in actor_actions.items():
        # Сортируем действия актера по кадрам
        actions_list.sort(key=lambda x: x["start_frame"])
        # Получаем диапазон кадров, в которых присутствует актер
        actor_frame_range = actor_frames.get(actor_id, [])
        last_pose_frame = max(actor_frame_range) if actor_frame_range else 0
        
        # Для каждого действия устанавливаем интервал до следующего действия или последнего кадра с позой
        for i, action_info in enumerate(actions_list):
            start_frame = action_info["start_frame"]
            # Конечный кадр определяется как минимум из:
            # - начало следующего действия
            # - последний кадр с позой актера + 1
            # - max_frame
            next_action_frame = actions_list[i + 1]["start_frame"] if i + 1 < len(actions_list) else float('inf')
            last_frame = min(next_action_frame, last_pose_frame + 1, max_frame or float('inf'))
            # Проверяем, что интервал валиден
            if start_frame < last_frame:
                # Добавляем действие во все кадры интервала
                for frame_idx in range(start_frame, int(last_frame)):
                    if frame_idx not in by_frame:
                        by_frame[frame_idx] = []
                    by_frame[frame_idx].append(action_info["action"])
    
    return by_frame
def load_boxes_by_frame(json_path: str | Path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("raw_poses",[])
    by_frame = {}
    for pose in poses:
        frame_idx = pose["frame_idx"]
        by_frame.setdefault(frame_idx, []).append(pose)
    return by_frame







def visualize_from_json(
    video_path: str | Path,
    json_path: str | Path,
    out_video_path: str | Path | None = None,
    show: bool = False,
    debug_joints: bool = False,
):
    poses_by_frame = load_poses_by_frame(json_path)
    boxes_by_frame = load_boxes_by_frame(json_path)
    
    # Получаем общее количество кадров из видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Загружаем действия с учетом максимального количества кадров
    actions_by_frame = load_actions_by_frame(json_path, max_frame=total_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video_path is not None:
        out_video_path = str(out_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for pose in poses_by_frame.get(frame_idx, []):
            if debug_joints:
                debug_draw_joints(frame, pose)
            else:
                draw_pose(frame, pose)

        actions = actions_by_frame.get(frame_idx, [])
        boxes = boxes_by_frame.get(frame_idx, [])

        draw_actions_on_frame(frame, boxes, actions, min_score = 0.1)

        if writer is not None:
            writer.write(frame)

        if show:
            cv2.imshow("pose_viz", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", dest="out_video")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug-joints", action="store_true")
    args = parser.parse_args()

    visualize_from_json(
        video_path=args.video,
        json_path=args.json,
        out_video_path=args.out_video,
        show=args.show,
        debug_joints=args.debug_joints,
    )
    