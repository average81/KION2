import json
from pathlib import Path
from utils.visualize import draw_pose, debug_draw_joints

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







def visualize_from_json(
    video_path: str | Path,
    json_path: str | Path,
    out_video_path: str | Path | None = None,
    show: bool = False,
    debug_joints: bool = False,
):
    poses_by_frame = load_poses_by_frame(json_path)
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
    