"""Разбор одного файла NTU RGB+D .skeleton (2D color)."""
import math
from pathlib import Path

NUM_JOINT = 25

def read_skeleton_file(path):
    """Читает один .skeleton и возвращает список кадров.
    frames[t] = [body1, body2, ...], body['joints'] = список длины 25
    joint = {'colorX': float, 'colorY': float}
    """
    path = Path(path)
    with path.open('r') as f:
        num_frames = int(f.readline())
        frames = []
        for _ in range(num_frames):
            num_bodies = int(f.readline())
            bodies = []
            for _ in range(num_bodies):
                body_info = f.readline().split()   # можно игнорировать
                num_joints = int(f.readline())
                joints = []
                for _ in range(num_joints):
                    items = f.readline().split()
                    # формат NTU: x,y,z, depthX,depthY,colorX,colorY,...
                    if len(items) < 7:
                        colorX, colorY = 0.0, 0.0
                    else:
                        colorX = float(items[5])
                        colorY = float(items[6])
                    # nan/inf в .skeleton дают nan в numpy → loss nan при обучении
                    if not (math.isfinite(colorX) and math.isfinite(colorY)):
                        colorX, colorY = 0.0, 0.0
                    joints.append({"colorX": colorX, "colorY": colorY})
                bodies.append({"joints": joints})
            frames.append(bodies)
    return frames