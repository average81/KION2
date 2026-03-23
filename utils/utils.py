import yaml
import numpy as np
import torch
from models.pose_format import Pose
from pathlib import Path
import json

#Открытие файла настроек в  yaml
def open_yaml(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

#Сохранение yaml
def save_yaml(file, data):
    with open(file, 'w') as f:
        yaml.dump(data, f)

# Конвертер объектов в списки
def numpy_to_builtin(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(numpy_to_builtin(v) for v in obj)
    return obj

def read_ntu_pose_file(file_path, max_bodies=4, njoints=25):
    """
    Читает файл поз из датасета NTU-RGB+D 120 и возвращает список объектов Pose.
    
    Args:
        file_path (str): Путь к файлу с данными позы
        max_bodies (int): Максимальное количество тел в последовательности
        njoints (int): Количество суставов
    
    Returns:
        list: Список объектов Pose, каждый из которых содержит данные о позе
    """
    with open(file_path, 'r') as f:
        datas = f.readlines()
    
    nframe = int(datas[0][:-1])

    
    # Создаем список для хранения всех поз
    poses = []
    
    cursor = 0
    
    for frame_idx in range(nframe):
        cursor += 1  # Переходим к следующей строке
        bodycount = int(datas[cursor][:-1])
        
        # Обрабатываем каждое тело в кадре
        for body in range(bodycount):
            if body >= max_bodies:
                break  # Пропускаем избыточные тела
                
            cursor += 1  # Пропускаем строку с информацией о теле
            cursor += 1  # Пропускаем строку с количеством суставов
            
            # Создаем объект Pose для текущего тела
            pose = Pose()
            pose.id = body
            pose.frame_idx = frame_idx
            
            # Читаем данные для каждого сустава
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                
                # Используем координаты RGB (индексы 5,6) как x,y координаты
                x, y = jointinfo[5:7]

                our_joint_idx = joint   # Описание точек совпадает с этим датасетом в pose_format.py
                pose.keypoints[our_joint_idx] = [x, y]
                pose.keypoints_conf[our_joint_idx] = 1.0  # Предполагаем полную уверенность
                # Для суставов, отсутствующих в NTU, значения останутся нулевыми
            
            poses.append(pose)
    
    return poses

# Чтение поз нашего формата
def load_own_poses(json_path: str | Path):
    json_path = Path(json_path)
    data = None
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Расчет IoU двух прямоугольников
def calculate_iou(box1, box2):
    """
    Рассчитывает Intersection over Union (IoU) между двумя прямоугольниками.

    Args:
        box1 (list or tuple): [x1, y1, x2, y2] — координаты двух углов первого прямоугольника
        box2 (list or tuple): [x1, y1, x2, y2] — координаты двух углов второго прямоугольника

    Returns:
        float: Значение IoU в диапазоне [0, 1]
    """
    # Распаковываем координаты
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Приводим к корректному порядку (min, max)
    x1_min, x1_max = min(x1_min, x1_max), max(x1_min, x1_max)
    y1_min, y1_max = min(y1_min, y1_max), max(y1_min, y1_max)
    x2_min, x2_max = min(x2_min, x2_max), max(x2_min, x2_max)
    y2_min, y2_max = min(y2_min, y2_max), max(y2_min, y2_max)

    # Находим пересечение
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Ширина и высота пересечения
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    # Площадь пересечения
    inter_area = inter_width * inter_height

    # Площади каждого прямоугольника
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Площадь объединения
    union_area = area1 + area2 - inter_area

    # Защита от деления на ноль
    if union_area == 0:
        return 0.0

    # Возвращаем IoU
    return inter_area / union_area