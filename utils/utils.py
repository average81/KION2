import yaml
import numpy as np
from models.pose_format import Pose

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