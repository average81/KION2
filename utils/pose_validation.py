"""
Скрипт для валидации качества предсказаний поз с использованием метрики MPJPE
(Mean Per Joint Position Error).
"""

import numpy as np
from models.pose_format import Pose, JOINTS
from typing import List, Tuple

def load_pose_data(file_path: str) -> List[Pose]:
    """
    Загружает данные поз из файла.
    
    Поддерживаемые форматы: JSON, NPZ, TXT (NTU-RGB+D)
    
    Args:
        file_path: Путь к файлу с данными поз
    
    Returns:
        Список объектов Pose
    
    Raises:
        ValueError: Если формат файла не поддерживается или данные повреждены
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.json':
        return _load_pose_json(file_path)
    elif ext == '.npz':
        return _load_pose_npz(file_path)
    elif ext == '.txt':
        from utils.utils import read_ntu_pose_file
        return read_ntu_pose_file(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}. Поддерживаются: .json, .npz, .txt")

def _load_pose_json(file_path: str) -> List[Pose]:
    """
    Загружает позы из JSON файла.
    
    Ожидается формат, совместимый с VideoProcessor.
    """
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    poses = []
    raw_poses = data.get('raw_poses', [])

    for pose_data in raw_poses:
        pose = Pose()
        pose.frame_idx = pose_data['frame_idx']
        pose.id = pose_data['person_id']
        
        # Конвертируем keypoints в numpy массив
        keypoints = np.array(pose_data['keypoints'])
        
        pose.keypoints = keypoints
        
        # Создаем массив confidence
        pose.keypoints_conf = pose_data['keypoints_conf']

        poses.append(pose)
    
    return poses

def _load_pose_npz(file_path: str) -> List[Pose]:
    """
    Загружает позы из NPZ файла.
    
    Ожидается, что файл содержит массивы 'keypoints' и 'keypoints_conf'.
    """
    data = np.load(file_path)
    
    poses = []
    keypoints_array = data['keypoints']  # Ожидается форма (N, K, 2) или (N, K, 3)
    
    # Проверяем форму массива
    if len(keypoints_array.shape) != 3:
        raise ValueError(f"Неправильная форма массива keypoints: {keypoints_array.shape}")
    
    n_frames, n_joints, n_coords = keypoints_array.shape
    
    # Извлекаем confidence, если он есть
    if n_coords == 3:
        keypoints = keypoints_array[:, :, :2]
        keypoints_conf = keypoints_array[:, :, 2]
    else:
        keypoints = keypoints_array
        keypoints_conf = np.ones((n_frames, n_joints))

    for i in range(n_frames):
        pose = Pose()
        pose.frame_idx = i
        pose.id = 0  # По умолчанию один человек
        pose.keypoints = keypoints[i]
        pose.keypoints_conf = keypoints_conf[i]
        poses.append(pose)
    
    return poses

def calculate_mpjpe(predicted_poses: List[Pose], ground_truth_poses: List[Pose]) -> Tuple[float, dict]:
    """
    Вычисляет метрику Mean Per Joint Position Error (MPJPE) между предсказанными
    и истинными позами.
    
    MPJPE вычисляется как среднее евклидово расстояние между соответствующими
    суставами в предсказанных и истинных позах.
    
    Args:
        predicted_poses: Список предсказанных поз
        ground_truth_poses: Список истинных поз (ground truth)
    
    Returns:
        Кортеж из:
        - Среднее значение MPJPE по всем суставам и кадрам
        - Словарь с MPJPE по каждому суставу
    """
    if len(predicted_poses) == 0 or len(ground_truth_poses) == 0:
        raise ValueError("Списки поз не могут быть пустыми")
    
    if len(predicted_poses) != len(ground_truth_poses):
        print(f"Предупреждение: разное количество поз в наборах: {len(predicted_poses)} vs {len(ground_truth_poses)}")
        # Обрезаем до минимального размера
        #min_len = min(len(predicted_poses), len(ground_truth_poses))
        #predicted_poses = predicted_poses[:min_len]
        #ground_truth_poses = ground_truth_poses[:min_len]
    
    # Словарь для хранения ошибок по каждому суставу
    joint_errors = {joint_idx: [] for joint_idx in JOINTS.keys()}
    
    # Сравниваем соответствующие позы

    for pred_pose in predicted_poses:
        gt_pose = None
        for gt_pose in ground_truth_poses:
            if pred_pose.frame_idx == gt_pose.frame_idx and pred_pose.id==gt_pose.id:
                break

        if gt_pose == None:
            break

        # Проверяем, что оба объекта имеют необходимые атрибуты
        if not hasattr(pred_pose, 'keypoints') or not hasattr(gt_pose, 'keypoints'):
            continue
            
        # Проверяем размерность keypoints
        if pred_pose.keypoints.shape != gt_pose.keypoints.shape:
            continue
            
        # Вычисляем евклидово расстояние между соответствующими суставами
        for joint_idx in JOINTS.keys():
            pred_kpt = pred_pose.keypoints[joint_idx]
            gt_kpt = gt_pose.keypoints[joint_idx]
            
            # Проверяем, что координаты не нулевые (отсутствующие суставы)
            if np.any(gt_kpt != 0):
                distance = np.linalg.norm(pred_kpt - gt_kpt)
                joint_errors[joint_idx].append(distance)
    
    # Вычисляем среднее значение MPJPE по каждому суставу
    mpjpe_per_joint = {}
    total_errors = []
    
    for joint_idx, errors in joint_errors.items():
        if len(errors) > 0:
            mpjpe_per_joint[joint_idx] = np.mean(errors)
            total_errors.extend(errors)
        else:
            mpjpe_per_joint[joint_idx] = float('nan')
    
    # Вычисляем общее среднее MPJPE
    overall_mpjpe = np.mean(total_errors) if total_errors else float('nan')
    
    return overall_mpjpe, mpjpe_per_joint

def validate_poses(predicted_file: str, ground_truth_file: str) -> dict:
    """
    Основная функция для валидации поз.
    
    Загружает два набора данных и вычисляет метрики качества.
    
    Args:
        predicted_file: Путь к файлу с предсказанными позами
        ground_truth_file: Путь к файлу с истинными позами
    
    Returns:
        Словарь с результатами валидации
    """
    print(f"Начинаем валидацию поз...")
    print(f"Предсказанные позы: {predicted_file}")
    print(f"Истинные позы: {ground_truth_file}")
    
    # Загружаем данные
    predicted_poses = load_pose_data(predicted_file)
    ground_truth_poses = load_pose_data(ground_truth_file)
    
    if len(predicted_poses) == 0 or len(ground_truth_poses) == 0:
        raise ValueError("Не удалось загрузить данные из файлов")
    
    print(f"Загружено {len(predicted_poses)} предсказанных поз")
    print(f"Загружено {len(ground_truth_poses)} истинных поз")
    
    # Вычисляем MPJPE
    overall_mpjpe, mpjpe_per_joint = calculate_mpjpe(predicted_poses, ground_truth_poses)
    
    # Формируем результаты
    results = {
        "overall_mpjpe": overall_mpjpe,
        "mpjpe_per_joint": {},
        "joint_names": {}
    }
    
    # Преобразуем индексы суставов в имена
    for joint_idx, mpjpe in mpjpe_per_joint.items():
        joint_name = JOINTS[joint_idx]
        results["mpjpe_per_joint"][joint_name] = mpjpe
        results["joint_names"][joint_name] = joint_idx
    
    print(f"Общее MPJPE: {overall_mpjpe:.4f} пикселей")
    
    # Выводим ошибки по суставам
    print("\nMPJPE по суставам:")
    for joint_name in sorted(results["mpjpe_per_joint"]):
        mpjpe = results["mpjpe_per_joint"][joint_name]
        if not np.isnan(mpjpe):
            print(f"{joint_name:20}: {mpjpe:.4f} пикселей")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Валидация качества предсказаний поз')
    parser.add_argument('predicted_file', type=str, 
                        help='Путь к файлу с предсказанными позами')
    parser.add_argument('ground_truth_file', type=str, 
                        help='Путь к файлу с истинными позами (ground truth)')
    parser.add_argument('--output', type=str, default='pose_validation_results.json',
                        help='Путь к файлу для сохранения результатов (по умолчанию: pose_validation_results.json)')
    
    args = parser.parse_args()
    
    try:
        results = validate_poses(
            args.predicted_file,
            args.ground_truth_file
        )
        
        # Сохраняем результаты в файл
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nРезультаты сохранены в {args.output}")
        
    except Exception as e:
        print(f"Ошибка при валидации: {e}")
