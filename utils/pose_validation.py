"""
Скрипт для валидации качества предсказаний поз с использованием метрики mAP на основе PCK.
"""

import numpy as np
from models.pose_format import Pose, JOINTS
from typing import List, Tuple, Dict
from dataclasses import dataclass
import warnings
from utils import calculate_iou

@dataclass
class Detection:
    """Представление одного обнаруженного скелета."""
    pose: Pose
    confidence: float = 1.0
    def __post_init__(self):
        self.bbox = self._calculate_bbox()

    def _calculate_bbox(self) -> Tuple[float, float, float, float]:
        """
        Вычисляет bounding box вокруг ключевых точек позы.

        Returns:
            (x1, y1, x2, y2) — координаты углов bbox
        """
        keypoints = self.pose.keypoints

        # Фильтруем нулевые (отсутствующие) точки
        valid_points = [kpt for kpt in keypoints if np.any(kpt != 0)]

        if len(valid_points) == 0:
            return (0, 0, 0, 0)

        points_array = np.array(valid_points)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)

        return (x_min, y_min, x_max, y_max)

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

    # Словарь для хранения ошибок по каждому суставу
    joint_errors = {joint_idx: [] for joint_idx in JOINTS.keys()}

    # Группировка поз по frame_idx
    pred_by_frame = {}
    gt_by_frame = {}

    for pose in predicted_poses:
        if pose.frame_idx not in pred_by_frame:
            pred_by_frame[pose.frame_idx] = []
        pred_by_frame[pose.frame_idx].append(pose)

    for pose in ground_truth_poses:
        if pose.frame_idx not in gt_by_frame:
            gt_by_frame[pose.frame_idx] = []
        gt_by_frame[pose.frame_idx].append(pose)

    # Обработка каждого кадра
    common_frames = set(pred_by_frame.keys()) & set(gt_by_frame.keys())

    for frame_idx in common_frames:
        preds = pred_by_frame[frame_idx]
        gts = gt_by_frame[frame_idx]

        # Сопоставляем позы в кадре по максимальному IoU bbox
        matched_gt = set()

        for pred_pose in preds:
            best_iou = 0.0
            best_gt_idx = -1

            pred_bbox = Detection(pred_pose)._calculate_bbox()

            for gt_idx, gt_pose in enumerate(gts):
                if gt_idx in matched_gt:
                    continue

                gt_bbox = Detection(gt_pose)._calculate_bbox()
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Если найдено хорошее совпадение
            if best_gt_idx != -1 and best_iou > 0.1:  # Порог IoU для сопоставления
                gt_pose = gts[best_gt_idx]

                # Вычисляем ошибки по суставам
                for joint_idx in JOINTS.keys():
                    pred_kpt = pred_pose.keypoints[joint_idx]
                    gt_kpt = gt_pose.keypoints[joint_idx]

                    if np.any(gt_kpt != 0):
                        distance = np.linalg.norm(pred_kpt - gt_kpt)
                        joint_errors[joint_idx].append(distance)

                matched_gt.add(best_gt_idx)

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

def match_poses_by_iou_in_frame(
        predictions: List[Detection],
        ground_truths: List[Pose],
        iou_threshold: float = 0.5
) -> List[Tuple[Detection, Pose]]:
    """
    Сопоставляет предсказания и ground truth по максимальному IoU bbox в каждом кадре.

    Использует жадный алгоритм: для каждого предсказания выбирается GT с максимальным IoU,
    при условии, что IoU > порог и GT ещё не сопоставлен.

    Args:
        predictions: Список предсказанных детекций
        ground_truths: Список истинных поз
        iou_threshold: Минимальный IoU для считать совпадением

    Returns:
        Список кортежей (prediction, ground_truth) сопоставленных поз
    """
    # Группируем по кадрам
    pred_by_frame = {}
    gt_by_frame = {}

    for det in predictions:
        frame_idx = det.pose.frame_idx
        if frame_idx not in pred_by_frame:
            pred_by_frame[frame_idx] = []
        pred_by_frame[frame_idx].append(det)

    for pose in ground_truths:
        frame_idx = pose.frame_idx
        if frame_idx not in gt_by_frame:
            gt_by_frame[frame_idx] = []
        gt_by_frame[frame_idx].append(pose)

    matches = []

    # Обрабатываем каждый кадр
    all_frame_indices = set(pred_by_frame.keys()) | set(gt_by_frame.keys())

    for frame_idx in all_frame_indices:
        preds = pred_by_frame.get(frame_idx, [])
        gts = gt_by_frame.get(frame_idx, [])

        if len(preds) == 0 or len(gts) == 0:
            continue

        # Сортируем предсказания по уверенности (убывание)
        sorted_preds = sorted(preds, key=lambda x: x.confidence, reverse=True)

        matched_gt = [False] * len(gts)

        # Для каждого предсказания ищем лучшее совпадение
        for pred_det in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_pose in enumerate(gts):
                if matched_gt[gt_idx]:
                    continue

                iou = calculate_iou(pred_det.bbox, Detection(gt_pose).bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Если найдено хорошее совпадение
            if best_gt_idx != -1 and best_iou >= iou_threshold:
                matches.append((pred_det, gts[best_gt_idx]))
                matched_gt[best_gt_idx] = True

    return matches

def calculate_pck_at_threshold(
        matched_pairs: List[Tuple[Detection, Pose]],
        threshold: float,
        reference_distance: float = None
) -> float:
    """
    Вычисляет Percentage of Correct Keypoints (PCK) при заданном пороге.

    PCK — доля ключевых точек, где ошибка меньше порога.

    Args:
        matched_pairs: Список кортежей (prediction, ground_truth)
        threshold: Пороговое расстояние в пикселях
        reference_distance: Опорное расстояние (например, длина тела). Если None, используется абсолютный порог.

    Returns:
        float: Значение PCK в диапазоне [0, 1]
    """
    if len(matched_pairs) == 0:
        return 0.0

    correct_keypoints = 0
    total_keypoints = 0

    for pred_det, gt_pose in matched_pairs:
        pred_pose = pred_det.pose

        for joint_idx in JOINTS.keys():
            pred_kpt = pred_pose.keypoints[joint_idx]
            gt_kpt = gt_pose.keypoints[joint_idx]

            # Пропускаем отсутствующие точки
            if np.any(gt_kpt == 0):
                continue

            # Вычисляем расстояние
            distance = np.linalg.norm(pred_kpt - gt_kpt)

            # Проверяем порог
            if reference_distance is not None:
                normalized_distance = distance / reference_distance
                is_correct = normalized_distance <= threshold
            else:
                is_correct = distance <= threshold

            if is_correct:
                correct_keypoints += 1
            total_keypoints += 1

    return correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0

def calculate_map_from_pck(
        predictions: List[Detection],
        ground_truths: List[Pose],
        thresholds: List[float] = None,
        reference_distance: float = None,
        iou_matching_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Вычисляет mean Average Precision (mAP) на основе PCK при разных порогах.

    В контексте поз, mAP можно определить как среднее значение PCK
    по разным пороговым значениям.

    Args:
        predictions: Предсказанные детекции
        ground_truths: Истинные позы
        thresholds: Список порогов для усреднения (в пикселях)
        reference_distance: Опорное расстояние для нормализации (например, длина тела)
        iou_matching_threshold: Порог IoU для сопоставления поз в кадре

    Returns:
        Словарь с результатами mAP
    """
    if thresholds is None:
        # По умолчанию используем пороги от 5 до 50 пикселей с шагом 5
        thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    if len(predictions) == 0 or len(ground_truths) == 0:
        warnings.warn("Один из наборов поз пустой")
        return {"mAP": 0.0, "PCK_per_threshold": {}}

    # Сопоставляем позы по максимальному IoU bbox в каждом кадре
    matched_pairs = match_poses_by_iou_in_frame(predictions, ground_truths, iou_matching_threshold)

    if len(matched_pairs) == 0:
        warnings.warn("Не найдено ни одной пары поз с достаточным IoU bbox")
        return {"mAP": 0.0, "PCK_per_threshold": {}}

    # Вычисляем PCK для каждого порога
    pck_values = {}
    for thresh in thresholds:
        pck = calculate_pck_at_threshold(matched_pairs, thresh, reference_distance)
        pck_values[f"PCK@{thresh}px"] = pck

    # Вычисляем mAP как среднее PCK по всем порогам
    map_value = np.mean(list(pck_values.values()))

    results = {
        "mAP": map_value,
        "PCK_per_threshold": pck_values,
        "thresholds_px": thresholds,
        "reference_distance": reference_distance,
        "iou_matching_threshold": iou_matching_threshold,
        "n_matched_pairs": len(matched_pairs),
        "n_predictions": len(predictions),
        "n_ground_truth": len(ground_truths)
    }

    return results

def validate_poses_with_map(predicted_file: str, ground_truth_file: str) -> dict:
    """
    Основная функция для валидации поз с использованием mAP на основе PCK.

    Args:
        predicted_file: Путь к файлу с предсказанными позами
        ground_truth_file: Путь к файлу с истинными позами

    Returns:
        Словарь с результатами валидации
    """
    print(f"Начинаем валидацию поз с метрикой mAP (на основе PCK)...")
    print(f"Предсказанные позы: {predicted_file}")
    print(f"Истинные позы: {ground_truth_file}")

    # Загружаем данные
    predicted_poses = load_pose_data(predicted_file)
    ground_truth_poses = load_pose_data(ground_truth_file)

    if len(predicted_poses) == 0 or len(ground_truth_poses) == 0:
        raise ValueError("Не удалось загрузить данные из файлов")

    print(f"Загружено {len(predicted_poses)} предсказанных поз")
    print(f"Загружено {len(ground_truth_poses)} истинных поз")

    # Фильтрация по уверенности
    detections = []
    for pose in predicted_poses:
        # Используем среднюю уверенность как confidence детекции
        if hasattr(pose, 'keypoints_conf'):
            conf = np.mean([c for c in pose.keypoints_conf if c > 0])
        else:
            conf = 1.0

        detections.append(Detection(pose=pose, confidence=conf))

    # Вычисляем mAP на основе PCK
    map_results = calculate_map_from_pck(detections, ground_truth_poses)

    # Вычисляем MPJPE
    try:
        overall_mpjpe, mpjpe_per_joint = calculate_mpjpe(predicted_poses, ground_truth_poses)
        map_results["overall_mpjpe"] = overall_mpjpe
        map_results["mpjpe_per_joint"] = mpjpe_per_joint
    except Exception as e:
        print(f"Предупреждение: не удалось вычислить MPJPE: {e}")
        map_results["overall_mpjpe"] = float('nan')
        map_results["mpjpe_per_joint"] = {}

    print(f"\nmAP (средний PCK): {map_results['mAP']:.4f}")

    print("\nPCK при разных порогах:")
    for thresh, pck in map_results["PCK_per_threshold"].items():
        print(f"{thresh}: {pck:.4f}")

    if "overall_mpjpe" in map_results:
        print(f"\nОбщее MPJPE: {map_results['overall_mpjpe']:.4f} пикселей")

    return map_results

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

# Обновляем основной блок запуска
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Валидация качества предсказаний поз')
    parser.add_argument('predicted_file', type=str,
                        help='Путь к файлу с предсказанными позами')
    parser.add_argument('ground_truth_file', type=str,
                        help='Путь к файлу с истинными позами (ground truth)')
    parser.add_argument('--output', type=str, default='pose_validation_results.json',
                        help='Путь к файлу для сохранения результатов')
    parser.add_argument('--metric', type=str, choices=['mpjpe', 'map', 'both'], default='both',
                        help='Метрика для вычисления: mpjpe, map или both')

    args = parser.parse_args()

    try:
        if args.metric == 'mpjpe':
            results = validate_poses(args.predicted_file, args.ground_truth_file)
        elif args.metric == 'map':
            results = validate_poses_with_map(args.predicted_file, args.ground_truth_file)
        else:  # both
            print("Вычисление MPJPE...")
            mpjpe_results = validate_poses(args.predicted_file, args.ground_truth_file)
            print("\nВычисление mAP...")
            map_results = validate_poses_with_map(args.predicted_file, args.ground_truth_file)
            # Объединяем результаты
            results = {**mpjpe_results, **{k: v for k, v in map_results.items() if k not in mpjpe_results}}

        # Сохраняем результаты в файл
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nРезультаты сохранены в {args.output}")

    except Exception as e:
        print(f"Ошибка при валидации: {e}")
        raise