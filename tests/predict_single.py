import os
import sys
import logging
from pathlib import Path
from utils.utils import open_yaml
from app.video_processor import VideoProcessor
from models.action_format import ACTIONS
import torch

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_action(video_path, config_path, output_dir="output", top_k=5):
    """
    Обработка одного видеофайла и предсказание действия.

    Args:
        video_path (str): Путь к видеофайлу
        config_path (str): Путь к конфигурационному файлу
        output_dir (str): Каталог для сохранения выходных данных (опционально)
    """
    # Проверка входных файлов
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        return None

    if not os.path.exists(config_path):
        logger.error(f"Config file does not exist: {config_path}")
        return None

    # Загрузка конфигурации
    config = open_yaml(config_path)
    min_duration = config.get('act_frames', 30)  # Минимальная длительность действия в кадрах

    logger.info(f"Processing single video: {video_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Minimum action duration: {min_duration} frames")

    # Создание директории для вывода
    os.makedirs(output_dir, exist_ok=True)

    # Инициализация VideoProcessor
    processor = VideoProcessor(
        input_file=video_path,
        output_dir=output_dir,
        verbose=True,
        config_path=config_path
    )

    try:
        # Обработка видео
        results,raw_res = processor.process()

        if results['pose_actions'] is None or len(results['pose_actions']) == 0:
            logger.warning("No actions detected in the video.")
            return None

        # Фильтрация по длительности
        valid_predictions = [
            act for act in results['pose_actions']
        ]

        if not valid_predictions:
            logger.warning(f"No actions meet minimum duration of {min_duration} frames.")
            return None

        # Выбор действия с наибольшей уверенностью
        best_prediction = max(valid_predictions, key=lambda x: x['action'].get('conf', 0))
        action_data = best_prediction['action']
        action_id = action_data['action_id']
        confidence = action_data['conf']
        label = ACTIONS.get(action_id, "Unknown Action")
        valid_logits = [
            res["logits"] for res in raw_res
        ]
        all_preds = []
        for logits in valid_logits:
            prob = torch.softmax(logits, dim=1)[0]  # Вероятности для одного кадра
            topk_vals, topk_ids = torch.topk(prob, k=top_k)
            for val, idx in zip(topk_vals, topk_ids):
                all_preds.append((val.item(), idx.item()))

        # Сортировка по значению (confidence) и выбор топ-K
        sorted_topk = sorted(all_preds, key=lambda x: x[0], reverse=True)[:top_k]
        
        # Перекодировка индексов через actions_mapping, если он существует
        if (hasattr(processor.pose_action_classifier.model, 'actions_mapping') and
                processor.pose_action_classifier.model.actions_mapping is not None):
            sorted_topk = [
                (conf, processor.pose_action_classifier.model.actions_mapping.get(act_id, act_id))
                for conf, act_id in sorted_topk
            ]

        # Вывод результата
        print("\n" + "="*60)
        print("ACTION PREDICTION RESULT")
        print("="*60)
        print(f"Video file:         {Path(video_path).name}")
        print(f"Predicted action:   {label}")
        print(f"Action ID:          {action_id}")
        print(f"Confidence:         {confidence:.4f}")
        print(f"Start frame:        {best_prediction['frame_idx']}")
        print(f"Output saved to:    {output_dir}")

        # Вывод топ-K предсказаний
        print(f"\nTop {len(sorted_topk)} predictions:")
        print("-" * 40)
        for i, (conf, act_id) in enumerate(sorted_topk, 1):
            lbl = ACTIONS.get(act_id, "Unknown Action")
            print(f"{i:2d}. {lbl:<30} (ID: {act_id}, Conf: {conf:.4f})")

        return {
            "video": Path(video_path).name,
            "action_id": action_id,
            "action_label": label,
            "confidence": confidence,
            "start_frame": best_prediction['frame_idx'],
            "top_k_predictions": [
                {
                    "rank": i+1,
                    "action_id": act_id,
                    "label": ACTIONS.get(act_id, "Unknown Action"),
                    "confidence": conf
                }
                for i, (conf, act_id) in enumerate(sorted_topk)
            ],
            "raw_results": results
        }

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict_single.py <video_file> <config_path> [output_dir]")
        print("Example: python predict_single.py ./videos/S001C001P001R001A002_rgb.avi ./config.yml output/")
        sys.exit(1)

    video_file = sys.argv[1]
    config_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else "output"

    result = predict_action(video_file, config_file, output_directory)

    if result:
        logger.info("Prediction completed successfully.")
    else:
        logger.error("Prediction failed or no action detected.")