import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
from app.video_processor import VideoProcessor
import utils.utils as utils


def extract_label_from_filename(filename):
    """
    Извлекает индекс действия из имени файла.
    Ожидается формат: SxxxCxxPxxRxxAxxx_rgb.ext
    """
    base_name = Path(filename).stem  # Убираем расширение
    if 'A' in base_name:
        try:
            # Находим часть после 'A' и берем первые 3 цифры
            action_part = base_name.split('A')[1]
            action_id = int(action_part[:3])
            return action_id - 1  # Преобразуем в 0-индексацию
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not extract label from {filename}: {e}")
            return None
    else:
        logging.warning(f"No 'A' found in filename {filename}")
        return None

def calculate_metrics(y_true, y_pred, classes):
    """
    Рассчитывает метрики качества классификации.
    """
    if len(y_true) == 0:
        logging.warning("No valid predictions to calculate metrics.")
        return {}
    
    # Основные метрики
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    metrics = {
        'classification_report': report,
        'macro_avg_precision': float(precision),
        'macro_avg_recall': float(recall),
        'macro_avg_f1': float(f1),
        'accuracy': float(report['accuracy'])
    }
    
    return metrics

def main(video_folder, config_path, output_json='validation_results.json'):
    """
    Основная функция валидации.
    
    Args:
        video_folder (str): Путь к папке с видеофайлами
        config_path (str): Путь к файлу конфигурации
        output_json (str): Имя выходного JSON файла
    """
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Проверка входных данных
    if not os.path.exists(video_folder):
        logger.error(f"Video folder does not exist: {video_folder}")
        sys.exit(1)
        
    if not os.path.exists(config_path):
        logger.error(f"Config file does not exist: {config_path}")
        sys.exit(1)
        
    # Загрузка конфигурации
    config = utils.open_yaml(config_path)
    min_duration = config.get('act_frames', 30)  # Минимальная длительность действия в кадрах
    
    logger.info(f"Starting validation...")
    logger.info(f"Video folder: {video_folder}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Minimum action duration: {min_duration} frames")
    
    # Список видеофайлов
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        logger.warning(f"No video files found in {video_folder}")
        sys.exit(0)
        
    logger.info(f"Found {len(video_files)} video files")
    
    # Инициализация VideoProcessor (с первым видео для загрузки моделей)
    first_video_path = os.path.join(video_folder, video_files[0])
    processor = VideoProcessor(
        input_file=first_video_path,
        output_dir="temp_output",
        verbose=False,
        config_path=config_path
    )
    
    # Список классов (предполагаем, что их 60)
    classes = [
        "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
        "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping",
        "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
        "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
        "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
        "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up",
        "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
        "pointing to something with finger", "taking a selfie", "check time (from watch)",
        "rub two hands together", "nod head/bow", "shake head", "wipe face",
        "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough",
        "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
        "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition",
        "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
        "kicking other person", "pushing other person", "pat on back of other person",
        "point finger at the other person", "hugging other person", "giving something to other person",
        "touch other person's pocket", "handshaking", "walking towards each other",
        "walking apart from each other"
    ]
    
    # Сбор предсказаний и истинных меток
    y_true = []
    y_pred = []
    processed_videos = 0
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        
        try:
            # Извлечение истинной метки из имени файла
            true_label = extract_label_from_filename(video_file)
            if true_label is None:
                logger.warning(f"Skipping {video_file}: could not extract label")
                continue
                
            # Обработка видео
            logger.info(f"Processing {video_file}...")
            processor.input_file = video_path
            results = processor.process()
            
            # Проверка результатов
            if results['pose_actions'] is None:
                logger.warning(f"No pose actions detected for {video_file}")
                continue
                
            # Фильтрация предсказаний по длительности
            valid_predictions = []
            for action in results['pose_actions']:
                duration = action.get('end_frame', 0) - action.get('start_frame', 0)
                if duration >= min_duration:
                    valid_predictions.append(action)
                
            if not valid_predictions:
                logger.warning(f"No valid predictions (duration >= {min_duration}) for {video_file}")
                continue
                
            # Берем предсказание с максимальной уверенностью среди валидных
            best_prediction = max(valid_predictions, key=lambda x: x.get('confidence', 0))
            pred_label = best_prediction.get('predicted_action', -1)
            
            if pred_label == -1:
                logger.warning(f"No predicted action in best prediction for {video_file}")
                continue
                
            y_true.append(true_label)
            y_pred.append(pred_label)
            processed_videos += 1
            
            logger.info(f"{video_file}: true={true_label}, pred={pred_label}, confidence={best_prediction.get('confidence', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
            continue
    
    # Рассчет метрик
    logger.info(f"\nProcessed {processed_videos} out of {len(video_files)} videos")
    
    if y_true:
        metrics = calculate_metrics(y_true, y_pred, classes)
        
        # Вывод на экран
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(f"Processed videos: {processed_videos}/{len(video_files)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Avg Precision: {metrics['macro_avg_precision']:.4f}")
        print(f"Macro Avg Recall: {metrics['macro_avg_recall']:.4f}")
        print(f"Macro Avg F1-Score: {metrics['macro_avg_f1']:.4f}")
        print("\nClassification Report:")
        for class_name, class_idx in zip(classes, range(len(classes))):
            if str(class_idx) in metrics['classification_report']:
                cls_metrics = metrics['classification_report'][str(class_idx)]
                print(f"{class_name:40s} (#{class_idx:2d}): "
                      f"Prec={cls_metrics['precision']:5.3f} "
                      f"Rec={cls_metrics['recall']:5.3f} "
                      f"F1={cls_metrics['f1-score']:5.3f} "
                      f"Sup={cls_metrics['support']:3.0f}")
        
        # Сохранение в JSON
        output_data = {
            'config': config,
            'video_folder': video_folder,
            'total_videos': len(video_files),
            'processed_videos': processed_videos,
            'metrics': metrics,
            'predictions': [
                {"video": video_files[i], "true_label": y_true[i], "pred_label": y_pred[i]}
                for i in range(len(y_true))
            ]
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_json}")
        
    else:
        logger.warning("No valid predictions to calculate metrics.")
        print("No valid predictions to calculate metrics.")
        
    logger.info("Validation completed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python valid_proj.py <video_folder> <config_path> [output_json]")
        print("Example: python valid_proj.py ./videos ./config.yml results.json")
        sys.exit(1)
        
    video_folder = sys.argv[1]
    config_path = sys.argv[2]
    output_json = sys.argv[3] if len(sys.argv) > 3 else 'validation_results.json'
    
    main(video_folder, config_path, output_json)