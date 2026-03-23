import argparse
import logging
import os
import time
import utils.utils as utils
from app.video_processor import VideoProcessor
import json
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input video file to process.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory to save results.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--config_path", type=str, default="config.yml", help="Path to the configuration file.")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    start_time = time.time()
    logger.info("Starting video processing...")
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} does not exist.")
        exit()
    processor=VideoProcessor(args.input_file,output_dir=args.output_dir,verbose = args.verbose,
                             config_path =args.config_path)
    result,_ = processor.process()
    serializable_result = utils.numpy_to_builtin(result)
    # Сохранение результатов в JSON (аналогично test_poses.py)
    output_json = os.path.join(args.output_dir, "results.json")
    # Создаем директорию для сохранения
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Results saved to {output_json}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")