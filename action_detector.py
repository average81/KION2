import argparse
import logging
import os
import utils.utils as utils
from app.video_processor import VideoProcessor

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
    logger.info("Starting video processing...")
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_dir} does not exist.")
        exit()
    processor=VideoProcessor(args.input_file,output_dir=args.output_dir,verbose = args.verbose,
                             config_path =args.config_path)
