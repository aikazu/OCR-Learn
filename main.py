import argparse
import cv2
import os
import json
import sys
import logging
import configparser

# Import functions and constants from our modules
from ocr_processor import perform_ocr, set_tesseract_path
from image_utils import (
    preprocess_image, highlight_text, DEFAULT_TARGET_DPI,
    DEFAULT_ADAPTIVE_BLOCK_SIZE, DEFAULT_ADAPTIVE_C # Import defaults
)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration File Handling ---
CONFIG_FILE_NAME = 'ocr_config.ini'

def load_config():
    config = configparser.ConfigParser()
    # Define defaults in case file/sections/keys are missing
    config_defaults = {
        'Defaults': {
            'language': 'eng'
        },
        'Tesseract': {
            'command_path': ''
        },
        'Preprocessing': {
            'adaptive_block_size': str(DEFAULT_ADAPTIVE_BLOCK_SIZE),
            'adaptive_c': str(DEFAULT_ADAPTIVE_C)
        }
    }
    config.read_dict(config_defaults)

    if os.path.exists(CONFIG_FILE_NAME):
        logger.info(f"Reading configuration from {CONFIG_FILE_NAME}...")
        try:
            # read() merges with existing defaults
            config.read(CONFIG_FILE_NAME)
        except configparser.Error as e:
            logger.warning(f"Could not parse configuration file {CONFIG_FILE_NAME}: {e}. Using defaults.")
    else:
        logger.info(f"Configuration file {CONFIG_FILE_NAME} not found. Using defaults.")

    return config

def main():
    # Load configuration first
    config = load_config()
    # Get defaults from config, falling back to hardcoded defaults if necessary
    default_lang = config.get('Defaults', 'language', fallback='eng')
    tesseract_path = config.get('Tesseract', 'command_path', fallback='')
    default_block_size = config.getint('Preprocessing', 'adaptive_block_size', fallback=DEFAULT_ADAPTIVE_BLOCK_SIZE)
    default_c_value = config.getint('Preprocessing', 'adaptive_c', fallback=DEFAULT_ADAPTIVE_C)

    # Set Tesseract path if specified in config
    if tesseract_path:
        logger.info(f"Setting Tesseract command path from config: {tesseract_path}")
        set_tesseract_path(tesseract_path)

    parser = argparse.ArgumentParser(
        description='Perform OCR on an image file.',
        epilog=f'Configuration is read from {CONFIG_FILE_NAME} if it exists. \
                 Command-line arguments override configuration file settings.'
    )
    # --- General Arguments ---
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to save the output text/JSON file. If not specified, prints to console.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose (DEBUG level) logging.')

    # --- OCR Arguments ---
    ocr_group = parser.add_argument_group('OCR Options')
    ocr_group.add_argument('-l', '--lang', type=str, default=default_lang,
                         help=f'Language code(s) for Tesseract OCR (e.g., eng, eng+fra). Default: {default_lang} (from config or hardcoded: eng)')
    ocr_group.add_argument('-f', '--format', type=str, default='text', choices=['text', 'json'],
                         help='Output format (text or json). Default: text')

    # --- Preprocessing Arguments ---
    preproc_group = parser.add_argument_group('Preprocessing Options')
    preproc_group.add_argument('--preprocess', action='store_true',
                           help='Enable the full image preprocessing pipeline.')
    preproc_group.add_argument('--target-dpi', type=int, default=None,
                           help=f'Target DPI for image resizing during preprocessing. Requires --preprocess. \
                                Resizing skipped if omitted.')
    preproc_group.add_argument('--adaptive-block-size', type=int, default=default_block_size,
                           help=f'Block size for adaptive thresholding (must be odd, >1). Requires --preprocess. \
                                Default: {default_block_size} (from config or hardcoded: {DEFAULT_ADAPTIVE_BLOCK_SIZE})')
    preproc_group.add_argument('--adaptive-c', type=int, default=default_c_value,
                           help=f'Constant C subtracted during adaptive thresholding. Requires --preprocess. \
                                Default: {default_c_value} (from config or hardcoded: {DEFAULT_ADAPTIVE_C})')

    # --- Highlighting Arguments ---
    highlight_group = parser.add_argument_group('Highlighting Options')
    highlight_group.add_argument('--highlight', type=str,
                               help='Path to save the image with highlighted text boxes. Requires --format json.')

    args = parser.parse_args()

    # Adjust logging level if verbose flag is set
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # --- Input Validation ---
    logger.debug(f"Parsed arguments: {args}")
    if not os.path.exists(args.image_path):
        logger.error(f"Input image file not found: {args.image_path}")
        sys.exit(1)

    if args.highlight and args.format != 'json':
        logger.error("Highlighting requires JSON output format (--format json).")
        sys.exit(1)

    if args.target_dpi is not None and not args.preprocess:
        logger.warning("--target-dpi requires --preprocess to be enabled. Resizing will be skipped.")
        args.target_dpi = None
    if (args.adaptive_block_size != default_block_size or args.adaptive_c != default_c_value) and not args.preprocess:
         logger.warning("--adaptive-block-size and --adaptive-c require --preprocess. Values will be ignored.")

    # --- Load Image ---
    logger.info(f"Loading image: {args.image_path}...")
    image = cv2.imread(args.image_path)
    if image is None:
        logger.error(f"Could not load image file: {args.image_path}. \
                     Please check that the path is correct and the file is a valid, \
                     uncorrupted image format supported by OpenCV (e.g., PNG, JPG, TIFF, BMP).")
        sys.exit(1)
    logger.info(f"Image loaded successfully ({image.shape[1]}x{image.shape[0]} pixels).")

    # --- Preprocessing ---
    processed_image = None
    if args.preprocess:
        logger.info("Preprocessing enabled...")
        # Pass relevant preprocessing arguments
        processed_image = preprocess_image(image,
                                           target_dpi=args.target_dpi,
                                           adaptive_block_size=args.adaptive_block_size,
                                           adaptive_c=args.adaptive_c)
    else:
        logger.info("Preprocessing skipped.")
        processed_image = image # Use the original image

    # --- Perform OCR ---
    logger.info(f"Starting OCR process (Lang: {args.lang}, Format: {args.format})...")
    ocr_result = perform_ocr(processed_image, lang=args.lang, output_format=args.format)

    if ocr_result is None:
        logger.error("OCR failed. Exiting.")
        sys.exit(1)
    logger.info("OCR process completed.")

    # --- Handle Output ---
    if args.output:
        logger.info(f"Saving output to: {args.output}...")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(ocr_result, f, indent=4)
                else:
                    f.write(ocr_result)
            logger.info(f"Output saved successfully to {args.output}.")
        except IOError as e:
            logger.error(f"Error writing output file {args.output}: {e}")
            # Fallback print remains
            print("\n--- OCR Result (Console Fallback) ---")
            if args.format == 'json':
                print(json.dumps(ocr_result, indent=4))
            else:
                print(ocr_result)
            print("-------------------------------------")
    else:
        # Fallback print remains
        logger.info("Outputting OCR result to console.")
        print("\n--- OCR Result ---")
        if args.format == 'json':
            print(json.dumps(ocr_result, indent=4))
        else:
            print(ocr_result)
        print("------------------")

    # --- Highlight Text (Optional) ---
    if args.highlight:
        if args.format == 'json':
            logger.info(f"Highlighting text and saving to: {args.highlight}...")
            highlighted_img = highlight_text(image, ocr_result)
            try:
                success = cv2.imwrite(args.highlight, highlighted_img)
                if success:
                    logger.info(f"Highlighted image saved successfully to {args.highlight}.")
                else:
                    logger.error(f"Could not save highlighted image to {args.highlight}.")
            except Exception as e:
                 logger.error(f"Error saving highlighted image {args.highlight}: {e}")

    logger.info("OCR Tool finished.")

if __name__ == "__main__":
    main()