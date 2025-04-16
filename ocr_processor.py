import pytesseract
from PIL import Image
import json
import cv2 # Needed for color conversion if input is BGR
import logging
import sys

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Tesseract Configuration ---
# This function allows setting the path from main.py based on config
def set_tesseract_path(path):
    """Sets the command path for the pytesseract library."""
    if path:
        try:
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Pytesseract command path set to: {path}")
        except Exception as e:
            logger.error(f"Failed to set Tesseract command path to '{path}': {e}")
    else:
        logger.debug("No Tesseract command path provided, using system PATH.")

# Note: The old manual path setting is replaced by the function above.
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image, lang='eng', output_format='text'):
    """
    Performs OCR on the preprocessed image using Tesseract.

    Args:
        image: The preprocessed image (NumPy array from OpenCV - assumed BGR or Grayscale).
        lang (str): The language code(s) for Tesseract (e.g., 'eng', 'eng+fra').
        output_format (str): The desired output format ('text' or 'json').

    Returns:
        str or dict: The extracted text or structured data (JSON compatible dict).
        Returns None if OCR fails.
    """
    logger.info(f"Performing OCR with language: {lang}, format: {output_format}...")

    try:
        # Tesseract works best with RGB images, OpenCV loads as BGR by default
        logger.debug("Preparing image for Tesseract...")
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            logger.debug("Converting BGR image to RGB.")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            # Assume grayscale or already correct format
            logger.debug("Image is single-channel, using as is.")
            pil_image = Image.fromarray(image)

        # Use pytesseract to extract data
        # Common config: OEM 3 (LSTM engine), PSM 3 (Auto page segmentation) or 6 (Assume single uniform block)
        # Adjust PSM based on expected document structure
        custom_config = r'--oem 3 --psm 3'
        logger.debug(f"Using Tesseract config: '{custom_config}'")

        if output_format == 'text':
            logger.debug("Calling pytesseract.image_to_string...")
            text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
            logger.debug(f"Extracted text length: {len(text)}")
            return text
        elif output_format == 'json':
            logger.debug("Calling pytesseract.image_to_data...")
            # Get detailed data including bounding boxes, confidence, etc.
            # Output format is a dictionary
            data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT, config=custom_config)
            logger.debug(f"Extracted data keys: {list(data.keys())}")
            logger.debug(f"Number of detected boxes/levels: {len(data.get('level', []))}")
            # You might want to structure this further, e.g., filter low confidence words
            return data
        else:
            # This case should ideally not be reached due to argparse choices
            logger.error(f"Unsupported output format '{output_format}'")
            return None

    except pytesseract.TesseractNotFoundError:
         # Log the error and provide helpful instructions
         error_message = (
             "Tesseract Error: Tesseract is not installed or not in your PATH. "
             "Check your system PATH or set 'command_path' in ocr_config.ini.\n"
             "Installation guides:\n"
             "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
             "  macOS: brew install tesseract tesseract-lang\n"
             "  Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr tesseract-ocr-all\n"
             "  Linux (Fedora): sudo dnf install tesseract tesseract-langpack-eng"
         )
         logger.error(error_message)
         # Also print to stderr for immediate visibility even if logging is redirected
         print(f"\n{error_message}\n", file=sys.stderr)
         return None
    except Exception as e:
        logger.error(f"Error during Tesseract OCR processing: {e}", exc_info=True) # Include traceback
        # Consider checking if the language data is available
        if "Failed loading language" in str(e):
             lang_error_message = f"Error: Language data for '{lang}' might be missing. Please install the required Tesseract language packs (e.g., on Debian/Ubuntu: sudo apt-get install tesseract-ocr-{lang})."
             logger.error(lang_error_message)
             print(f"\n{lang_error_message}\n", file=sys.stderr)
        return None 