import cv2
import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_TARGET_DPI = 300
# Assume a common screen DPI if no metadata is available. This is a heuristic.
DEFAULT_SOURCE_DPI = 96
# Default Parameters for adaptive thresholding (used if not passed)
DEFAULT_ADAPTIVE_BLOCK_SIZE = 11 # Must be odd
DEFAULT_ADAPTIVE_C = 5 # Constant subtracted from the mean/weighted mean

def resize_to_dpi(image, target_dpi=DEFAULT_TARGET_DPI, source_dpi=DEFAULT_SOURCE_DPI):
    """Resizes the image to match a target DPI."""
    h, w = image.shape[:2]
    scale_factor = target_dpi / source_dpi

    if abs(scale_factor - 1.0) < 0.01: # Don't resize if already close
        logger.info(f"Skipping resize: Image already near target DPI ({target_dpi}).")
        return image

    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    logger.info(f"Resizing image from ({w}x{h}) to ({new_w}x{new_h}) for target DPI {target_dpi} (source DPI assumed {source_dpi}).")

    # Choose interpolation based on whether we are scaling up or down
    if scale_factor > 1.0:
        interpolation = cv2.INTER_CUBIC # Better for enlarging
    else:
        interpolation = cv2.INTER_AREA # Better for shrinking

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized_image

def compute_skew_angle(binary_image):
    """Computes the skew angle of the text in a binarized image."""
    logger.debug("Computing skew angle...")
    # Invert the image (text should be white on black background for findContours)
    # Ensure the image is 8-bit single channel
    if len(binary_image.shape) > 2:
         # If it's somehow not grayscale, convert it
         logger.debug("Input for skew calculation is not grayscale, converting...")
         gray_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
         # Use THRESH_BINARY_INV directly here
         _, binary_image_inv = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif binary_image.max() == 255: # Check if text is black (on white background)
         logger.debug("Input for skew calculation is black text on white bg, inverting...")
         # Invert black text on white background
         binary_image_inv = cv2.bitwise_not(binary_image)
    else: # Assume text is already white on black
         logger.debug("Input for skew calculation is white text on black bg.")
         binary_image_inv = binary_image

    # Find contours on the inverted image
    contours, _ = cv2.findContours(binary_image_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found for skew calculation.")
        return 0.0 # No skew detected or calculable

    # Combine all contours into a single point set
    # Filter small contours that might be noise
    min_contour_area = 10 # Adjust as needed
    all_points = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            all_points.append(contour)

    if not all_points:
        logger.warning("No sufficiently large contours found for skew calculation.")
        return 0.0

    points = np.vstack(all_points).squeeze()
    # Check points shape after squeeze
    if points.ndim == 0: # Single point detected, unlikely for minAreaRect
         logger.warning("Only a single point contour found, cannot calculate skew.")
         return 0.0
    elif points.shape[0] < 5: # minAreaRect requires at least 5 points
        logger.warning(f"Not enough contour points ({points.shape[0]}) found for accurate skew calculation.")
        return 0.0

    # Find the minimum area rectangle enclosing the points
    # rect = ((center_x, center_y), (width, height), angle_of_rotation)
    try:
        rect = cv2.minAreaRect(points)
    except Exception as e:
         logger.warning(f"cv2.minAreaRect failed: {e}. Skipping skew calculation.")
         return 0.0

    angle = rect[-1]
    logger.debug(f"Initial angle from minAreaRect: {angle:.2f}")

    # The angle returned by minAreaRect is in [-90, 0).
    # Adjust it based on the rectangle's aspect ratio to be in [-45, 45].
    (width, height) = rect[1]
    if width < height: # Portrait orientation box
        logger.debug("Adjusting angle for portrait rectangle.")
        angle = angle + 90
    # else: Landscape orientation box, angle is likely correct

    # Ensure angle is in the [-45, 45] range
    if angle > 45:
        logger.debug("Angle > 45, adjusting...")
        angle = angle - 90
    elif angle < -45:
         logger.debug("Angle < -45, adjusting...")
         angle = angle + 90

    # Filter out near-zero angles after correction
    original_angle = angle
    if abs(angle) < 0.1:
        angle = 0.0
        if abs(original_angle) >= 0.1:
             logger.debug("Adjusted near-zero angle to 0.0")

    logger.info(f"Computed skew angle: {angle:.2f} degrees")
    return angle

def deskew(image, angle):
    """Rotates the image to correct the computed skew angle."""
    if abs(angle) < 0.1: # Don't rotate for tiny angles
         logger.info("Skipping rotation for negligible skew angle.")
         return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    logger.debug(f"Rotation matrix calculated for angle {angle:.2f}")

    # Determine background color based on image type (assume white)
    background_color = 255

    rotated = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_CUBIC, # Smoother interpolation for rotation
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=background_color)

    logger.info(f"Image rotated by {-angle:.2f} degrees to correct skew.")
    return rotated

def preprocess_image(image, target_dpi=None,
                     adaptive_block_size=DEFAULT_ADAPTIVE_BLOCK_SIZE,
                     adaptive_c=DEFAULT_ADAPTIVE_C):
    """
    Applies various preprocessing steps to the image to improve OCR accuracy.
    Includes optional Resizing -> Grayscale -> Blur -> Otsu Binarization (for skew) -> Deskewing -> Adaptive Binarization.

    Args:
        image: The input image (NumPy array, assumed BGR from cv2.imread).
        target_dpi (int, optional): If provided, resize the image to this target DPI.
        adaptive_block_size (int): Block size for adaptive thresholding (must be odd).
        adaptive_c (int): Constant C subtracted for adaptive thresholding.

    Returns:
        The preprocessed image (NumPy array, grayscale, binarized, and possibly resized/deskewed).
    """
    logger.info("Starting image preprocessing pipeline...")
    processed_image = image.copy()

    # 0. Optional Resizing (before other steps)
    if target_dpi is not None:
        logger.debug(f"Applying resizing to target DPI: {target_dpi}")
        processed_image = resize_to_dpi(processed_image, target_dpi=target_dpi)

    # 1. Convert to Grayscale
    logger.debug("Applying grayscale conversion...")
    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        logger.debug("Image is already single-channel.")
        gray = processed_image # Assume it's already grayscale or was resized to grayscale

    # 2. Noise Reduction
    logger.debug("Applying Gaussian blur...")
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Initial Binarization (Global Otsu for skew calculation)
    logger.debug("Applying initial global binarization (Otsu for skew calculation)...")
    try:
        _, binary_image_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        logger.debug("Initial global binarization successful (Otsu).")
    except Exception as e:
        logger.warning(f"Otsu's thresholding failed ({e}) for skew calc. Falling back to simple binary threshold.")
        _, binary_image_otsu = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)

    # 4. Deskewing (based on global threshold image)
    logger.debug("Applying deskewing...")
    skew_angle = compute_skew_angle(binary_image_otsu)
    # Apply deskewing to the denoised grayscale image for better quality
    deskewed_image = deskew(denoised, skew_angle)

    # Validate adaptive block size (must be odd and > 1)
    if adaptive_block_size <= 1 or adaptive_block_size % 2 == 0:
        logger.warning(f"Invalid adaptive block size ({adaptive_block_size}). It must be odd and > 1. Using default: {DEFAULT_ADAPTIVE_BLOCK_SIZE}")
        adaptive_block_size = DEFAULT_ADAPTIVE_BLOCK_SIZE

    # 5. Final Binarization (Adaptive)
    logger.debug(f"Applying final adaptive binarization (BlockSize={adaptive_block_size}, C={adaptive_c})...")
    try:
        # Adaptive thresholding works best on grayscale images
        final_processed_image = cv2.adaptiveThreshold(deskewed_image,
                                                   maxValue=255,
                                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   thresholdType=cv2.THRESH_BINARY,
                                                   blockSize=adaptive_block_size,
                                                   C=adaptive_c)
        logger.debug("Final adaptive binarization successful.")
    except Exception as e:
        logger.error(f"Adaptive thresholding failed: {e}. Falling back to global Otsu on deskewed image.", exc_info=True)
        # Fallback to Otsu if adaptive fails (less likely but possible)
        try:
            _, final_processed_image = cv2.threshold(deskewed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        except Exception as e_otsu:
             logger.error(f"Fallback Otsu thresholding also failed: {e_otsu}. Falling back to simple binary.", exc_info=True)
             _, final_processed_image = cv2.threshold(deskewed_image, 127, 255, cv2.THRESH_BINARY)

    logger.info("Image preprocessing pipeline complete.")
    return final_processed_image

def highlight_text(image, ocr_data, confidence_threshold=60):
    """
    Draws bounding boxes around detected text on the original image.

    Args:
        image: The original image (NumPy array, BGR).
        ocr_data (dict): The structured OCR data from pytesseract (image_to_data DICT format).
        confidence_threshold (int): Minimum confidence level (0-100) to draw a box.

    Returns:
        The image with highlighted text (NumPy array, BGR).
    """
    logger.info(f"Highlighting text with confidence >= {confidence_threshold}...")
    highlighted_image = image.copy() # Work on a copy
    n_boxes = len(ocr_data.get('level', [])) # Use .get for safety

    if n_boxes == 0:
        logger.warning("No text boxes found in OCR data to highlight.")
        return highlighted_image

    highlight_count = 0
    for i in range(n_boxes):
        # Check if the confidence level is valid and meets the threshold
        try:
            # Ensure conf value exists and is convertible to float
            conf_val = ocr_data['conf'][i]
            if conf_val == '': continue # Skip empty confidence values
            conf = int(float(conf_val))

            if conf >= confidence_threshold:
                # Extract bounding box coordinates
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])

                # Ensure coordinates are valid (non-negative width/height)
                if w > 0 and h > 0:
                    logger.debug(f"Highlighting box: x={x}, y={y}, w={w}, h={h}, conf={conf}")
                    # Draw rectangle: image, top-left corner, bottom-right corner, color (BGR), thickness
                    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
                    highlight_count += 1
        except (ValueError, KeyError, IndexError) as e:
            # Handle cases where confidence might not be a number, key is missing, or index is out of bounds
            logger.debug(f"Skipping box {i} due to data issue: {e} (Value: {ocr_data.get('conf', [])[i] if i < len(ocr_data.get('conf', [])) else 'N/A'}) ")
            continue # Skip this box

    logger.info(f"Highlighted {highlight_count} text boxes.")
    return highlighted_image

# --- TODO: Implement helper functions like deskew, resize if needed --- # Kept for clarity 