# OCR Tool

A command-line tool to extract text from images using Tesseract OCR, with image preprocessing capabilities.

## Features

*   Extracts text from various image formats (PNG, JPG, TIFF, etc.).
*   Supports multiple languages via Tesseract language packs.
*   Outputs extracted text as plain text or structured JSON (including bounding box data).
*   Optional image preprocessing pipeline:
    *   Grayscale conversion
    *   Gaussian blur (noise reduction)
    *   Otsu's binarization (for skew detection)
    *   Deskewing (corrects text angle)
    *   Adaptive Gaussian thresholding (final binarization)
*   Optional highlighting of detected text regions on the original image (requires JSON output).
*   Configuration via `ocr_config.ini` file (see below).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ocr_tool
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` currently includes `pytesseract`, `opencv-python`, `numpy`)*

3.  **Install Tesseract OCR:**
    Tesseract is required by `pytesseract`. Follow the installation instructions for your OS:
    *   **Windows:** Download from [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add Tesseract to your system's PATH during installation or set the `command_path` in `ocr_config.ini`.
    *   **macOS:** `brew install tesseract tesseract-lang`
    *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-all`
    *   **Linux (Fedora):** `sudo dnf install tesseract tesseract-devel tesseract-langpack-eng`

    **Important:** Install the necessary language data packs for the languages you intend to use.

## Usage

```bash
python main.py <image_path> [options]
```

**Configuration File (`ocr_config.ini`):**

The tool will look for a file named `ocr_config.ini` in the directory where you run the `main.py` script. If found, it will read default values from it. Command-line arguments always override settings from the configuration file.

Example `ocr_config.ini`:

```ini
[Defaults]
# Default language(s) to use if -l/--lang is not specified
language = eng

[Tesseract]
# Optional: Full path to the Tesseract executable if it's not in your system PATH
command_path =

[Preprocessing]
# Defaults for adaptive thresholding parameters (used if --preprocess is enabled
# and the corresponding command-line args are not provided)
adaptive_block_size = 11
adaptive_c = 5
```

**Arguments:**

*   `image_path`: Path to the input image file (required).

**Options:**

*General:* 
*   `-o, --output OUTPUT_PATH`: Path to save the output text or JSON file. Prints to console if omitted.
*   `-v, --verbose`: Enable verbose (DEBUG level) logging.

*OCR Options:* 
*   `-l, --lang LANG`: Language code(s) for Tesseract (e.g., `eng`, `fra`). Overrides config. Default set by config or 'eng'.
*   `-f, --format FORMAT`: Output format (`text` or `json`). Default: `text`.

*Preprocessing Options:* (These require `--preprocess` to be set)
*   `--preprocess`: Enable the full image preprocessing pipeline.
*   `--target-dpi DPI`: Target DPI for image resizing.
*   `--adaptive-block-size INT`: Block size for adaptive thresholding (must be odd, >1). Overrides config. Default set by config or 11.
*   `--adaptive-c INT`: Constant C subtracted during adaptive thresholding. Overrides config. Default set by config or 5.

*Highlighting Options:* (Requires `--format json`)
*   `--highlight HIGHLIGHT_PATH`: Path to save the image with detected text boxes highlighted.

**Examples:**

1.  **Basic OCR (using defaults from config file):**
    ```bash
    python main.py images/sample.png
    ```

2.  **OCR with preprocessing, override language and adaptive C:**
    ```bash
    python main.py docs/invoice.jpg --preprocess -l deu --adaptive-c 7 -o output.txt
    ```

3.  **OCR using Tesseract path from config and specific block size:**
    ```bash
    python main.py screenshots/error.png --preprocess --adaptive-block-size 15 --format json -o results.json --highlight debug_highlight.png
    ```

## Project Structure

```
ocr_tool/
├── main.py           # Main script, CLI handler, Config loading
├── ocr_processor.py  # Core OCR logic using pytesseract
├── image_utils.py    # Image preprocessing and highlighting functions
├── requirements.txt  # Python dependencies
├── ocr_config.ini    # Optional configuration file (Example)
├── README.md         # This file
├── PLANNING.md       # Project planning details
└── TASK.md           # Task tracking
``` 