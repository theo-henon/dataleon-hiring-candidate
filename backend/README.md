# Table Detector Backend

## General requirements

**PDF Support Notice:**
This project uses the `pdf2image` Python package to process PDF files. `pdf2image` requires the [Poppler](https://poppler.freedesktop.org/) library to be installed on your system (providing tools like `pdftocairo` and `pdfinfo`).

**To install Poppler on Linux (Debian/Ubuntu):**
```sh
sudo apt-get update && sudo apt-get install -y poppler-utils
```
For other platforms, see the [Poppler installation instructions](https://github.com/Belval/pdf2image#installing-poppler).

## For Developers

This repository uses [uv](https://github.com/astral-sh/uv) for dependency management and Python environment handling. Please ensure you have `uv` installed before proceeding.

### Development Requirements

1. **Install uv:**
	- Follow the instructions at [uv installation guide](https://github.com/astral-sh/uv#installation).

2. **Install development dependencies:**
	 - Run:
		 ```sh
		 uv sync --dev
		 ```

3. **Run tests:**
	 - From the `backend/` directory:
		 ```sh
		 uv run pytest
		 ```

	 **Note:** Some tests are currently failing. See below for details.
	 
4. **Generate a coverage report:**
	 - Run:
		 ```sh
		 uv run pytest --cov=src --cov-report=html
		 ```
	 - The coverage report will be generated in the `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view detailed coverage information.

4. **Build the documentation:**
	 - Documentation is generated using [pdoc](https://pdoc.dev/):
		 ```sh
		 uv run pdoc -o docs src/
		 ```

---

## Test Status & Known Issues


**Latest test run summary:**

- In several cases, the number of detected tables does not match the expected number (either too many, too few, or none detected). This is common for:
  - Thick-edged tables (extra boxes detected)
  - Multiple tables with headers (extra boxes detected)
  - Small/borderless/with-header templates (no tables detected)

- For some borderless, centered, or multipage tables, the detected bounding boxes do not sufficiently overlap (IoU below 0.5) with the expected ones, causing failures even when a table is detected.

- Rotated and compressed tables are not robustly detected:
  - Rotated tables: detected box is far from the expected region (very low IoU)
  - Compressed tables: extra boxes detected

- Multipage table detection sometimes produces boxes in the wrong region or with poor overlap.

**Summary:**
- The model struggles with borderless, rotated, compressed, and multipage tables, as well as with templates that differ from the main training distribution.
- Assertion errors are mostly due to mismatched counts or low overlap (IoU) between detected and expected boxes.

---

## For Users

This project requires [uv](https://github.com/astral-sh/uv) for installing dependencies. Please install `uv` if you haven't already.

### Installation

1. **Install user requirements:**
	 - From the `backend/` directory, run:
		 ```sh
		 uv sync
		 ```

### CLI Usage

The main entry point is `main.py`, which provides a command-line interface for table detection in images or PDFs.

#### Usage

```sh
uv run python main.py <doc_path> <output_dir> [--threshold FLOAT] [--annotate] [--pretrained_model MODEL_NAME]
```

- `<doc_path>`: Path to the input document (image or PDF)
- `<output_dir>`: Path to the output directory
- `--threshold`: Confidence threshold for detections (default: 0.75)
- `--annotate`: Whether to annotate the image with detections (default: True)
- `--pretrained_model`: Pretrained model to use for table detection (default: TahaDouaji/detr-doc-table-detection)

#### Example

```sh
uv run python main.py tests/data/layout/TableBank_example.pdf output/ --threshold 0.8 --annotate
```

The annotated images (if `--annotate` is set) will be saved in the specified output directory.
