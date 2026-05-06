# Table Detector Backend

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

4. **Build the documentation:**
	 - Documentation is generated using [pdoc](https://pdoc.dev/):
		 ```sh
		 uv run pdoc -o docs src/
		 ```

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
