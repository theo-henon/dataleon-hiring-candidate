import argparse

import rich.traceback

rich.traceback.install()


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect tables in a document (image or PDF)")
    parser.add_argument("doc_path", help="Path to the input document (image or PDF)")
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Whether to annotate the image with detections",
        default=False,
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="TahaDouaji/detr-doc-table-detection",
        help="Pretrained model to use for table detection. Default is ",
    )
    return parser.parse_args()
