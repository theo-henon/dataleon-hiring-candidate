import argparse
import os

import rich.traceback
from PIL import Image

from src.table_detector import DetrTableDetector

rich.traceback.install()


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect tables in an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Whether to annotate the image with detections",
        default=True,
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="TahaDouaji/detr-doc-table-detection",
        help="Pretrained model to use for table detection. Default is ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()

    image = Image.open(args.image_path)

    table_detector = DetrTableDetector(threshold=args.threshold, annotate=args.annotate)
    scores, labels, boxes, annotated_image = table_detector.detect_tables(image)

    for score, label, box in zip(scores, labels, boxes, strict=False):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {table_detector.model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    # Save the image with detections
    if table_detector.annotate:
        output_path = os.path.join(
            args.output_dir, f"{os.path.basename(args.image_path).split('.')[0]}-annotated.png"
        )
        annotated_image.save(output_path)
        print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    main()
