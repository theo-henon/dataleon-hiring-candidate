import os
import sys

import rich.traceback
from PIL import Image

from src.table_detector import DetrTableDetector

rich.traceback.install()


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python main.py <IMAGE_PATH>")
        sys.exit(1)

    image_path = argv[1]
    image = Image.open(image_path)

    table_detector = DetrTableDetector(threshold=0.75, annotate=False)
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
            "..", "data", "output", f"{os.path.basename(image_path).split('.')[0]}-annotated.png"
        )
        annotated_image.save(output_path)
        print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv)
