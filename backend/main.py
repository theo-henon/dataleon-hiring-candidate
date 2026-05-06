import os

import rich.traceback

from src.cli import parse_cli
from src.table_detector import DetrTableDetector

rich.traceback.install()


def main() -> None:
    args = parse_cli()

    table_detector = DetrTableDetector(threshold=args.threshold, annotate=args.annotate)
    results = table_detector.detect_tables(args.doc_path)

    is_pdf = os.path.splitext(args.doc_path)[1].lower() == ".pdf"
    basename = os.path.basename(args.doc_path).split(".")[0]

    for idx, (scores, labels, boxes, annotated_image) in enumerate(results):
        page_info = f" (page {idx + 1})" if is_pdf else ""
        for score, label, box in zip(scores, labels, boxes, strict=False):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {table_detector.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}{page_info}"
            )

        # Save the image with detections
        if table_detector.annotate:
            if is_pdf:
                output_path = os.path.join(
                    args.output_dir, f"{basename}-page{idx + 1}-annotated.png"
                )
            else:
                output_path = os.path.join(args.output_dir, f"{basename}-annotated.png")
            annotated_image.save(output_path)
            print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    main()
