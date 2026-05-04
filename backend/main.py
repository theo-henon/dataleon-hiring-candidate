import os
import sys

import rich.traceback
import torch
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor

rich.traceback.install()


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python main.py <IMAGE_PATH>")
        sys.exit(1)

    image_path = argv[1]
    image = Image.open(image_path)

    processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
    model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.75
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.75
    )[0]

    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]

    for score, label, box in zip(scores, labels, boxes, strict=False):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        # Draw the bounding box on the image with the confidence score
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=2)
        draw.text(
            (box[0], box[1] - 10),
            f"{round(score.item(), 3)}",
            fill="red",
        )

    # Save the image with detections
    output_path = os.path.join(
        "..", "data", "output", f"{os.path.basename(image_path)}-annotated.png"
    )
    image.save(output_path)
    print(f"Output image saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv)
