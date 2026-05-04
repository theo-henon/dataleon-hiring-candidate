import rich.traceback
import torch
from PIL import Image, ImageDraw
from transformers import BatchFeature, DetrForObjectDetection, DetrImageProcessor

rich.traceback.install()


class TableDetector:
    processor: DetrImageProcessor
    model: DetrForObjectDetection
    threshold: float = 0.75
    annotate: bool = False

    def __init__(
        self,
        pretained_model: str = "TahaDouaji/detr-doc-table-detection",
        threshold: float = 0.75,
        annotate: bool = False,
    ) -> None:
        self.processor = DetrImageProcessor.from_pretrained(pretained_model)
        self.model = DetrForObjectDetection.from_pretrained(pretained_model)
        self.threshold = threshold
        self.annotate = annotate

    def __preprocess_image(self, image: Image) -> BatchFeature:
        return self.processor(images=image, return_tensors="pt")

    def __postprocess_outputs(self, outputs, image: Image):
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > self.threshold
        target_sizes = torch.tensor([image.size[::-1]])
        return self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

    def detect_tables(self, image: Image):
        inputs = self.__preprocess_image(image)
        outputs = self.model(**inputs)
        results = self.__postprocess_outputs(outputs, image)

        if self.annotate:
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"], strict=False
            ):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=2)
                draw.text(
                    (box[0], box[1] - 10),
                    f"{self.model.config.id2label[label.item()]}: {round(score.item(), 3)}",
                    fill="red",
                )

        return results["scores"], results["labels"], results["boxes"], image
