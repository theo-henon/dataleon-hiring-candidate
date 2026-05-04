import rich.traceback
import torch
from PIL import Image, ImageDraw
from transformers import BatchFeature, DetrForObjectDetection, DetrImageProcessor

rich.traceback.install()


class DetrTableDetector:
    processor: DetrImageProcessor
    model: DetrForObjectDetection
    threshold: float = 0.75
    annotate: bool = False

    def __init__(
        self,
        pretrained_model: str = "TahaDouaji/detr-doc-table-detection",
        threshold: float = 0.75,
        annotate: bool = False,
    ) -> None:
        self.processor = DetrImageProcessor.from_pretrained(pretrained_model)
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model)
        self.threshold = threshold
        self.annotate = annotate

    def __preprocess_image(self, image: Image) -> BatchFeature:
        return self.processor(images=image, return_tensors="pt")

    def __postprocess_outputs(self, outputs, image: Image) -> list:
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > self.threshold
        target_sizes = torch.tensor([image.size[::-1]])
        return self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

    def __annotate_image(
        self, image: Image, scores: torch.Tensor, labels: torch.Tensor, boxes: torch.Tensor
    ) -> Image:
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(scores, labels, boxes, strict=False):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text(
                (box[0], box[1] - 10),
                f"{self.model.config.id2label[label.item()]}: {round(score.item(), 3)}",
                fill="red",
            )
        return image

    def detect_tables(self, image: Image):
        inputs = self.__preprocess_image(image)
        outputs = self.model(**inputs)
        results = self.__postprocess_outputs(outputs, image)

        if self.annotate:
            image = self.__annotate_image(
                image, results["scores"], results["labels"], results["boxes"]
            )

        return results["scores"], results["labels"], results["boxes"], image
