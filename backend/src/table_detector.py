import os

import rich.traceback
import torch
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from transformers import BatchFeature, DetrForObjectDetection, DetrImageProcessor

rich.traceback.install()


class DetrTableDetector:
    """
    Table detection using a pretrained DETR (DEtection TRansformer) model.

    This class wraps a HuggingFace DETR model fine-tuned for table detection in documents.
    It supports both image and PDF input, and can optionally annotate detected tables on the output
    image(s).

    Attributes:
        processor (DetrImageProcessor): Preprocessing and postprocessing utility for DETR.
        model (DetrForObjectDetection): Pretrained DETR model for object detection.
        threshold (float): Minimum confidence score for a detection to be kept.
        annotate (bool): Whether to draw detected tables on the output images.

    Methods:
        detect_tables(image_path: str) -> list:
            Detect tables in an image or PDF file. Returns a list of tuples (scores, labels, boxes,
            image) for each page or image.
    """

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
        """
        Initialize the DetrTableDetector.

        Args:
            pretrained_model (str): Name or path of the pretrained DETR model.
            threshold (float): Minimum confidence score for a detection to be kept.
            annotate (bool): Whether to draw detected tables on the output images.
        """
        self.processor = DetrImageProcessor.from_pretrained(pretrained_model)
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model)
        self.threshold = threshold
        self.annotate = annotate

    def __preprocess_image(self, image: Image) -> BatchFeature:
        """
        Preprocess an image for DETR model inference.

        Args:
            image (PIL.Image): The input image.
        Returns:
            BatchFeature: Preprocessed image tensor for the model.
        """
        return self.processor(images=image, return_tensors="pt")

    def __postprocess_outputs(self, outputs, image: Image) -> list:
        """
        Postprocess model outputs to extract bounding boxes, scores, and labels.

        Args:
            outputs: Raw outputs from the DETR model.
            image (PIL.Image): The original input image.
        Returns:
            dict: Dictionary with keys 'scores', 'labels', and 'boxes'.
        """
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > self.threshold
        target_sizes = torch.tensor([image.size[::-1]])
        return self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

    def __annotate_image(
        self, image: Image, scores: torch.Tensor, labels: torch.Tensor, boxes: torch.Tensor
    ) -> Image:
        """
        Draw bounding boxes and labels on the image for detected tables.

        Args:
            image (PIL.Image): The image to annotate.
            scores (torch.Tensor): Detection confidence scores.
            labels (torch.Tensor): Detected class labels.
            boxes (torch.Tensor): Detected bounding boxes.
        Returns:
            PIL.Image: Annotated image.
        """
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

    def detect_tables(self, doc_path: str) -> list:
        """
        Detect tables in an image or PDF file.

        Args:
            doc_path (str): Path to an image file or PDF file.

        Returns:
            list: For each image or PDF page, a tuple (scores, labels, boxes, image):
                - scores (torch.Tensor): Detection confidence scores.
                - labels (torch.Tensor): Detected class labels.
                - boxes (torch.Tensor): Detected bounding boxes [[x0, y0, x1, y1], ...].
                - image (PIL.Image): The (optionally annotated) image.
        """
        ext = os.path.splitext(doc_path)[1].lower()
        results = []
        if ext == ".pdf":
            # Convert PDF pages to images
            pages = convert_from_path(doc_path)
            for page in pages:
                inputs = self.__preprocess_image(page)
                outputs = self.model(**inputs)
                res = self.__postprocess_outputs(outputs, page)
                img = page
                if self.annotate:
                    img = self.__annotate_image(img, res["scores"], res["labels"], res["boxes"])
                results.append((res["scores"], res["labels"], res["boxes"], img))
        else:
            image = Image.open(doc_path).convert("RGB")
            inputs = self.__preprocess_image(image)
            outputs = self.model(**inputs)
            res = self.__postprocess_outputs(outputs, image)
            img = image
            if self.annotate:
                img = self.__annotate_image(img, res["scores"], res["labels"], res["boxes"])
            results.append((res["scores"], res["labels"], res["boxes"], img))
        return results
