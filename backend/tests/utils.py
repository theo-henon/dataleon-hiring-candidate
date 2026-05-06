import json
import os

import numpy as np
from PIL import Image, ImageDraw


def compute_iou(boxA: list, boxB: list):
    """Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list): A list of 4 coordinates [x0, y0, x1, y1] representing the first bounding box.
        boxB (list): A list of 4 coordinates [x0, y0, x1, y1] representing the second bounding box.

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0

    return interArea / unionArea


def validate_tables(
    expected_tables: list, detected_scores: list, detected_boxes: list, tolerance: float = 0.5
):
    """
    Validate detected tables against expected tables using IoU (Intersection over Union).

    Args:
        expected_tables (list): List of expected bounding boxes [[x0, y0, x1, y1], ...]
        detected_scores (list): List of detected scores (not used for IoU)
        detected_boxes (list): List of detected bounding boxes [[x0, y0, x1, y1], ...]
        tolerance (float): Minimum IoU required to consider a match (0-1). Default is 0.5 for a
        "correct" detection.
    """
    assert len(detected_scores) == len(expected_tables) and len(detected_boxes) == len(
        expected_tables
    ), (
        f"Expected {len(expected_tables)} tables, but detected {len(detected_scores)} scores and"
        f" {len(detected_boxes)} boxes."
    )

    for idx, expected_bbox in enumerate(expected_tables):
        expected_bbox = np.array(expected_bbox)
        detected_bbox = np.array(detected_boxes[idx])
        iou = compute_iou(expected_bbox, detected_bbox)
        assert iou >= tolerance, (
            f"Table {idx} IoU {iou:.3f} is below tolerance {tolerance}. "
            f"Expected bbox: {expected_bbox}, Detected bbox: {detected_bbox}"
        )


def load_ground_truth(data_path: str, template_basename: str) -> list:
    """Load ground truth annotations for a given template.

    Args:
        data_path (str): The path to the directory containing the data and annotations.
        template_basename (str): The basename of the template file.

    Returns:
        list: A list of list of list of bounding boxes for tables for each page, where each bounding
        box is a list of 4 coordinates [x0, y0, x1, y1].
        Where x0, y0 is the top-left corner and x1, y1 is
        the bottom-right corner of the bounding box.
    """
    annotation_path = os.path.join(data_path, f"{template_basename}.json")
    with open(annotation_path) as f:
        annotation = json.load(f)

    pages = annotation.get("TABLE", [])
    all_bboxes = []
    for page in pages:
        bboxes = []
        # Handle two possible structures:
        # 1. page is a dict with 'tables' key (old structure)
        # 2. page is a list of dicts (new structure, as in the provided JSON)
        if isinstance(page, dict) and "tables" in page:
            tables = page.get("tables", [])
        elif isinstance(page, list):
            tables = page
        else:
            tables = []
        for table in tables:
            # Each table is a list of dicts (one per table instance) or a dict (one table instance)
            if isinstance(table, list):
                table_instances = table
            else:
                table_instances = [table]
            for t in table_instances:
                bbox = t.get("bbox")
                if bbox is not None:
                    # Flatten [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
                    if (
                        isinstance(bbox, list)
                        and len(bbox) == 2
                        and all(isinstance(pt, list) and len(pt) == 2 for pt in bbox)
                    ):
                        x0, y0 = bbox[0]
                        w, h = bbox[1]
                        flat_bbox = [x0, y0, x0 + w, y0 + h]
                        bboxes.append(flat_bbox)
                    else:
                        bboxes.append(bbox)
        all_bboxes.append(bboxes)
    return all_bboxes


def save_annotated_image(
    data_path: str, template_name: str, annotated_image: Image, expected_tables: list
):
    """Save an annotated image with expected tables highlighted.

    Args:
        data_path (str): The path to the directory containing the data and
        annotations.
        template_name (str): The name of the template file.
        annotated_image (Image): The image with annotations.
        expected_tables (list): A list of bounding boxes for the expected tables.
    """
    output_dir = os.path.join(data_path, "annotated")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{template_name}_annotated.jpg")

    # Anotate expected tables on the image for visual comparison
    draw = ImageDraw.Draw(annotated_image)
    for bbox in expected_tables:
        x0, y0, x1, y1 = bbox
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        draw.rectangle([x0, y0, x1, y1], outline="green", width=2)

    annotated_image.save(output_path)
    print(f"Annotated image saved to {output_path}")
