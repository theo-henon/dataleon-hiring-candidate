import json
import os

import numpy as np
from PIL import Image, ImageDraw


def validate_tables(
    expected_tables: list, detected_scores: list, detected_boxes: list, tolerance: int = 30
):
    # Check number of detected tables
    assert len(detected_scores) == len(expected_tables) and len(detected_boxes) == len(
        expected_tables
    ), (
        f"Expected {len(expected_tables)} tables, but detected {len(detected_scores)} scores and"
        f" {len(detected_boxes)} boxes."
    )

    # Check bounding boxes with a tolerance
    for idx, expected_bbox in enumerate(expected_tables):
        expected_bbox = np.array(expected_bbox)
        detected_bbox = np.array(detected_boxes[idx])
        assert np.all(np.abs(expected_bbox - detected_bbox) <= tolerance), (
            f"Table {idx} bounding box {detected_bbox} does not match expected {expected_bbox}"
            f" within tolerance {tolerance}."
        )


def load_ground_truth(baseline_data_path: str, template_basename: str) -> list:
    """Load ground truth annotations for a given template.

    Args:
        baseline_data_path (str): The path to the directory containing the baseline data and
        annotations.
        template_basename (str): The basename of the template file.

    Returns:
        list: A list of list of list of bounding boxes for tables for each page, where each bounding
        box is a list of 4 coordinates [x0, y0, x1, y1].
        Where x0, y0 is the top-left corner and x1, y1 is
        the bottom-right corner of the bounding box.
    """
    annotation_path = os.path.join(baseline_data_path, f"{template_basename}.json")
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
    baseline_data_path: str, template_name: str, annotated_image: Image, expected_tables: list
):
    output_dir = os.path.join(baseline_data_path, "annotated")
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
