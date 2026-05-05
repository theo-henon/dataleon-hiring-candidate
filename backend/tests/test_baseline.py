import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from table_detector import DetrTableDetector


@pytest.fixture
def baseline_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "baseline")


@pytest.fixture
def table_detector() -> DetrTableDetector:
    return DetrTableDetector(threshold=0.8, annotate=True)


def validate_tables(expected_tables, detected_scores, detected_boxes, tolerance=10):
    # Check number of detected tables
    assert len(detected_scores) == len(expected_tables) and len(detected_boxes) == len(
        expected_tables
    ), (
        f"Expected {len(expected_tables)} tables, but detected {len(detected_scores)} scores and"
        f" {len(detected_boxes)} boxes."
    )

    # Check bounding boxes with a tolerance
    for i, expected_table in enumerate(expected_tables):
        if (
            isinstance(expected_table, list)
            and len(expected_table) == 1
            and isinstance(expected_table[0], dict)
        ):
            expected_table = expected_table[0]
        expected_box = expected_table["bbox"]

        if isinstance(expected_box[0], list) and len(expected_box) == 2:
            # Flatten [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
            expected_box = np.array(expected_box).flatten().tolist()

        # Convert tensor to list
        detected_box = detected_boxes[i].tolist()
        assert all(
            abs(e - d) <= tolerance for e, d in zip(expected_box, detected_box, strict=False)
        ), f"Table {i}: Expected box {expected_box}, but detected {detected_box}."


def save_annotated_image(baseline_data_path: str, template_name: str, annotated_image: Image):
    output_dir = os.path.join(baseline_data_path, "annotated")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{template_name}_annotated.jpg")
    annotated_image.save(output_path)
    print(f"Annotated image saved to {output_path}")


def test_thick_edges_big(table_detector: DetrTableDetector, baseline_data_path: str):
    template_name = "FATURA_Template1_Instance0_thick-edges-big"
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    template_annotation_path = os.path.join(baseline_data_path, f"{template_name}.json")

    results_list = table_detector.detect_tables(template_img_path)
    annotation = json.load(open(template_annotation_path))
    expected_tables = annotation["TABLE"]

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(baseline_data_path, template_name, annotated_img)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores, detected_boxes, tolerance=10)


def test_thick_edges_medium(table_detector: DetrTableDetector, baseline_data_path: str):
    template_name = "FATURA_Template1_Instance4_thick-edges-medium"
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    template_annotation_path = os.path.join(baseline_data_path, f"{template_name}.json")

    results_list = table_detector.detect_tables(template_img_path)
    annotation = json.load(open(template_annotation_path))
    expected_tables = annotation["TABLE"]

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(baseline_data_path, template_name, annotated_img)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores, detected_boxes, tolerance=10)


def test_thick_edges_small(table_detector: DetrTableDetector, baseline_data_path: str):
    template_name = "FATURA_Template1_Instance2_thick-edges-small"
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    template_annotation_path = os.path.join(baseline_data_path, f"{template_name}.json")

    results_list = table_detector.detect_tables(template_img_path)
    annotation = json.load(open(template_annotation_path))
    expected_tables = annotation["TABLE"]

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(baseline_data_path, template_name, annotated_img)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores, detected_boxes, tolerance=10)
