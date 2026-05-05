import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from table_detector import DetrTableDetector


@pytest.fixture
def baseline_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "baseline")


@pytest.fixture
def table_detector() -> DetrTableDetector:
    return DetrTableDetector(threshold=0.75, annotate=False)


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


def test_thick_edges_big(table_detector: DetrTableDetector, baseline_data_path: str):
    template_name = "FATURA_Template1_Instance0_thick-edges-big"
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    template_annotation_path = os.path.join(baseline_data_path, f"{template_name}.json")

    results_list = table_detector.detect_tables(template_img_path)
    annotation = json.load(open(template_annotation_path))
    expected_tables = annotation["TABLE"]

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, _ = results_list[0]

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores, detected_boxes, tolerance=10)
