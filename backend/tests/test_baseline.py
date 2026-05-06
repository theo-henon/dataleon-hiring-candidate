import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_ground_truth, save_annotated_image, validate_tables

from table_detector import DetrTableDetector


@pytest.fixture
def table_detector() -> DetrTableDetector:
    return DetrTableDetector(threshold=0.8, annotate=True)


@pytest.fixture
def baseline_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "baseline")


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template1_Instance0_thick-edges-big"),
        ("medium", "FATURA_Template1_Instance4_thick-edges-medium"),
        ("small", "FATURA_Template1_Instance2_thick-edges-small"),
    ],
)
def test_thick_edges(table_detector, baseline_data_path, size, template_name):
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(baseline_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(baseline_data_path, template_name, annotated_img, expected_tables[0])
    validate_tables(expected_tables[0], detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template2_Instance2_clear-edges-big"),
        ("medium", "FATURA_Template2_Instance0_clear-edges-medium"),
        ("small", "FATURA_Template2_Instance12_clear-edges-small"),
    ],
)
def test_clear_edges(table_detector, baseline_data_path, size, template_name):
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(baseline_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(baseline_data_path, template_name, annotated_img, expected_tables[0])
    validate_tables(expected_tables[0], detected_scores.tolist(), detected_boxes.tolist())


def test_multiple_with_header(table_detector, baseline_data_path):
    template_name = "TableBank_%5BMS-OAPXBC%5D-170613_35_multiple-with-header"
    template_img_path = os.path.join(baseline_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(baseline_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(baseline_data_path, template_name, annotated_img, expected_tables[0])
    validate_tables(expected_tables[0], detected_scores.tolist(), detected_boxes.tolist())
