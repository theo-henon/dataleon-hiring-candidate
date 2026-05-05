import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from utils import load_ground_truth, save_annotated_image, validate_tables

from table_detector import DetrTableDetector


@pytest.fixture
def table_detector() -> DetrTableDetector:
    return DetrTableDetector(threshold=0.8, annotate=True)


@pytest.fixture
def layout_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "layout")


def test_titles_in_bold_big(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template6_Instance0_titles-in-bold-big"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_titles_in_bold_medium(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template6_Instance5_titles-in-bold-medium"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_titles_in_bold_small(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template6_Instance2_titles-in-bold-small"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_centered_text_big(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template7_Instance0_centered-text-big"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_centered_text_medium(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template7_Instance4_centered-text-medium"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_centered_text_small(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template7_Instance3_centered-text-small"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_titles_in_green_big(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template9_Instance13_titles-in-green-big"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_titles_in_green_medium(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template9_Instance0_titles-in-green-medium"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_titles_in_green_small(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template9_Instance14_titles-in-green-small"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_upper_table_big(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template12_Instance35_upper-table-big"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_upper_table_medium(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template12_Instance34_upper-table-medium"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_upper_table_small(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template12_Instance33_upper-table-small"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_borderless_big(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template10_Instance84_borderless-big"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_borderless_medium(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template10_Instance83_borderless-medium"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_borderless_small(table_detector: DetrTableDetector, layout_data_path: str):
    template_name = "FATURA_Template10_Instance82_borderless-small"
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)

    expected_tables = load_ground_truth(layout_data_path, template_name)

    # Check results list length (should match number of pages/images)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]

    # Save annotated image for visual inspection
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)

    # Check bounding boxes with a tolerance of 10 pixels
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())
