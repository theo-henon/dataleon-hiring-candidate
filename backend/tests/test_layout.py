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


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template6_Instance0_titles-in-bold-big"),
        ("medium", "FATURA_Template6_Instance5_titles-in-bold-medium"),
        ("small", "FATURA_Template6_Instance2_titles-in-bold-small"),
    ],
)
def test_titles_in_bold(table_detector, layout_data_path, size, template_name):
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(layout_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template7_Instance0_centered-text-big"),
        ("medium", "FATURA_Template7_Instance4_centered-text-medium"),
        ("small", "FATURA_Template7_Instance3_centered-text-small"),
    ],
)
def test_centered_text(table_detector, layout_data_path, size, template_name):
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(layout_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template9_Instance13_titles-in-green-big"),
        ("medium", "FATURA_Template9_Instance0_titles-in-green-medium"),
        ("small", "FATURA_Template9_Instance14_titles-in-green-small"),
    ],
)
def test_titles_in_green(table_detector, layout_data_path, size, template_name):
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(layout_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template12_Instance35_upper-table-big"),
        ("medium", "FATURA_Template12_Instance34_upper-table-medium"),
        ("small", "FATURA_Template12_Instance33_upper-table-small"),
    ],
)
def test_upper_table(table_detector, layout_data_path, size, template_name):
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(layout_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "size,template_name",
    [
        ("big", "FATURA_Template10_Instance84_borderless-big"),
        ("medium", "FATURA_Template10_Instance83_borderless-medium"),
        ("small", "FATURA_Template10_Instance82_borderless-small"),
    ],
)
def test_borderless(table_detector, layout_data_path, size, template_name):
    template_img_path = os.path.join(layout_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(layout_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(layout_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())
