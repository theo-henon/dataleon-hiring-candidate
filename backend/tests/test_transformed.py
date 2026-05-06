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
def transformed_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "transformed")


def test_blurry(table_detector, transformed_data_path):
    """Test the blurry image detection. (Applied Gaussian blur with sigma=2)"""
    template_name = "FATURA_Template24_Instance156_blurry"
    template_img_path = os.path.join(transformed_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(transformed_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(transformed_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "template_name",
    [
        "FATURA_Template24_Instance156_rotated-left",
        "FATURA_Template24_Instance156_rotated-right",
        "FATURA_Template24_Instance156_rotated-180",
        "FATURA_Template24_Instance156_rotated-5-left",
        "FATURA_Template24_Instance156_rotated-5-right",
    ],
)
def test_rotated(table_detector, transformed_data_path, template_name):
    """Test the rotated image detection. (Applied 90 degrees rotation)"""
    template_img_path = os.path.join(transformed_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(transformed_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(transformed_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


@pytest.mark.parametrize(
    "template_name",
    [
        "FATURA_Template24_Instance156_mirrored-horizontal",
        "FATURA_Template24_Instance156_mirrored-vertical",
    ],
)
def test_mirrored(table_detector, transformed_data_path, template_name):
    """Test the mirrored image detection. (Applied horizontal mirroring)"""
    template_img_path = os.path.join(transformed_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(transformed_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(transformed_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_shadow(table_detector, transformed_data_path):
    """Test the shadow image detection. (Applied shadow effect)"""
    template_name = "FATURA_Template24_Instance156_shadow"
    template_img_path = os.path.join(transformed_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(transformed_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(transformed_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())


def test_compressed(table_detector, transformed_data_path):
    """Test the compressed image detection. (Applied JPEG compression with quality=10)"""
    template_name = "FATURA_Template24_Instance156_compressed"
    template_img_path = os.path.join(transformed_data_path, f"{template_name}.jpg")
    results_list = table_detector.detect_tables(template_img_path)
    expected_tables = load_ground_truth(transformed_data_path, template_name)
    assert len(results_list) == 1
    detected_scores, _, detected_boxes, annotated_img = results_list[0]
    save_annotated_image(transformed_data_path, template_name, annotated_img, expected_tables)
    validate_tables(expected_tables, detected_scores.tolist(), detected_boxes.tolist())
