import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import pytest

import cli


def test_parse_cli_basic(monkeypatch):
    test_args = [
        "prog",
        "input.jpg",
        "output_dir",
        "--threshold",
        "0.8",
        "--annotate",
        "--pretrained_model",
        "custom/model",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    args = cli.parse_cli()
    assert args.doc_path == "input.jpg"
    assert args.output_dir == "output_dir"
    assert args.threshold == 0.8
    assert args.annotate is True
    assert args.pretrained_model == "custom/model"


def test_parse_cli_defaults(monkeypatch):
    test_args = ["prog", "input.pdf", "output_dir"]
    monkeypatch.setattr(sys, "argv", test_args)
    args = cli.parse_cli()
    assert args.doc_path == "input.pdf"
    assert args.output_dir == "output_dir"
    assert args.threshold == 0.75
    assert args.annotate is False
    assert args.pretrained_model == "TahaDouaji/detr-doc-table-detection"


def test_cli_missing_required(monkeypatch):
    test_args = ["prog", "input.pdf"]  # Missing output_dir
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_cli()
    assert excinfo.value.code != 0


def test_cli_invalid_threshold(monkeypatch):
    test_args = ["prog", "input.jpg", "output_dir", "--threshold", "not_a_float"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_cli()
    assert excinfo.value.code != 0


def test_cli_help_message(monkeypatch, capsys):
    test_args = ["prog", "--help"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_cli()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Detect tables in a document" in captured.out
    assert "--threshold" in captured.out
