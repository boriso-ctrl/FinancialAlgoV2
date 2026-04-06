"""Tests for centralized logging utilities."""

from __future__ import annotations

import json
import logging

from financial_algo.logging_utils import get_logger, setup_logging, trace_context


def test_setup_logging_with_plain_text() -> None:
    root = setup_logging(
        service="unit_test",
        environment="test",
        level="INFO",
        json_logs=False,
        log_to_file=False,
        force=True,
    )
    assert isinstance(root, logging.Logger)
    assert len(root.handlers) >= 1


def test_get_logger_returns_named_logger() -> None:
    setup_logging(force=True)
    logger = get_logger("tests.test_logging_utils")
    assert logger.name == "tests.test_logging_utils"


def test_json_logging_emits_trace_id(capsys) -> None:
    setup_logging(
        service="unit_test",
        environment="test",
        level="INFO",
        json_logs=True,
        log_to_file=False,
        force=True,
    )

    logger = get_logger("tests.logging")
    with trace_context("trace-abc"):
        logger.info("hello world")

    captured = capsys.readouterr().err.strip()
    assert captured
    payload = json.loads(captured)
    assert payload["trace_id"] == "trace-abc"
    assert payload["message"] == "hello world"
