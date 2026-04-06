"""Centralized logging configuration utilities for FinancialAlgoV2.

Provides:
- Structured JSON logs for infrastructure pipelines.
- Plain text logs for local development.
- Optional rotating file sink.
- Basic sensitive-data redaction filter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator


TRACE_ID_CTX: ContextVar[str] = ContextVar("trace_id", default="")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class SensitiveDataFilter(logging.Filter):
    """Redacts common sensitive tokens from log messages."""

    _PATTERNS = (
        re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*[^\s,;]+"),
        re.compile(r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*[^\s,;]+"),
        re.compile(r"(?i)bearer\s+[a-z0-9\-\._~\+\/]+=*"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            return True

        sanitized = rendered
        for pattern in self._PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        if sanitized != rendered:
            record.msg = sanitized
            record.args = ()

        return True


class ContextEnricherFilter(logging.Filter):
    """Adds request-scoped context fields to log records."""

    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment
        self._hostname = os.getenv("COMPUTERNAME", "unknown-host")

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self._service
        record.environment = self._environment
        record.host = self._hostname
        record.trace_id = TRACE_ID_CTX.get()
        return True


class JsonFormatter(logging.Formatter):
    """Outputs logs as newline-delimited JSON for log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, "service", "financial_algo"),
            "env": getattr(record, "environment", "dev"),
            "host": getattr(record, "host", "unknown-host"),
            "trace_id": getattr(record, "trace_id", ""),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def _build_stream_handler(use_json: bool) -> logging.Handler:
    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | trace=%(trace_id)s | %(message)s"
            )
        )
    return handler


def _build_file_handler(
    log_dir: Path,
    use_json: bool,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=str(log_dir / "financial_algo.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    if use_json:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | trace=%(trace_id)s | %(message)s"
            )
        )
    return file_handler


def setup_logging(
    *,
    service: str | None = None,
    environment: str | None = None,
    level: str | None = None,
    json_logs: bool | None = None,
    log_to_file: bool | None = None,
    log_dir: str | Path | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for the project and return the root logger."""
    root = logging.getLogger()
    already_configured = getattr(root, "_financial_algo_logging_configured", False)
    if already_configured and not force:
        return root

    service_name = service or os.getenv("FINANCIAL_ALGO_SERVICE", "financial_algo")
    env_name = environment or os.getenv("FINANCIAL_ALGO_ENV", "dev")
    level_name = (level or os.getenv("FINANCIAL_ALGO_LOG_LEVEL", "INFO")).upper()
    use_json = _env_bool("FINANCIAL_ALGO_LOG_JSON", True) if json_logs is None else json_logs
    use_file = _env_bool("FINANCIAL_ALGO_LOG_TO_FILE", False) if log_to_file is None else log_to_file
    resolved_log_dir = Path(log_dir or os.getenv("FINANCIAL_ALGO_LOG_DIR", "logs"))
    resolved_max_bytes = _env_int("FINANCIAL_ALGO_LOG_MAX_BYTES", 10_000_000) if max_bytes is None else max_bytes
    resolved_backup_count = _env_int("FINANCIAL_ALGO_LOG_BACKUP_COUNT", 5) if backup_count is None else backup_count

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(getattr(logging, level_name, logging.INFO))

    context_filter = ContextEnricherFilter(service=service_name, environment=env_name)
    redaction_filter = SensitiveDataFilter()

    stream_handler = _build_stream_handler(use_json=use_json)
    stream_handler.addFilter(context_filter)
    stream_handler.addFilter(redaction_filter)
    root.addHandler(stream_handler)

    if use_file:
        file_handler = _build_file_handler(
            log_dir=resolved_log_dir,
            use_json=use_json,
            max_bytes=resolved_max_bytes,
            backup_count=resolved_backup_count,
        )
        file_handler.addFilter(context_filter)
        file_handler.addFilter(redaction_filter)
        root.addHandler(file_handler)

    root._financial_algo_logging_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)


def set_trace_id(trace_id: str) -> None:
    """Set a request/run-scoped trace id for all subsequent logs in context."""
    TRACE_ID_CTX.set(trace_id)


@contextmanager
def trace_context(trace_id: str) -> Iterator[None]:
    """Temporarily set trace id for the active execution context."""
    token = TRACE_ID_CTX.set(trace_id)
    try:
        yield
    finally:
        TRACE_ID_CTX.reset(token)
