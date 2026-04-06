# FinancialAlgoV2

## Logging System

The project now has a centralized logging system in `financial_algo.logging_utils`.

### Quick start

```python
from financial_algo.logging_utils import get_logger, setup_logging

setup_logging(service="financial_algo", environment="dev")
logger = get_logger(__name__)
logger.info("startup complete")
```

### Features

- JSON logs by default for infrastructure pipelines.
- Optional rotating file logs.
- Trace-id context support via `trace_context(...)`.
- Sensitive token redaction for common secret patterns.

### Environment variables

- `FINANCIAL_ALGO_SERVICE`: service name (default `financial_algo`)
- `FINANCIAL_ALGO_ENV`: environment label (default `dev`)
- `FINANCIAL_ALGO_LOG_LEVEL`: `DEBUG|INFO|WARNING|ERROR|CRITICAL` (default `INFO`)
- `FINANCIAL_ALGO_LOG_JSON`: `true/false` (default `true`)
- `FINANCIAL_ALGO_LOG_TO_FILE`: `true/false` (default `false`)
- `FINANCIAL_ALGO_LOG_DIR`: directory for rotating log files (default `logs`)
- `FINANCIAL_ALGO_LOG_MAX_BYTES`: max file size before rotate (default `10000000`)
- `FINANCIAL_ALGO_LOG_BACKUP_COUNT`: rotated file count to keep (default `5`)