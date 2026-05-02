import logging
import json
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
from app.core.config import settings

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON logger for production observability."""
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        log_record['app'] = settings.APP_NAME

def setup_logging():
    """
    Initializes structured logging for the application.
    Configures standard output for container/cloud logging services (e.g., AWS CloudWatch).
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # JSON Handler for stdout (optimized for log aggregation tools)
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

    # Suppress verbose library logs
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('mlflow').setLevel(logging.INFO)

    return logger

# Initialize global logger instance
logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger instance for specific modules.
    """
    return logging.getLogger(name)

# Ensure settings are validated on module import
settings.validate_settings()

logger.info("Logging infrastructure initialized successfully.")