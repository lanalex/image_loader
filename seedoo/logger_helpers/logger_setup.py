import logging
import logging.handlers
from logging.handlers import SysLogHandler

def initialize_logger(write_to_file=False):
    # Get root logger
    logger = logging.getLogger()

    # Check if logger already has handlers (i.e., it's already initialized)
    if logger.hasHandlers():
        return logger

    # Create a standard log format
    log_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Syslog handler (logs to syslogd)
    syslog_handler = SysLogHandler(address='/dev/log', facility=SysLogHandler.LOG_LOCAL0)
    syslog_handler.setFormatter(formatter)

    logger.setLevel(logging.ERROR) # Default level to ERROR
    logger.addHandler(syslog_handler)

    # If the write_to_file flag is set, add file handlers
    if write_to_file:
        log_file = "logger.log"
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_log_file = "error_log.log"
        error_file_handler = logging.handlers.RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=5)
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    return logger

# Usage with file logging
initialize_logger(write_to_file=True)

# Usage without file logging
# initialize_logger()
