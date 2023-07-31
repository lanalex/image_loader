import logging
import logging.handlers
import os
from logging.handlers import SysLogHandler
import platform

import os
import multiprocessing

num_cores = multiprocessing.cpu_count()
os.environ['NUMEXPR_MAX_THREADS'] = str(num_cores)

def initialize_logger():
    LOGS_DIR = os.path.expanduser("~/seedoo_logs")

    write_to_file = bool(os.environ.get("SEEDOO_WRITE_LOGS_TO_FILE", "True"))
    # Get root logger
    logger = logging.getLogger()

    # Check if logger already has handlers (i.e., it's already initialized)
    if logger.hasHandlers():
        return logger

    # Create a standard log format
    log_format = "DoLoopSeeDoo: %(asctime)s [%(levelname)s] %(name)s (%(module)s.%(filename)s:%(lineno)d): %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    syslog_address = '/var/run/syslog' if platform.system() == 'Darwin' else '/dev/log'
    syslog_handler = SysLogHandler(address=syslog_address, facility=SysLogHandler.LOG_LOCAL0)
    syslog_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO) # Default level to ERROR
    logger.addHandler(syslog_handler)

    # If the write_to_file flag is set, add file handlers
    if write_to_file:
        if not os.path.exists(LOGS_DIR):
            os.mkdir(LOGS_DIR)

        log_file = os.path.join(LOGS_DIR, "seedoo.log")
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_log_file = os.path.join(LOGS_DIR, "seedoo_error.log")
        error_file_handler = logging.handlers.RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=5)
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    return logger
