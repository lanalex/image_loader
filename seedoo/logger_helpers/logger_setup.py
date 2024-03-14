import logging
import logging.handlers
import os
from logging.handlers import SysLogHandler
import platform

import os
import multiprocessing
import torch
import cv2

num_cores = multiprocessing.cpu_count()

# Setting for NumExpr
os.environ['NUMEXPR_MAX_THREADS'] = str(num_cores)

# Setting for OpenMP
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)

# Setting for PyTorch
torch.set_num_threads(num_cores)

# Setting for OpenCV (cv2)
cv2.setNumThreads(num_cores)

import io
import sys

def get_numpy_config():
    import numpy as np
    config_dict = {}

    for name, info_dict in np.__config__.__dict__.items():
        if name.startswith("_") or type(info_dict) is not dict:
            continue
        config_dict[name] = info_dict

    # Get additional CPU features information
    cpu_features = {
        'baseline': ','.join(np.core._multiarray_umath.__cpu_baseline__),
        'found': ','.join(feature for feature in np.core._multiarray_umath.__cpu_dispatch__ if np.core._multiarray_umath.__cpu_features__[feature]),
        'not_found': ','.join(feature for feature in np.core._multiarray_umath.__cpu_dispatch__ if not np.core._multiarray_umath.__cpu_features__[feature])
    }
    config_dict['cpu_features'] = cpu_features

    return config_dict


def check_optimizations_and_multithreading():
    initialize_logger()
    import numpy as np
    import cv2
    import torch

    logger = logging.getLogger(__name__)

    # NumPy
    np_config = get_numpy_config()
    blas_optimization = np_config.get('blas_opt_info', {})
    lapack_optimization = np_config.get('lapack_opt_info', {})

    if (blas_optimization or lapack_optimization) and \
            ('mkl' in str(blas_optimization) or 'openblas' in str(blas_optimization)):
        logger.info("NumPy has both optimization and multi-threading support.")
    else:
        logger.critical("NumPy does not have both optimization and multi-threading support.")

    # OpenCV (cv2)
    build_info = cv2.getBuildInformation()

    if ('TBB' in build_info or 'Parallel framework' in build_info) and \
            ('OpenBLAS' in build_info or 'LAPACK' in build_info):
        logger.critical("OpenCV has both multi-threading and optimization support.")
    else:
        logger.critical("OpenCV does not have both multi-threading and optimization support.")

    # PyTorch
    pytorch_optimization = torch.backends.mkl.is_available() or torch.cuda.is_available()
    # Assuming OpenMP support for PyTorch; can't directly check
    pytorch_multithreading = True if platform.system() != 'Darwin' else False

    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        # Special handling for Apple Silicon (M1) if needed
        pass

    if pytorch_optimization and pytorch_multithreading:
        logger.critical("PyTorch has both optimization and multi-threading support.")
    else:
        logger.critical("PyTorch does not have both optimization and multi-threading support.")



class WebsocketFilter(logging.Filter):
    def filter(self, record):
        # Reject logs with the specific message
        if 'websockets.server' in record.module:
            return False
        return True

class IgnoreSpecificModuleFilter(logging.Filter):
    def __init__(self, module_to_ignore):
        super().__init__()
        self.module_to_ignore = module_to_ignore

    def filter(self, record):
        # Check if the log record's module name is from the module we want to ignore
        return not record.name.startswith(self.module_to_ignore)


class ExceptionFilter(logging.Filter):
    def filter(self, record):
        return record.exc_info is not None


class ExceptionMessageHandler(logging.StreamHandler):
    def emit(self, record):
        if record.exc_info:
            etype, evalue, etb = record.exc_info
            record.msg = str(evalue)
            record.exc_info = None
        super(ExceptionMessageHandler, self).emit(record)

def initialize_logger():
    # Check if logger already has handlers (i.e., it's already initialized)
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    LOGS_DIR = os.path.expanduser("~/seedoo_logs")
    write_to_file = bool(os.environ.get("SEEDOO_WRITE_LOGS_TO_FILE", "True"))
    log_level_str = os.environ.get("SEEDOO_LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Create a standard log format
    log_format = "SeeDoo:  [%(levelname)s] %(asctime)s (%(name)s.%(funcName)s:%(lineno)d): %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Check if we are in jupyter, if so we don't want to write to stdout
    # Note we catch ONLY the import error, in case there is no ipython or jupyter installed
    # we don't catch a generic error
    add_stdout = True
    try:
        from IPython import get_ipython

        # Determine the correct progress bar to use
        if get_ipython() is not None:
            add_stdout = False
    except ImportError:
        pass

    if add_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(WebsocketFilter())
        logger.addHandler(stdout_handler)

    syslog_address = '/var/run/syslog' if platform.system() == 'Darwin' else '/dev/log'
    syslog_handler = SysLogHandler(address=syslog_address, facility=SysLogHandler.LOG_LOCAL0)
    syslog_handler.setFormatter(formatter)


    logger.setLevel(log_level)
    logger.addHandler(syslog_handler)

    # Exception Handler for stdout - only exception messages, no stack trace
    #exception_handler_stdout = ExceptionMessageHandler(sys.stdout)
    #exception_handler_stdout.setLevel(logging.ERROR)
    #exception_handler_stdout.setFormatter(logging.Formatter("%(message)s"))
    #logger.addHandler(exception_handler_stdout)

    # Exception Handler for stderr - only exception messages, no stack trace
    #exception_handler_stderr = ExceptionMessageHandler(sys.stderr)
    #exception_handler_stderr.setLevel(logging.ERROR)
    #exception_handler_stderr.setFormatter(logging.Formatter("%(message)s"))
    #logger.addHandler(exception_handler_stderr)

    # If the write_to_file flag is set, add file handlers
    if write_to_file:
        if not os.path.exists(LOGS_DIR):
            os.mkdir(LOGS_DIR)

        log_file = os.path.join(LOGS_DIR, f"seedoo_{os.getpid()}.log")
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        exception_log_file = os.path.join(LOGS_DIR, f"seedoo_exception_{os.getpid()}.log")
        exception_file_handler = logging.handlers.RotatingFileHandler(exception_log_file, maxBytes=10*1024*1024, backupCount=5)
        exception_file_handler.setLevel(logging.ERROR)
        exception_file_handler.setFormatter(formatter)
        exception_file_handler.addFilter(ExceptionFilter())
        logger.addHandler(exception_file_handler)

    logging.getLogger('seedoo.io.pandas').setLevel(logging.WARNING)

    return logger


check_optimizations_and_multithreading()