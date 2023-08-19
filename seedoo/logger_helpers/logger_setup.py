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
    logger.addFilter(WebsocketFilter())

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

check_optimizations_and_multithreading()