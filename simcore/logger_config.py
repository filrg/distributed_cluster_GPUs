import logging, os
from logging.handlers import RotatingFileHandler

LOGGER_NAME = "SIMU_DC"


def get_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path=os.path.join(log_dir, "project.log")
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Handler ghi log ra file (append mode)
        file_handler = RotatingFileHandler(
            log_path,
            mode='a',            # ghi nối tiếp, không ghi đè
            maxBytes=5_000_000,  # xoay file sau 5MB
            backupCount=3,       # giữ tối đa 3 file cũ
            encoding="utf-8"
        )

        # Định dạng log
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Gắn handler
        logger.addHandler(file_handler)

    return logger
