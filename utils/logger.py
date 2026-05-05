import logging
import datetime
from pathlib import Path


def get_logger(name:str):
    
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        Path("logs").mkdir(exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(f"logs/{date_str}.log")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "time: %(asctime)s | "
            "name: %(name)s | "
            "level: %(levelname)s | "
            "message: %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger  
