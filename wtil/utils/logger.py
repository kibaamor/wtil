import logging
import os
from typing import Optional

import colorlog

LOG_FORMAT = "[{asctime}][{filename}:{lineno}][{levelname}] {message}"


def create_terminal_handler() -> logging.StreamHandler:
    log_colors_config = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }
    formatter = colorlog.ColoredFormatter(style="{", fmt="{log_color}" + LOG_FORMAT, log_colors=log_colors_config)
    terminal = logging.StreamHandler()
    terminal.setFormatter(formatter)
    return terminal


def create_file_handler(filename: str) -> logging.FileHandler:
    log_dir = os.path.dirname(filename)
    os.makedirs(log_dir, exist_ok=True)

    file = logging.FileHandler(filename)
    formatter = logging.Formatter(fmt=LOG_FORMAT, style="{")
    file.setFormatter(formatter)
    return file


class ExitOnExceptionHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            raise SystemExit(-1)


def config_logger(no_debug: bool, no_terminal: bool, filename: Optional[str]):
    handlers = []
    if not no_terminal:
        handlers.append(create_terminal_handler())
    if filename is not None and len(filename) > 0:
        handlers.append(create_file_handler(filename))
    handlers.append(ExitOnExceptionHandler())

    level = logging.INFO if no_debug else logging.DEBUG

    logging.basicConfig(
        force=True,
        level=level,
        handlers=handlers,
        format=LOG_FORMAT,
        style="{",
    )

    logger = logging.getLogger()
    logger.info(f"log level: {logging.getLevelName(level)}")
    if filename is not None and len(filename) > 0:
        logger.info(f"log file: {filename}")
