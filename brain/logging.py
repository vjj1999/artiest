"""Brain 轻量日志模块"""
import logging
import sys

logger = logging.getLogger("brain")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
