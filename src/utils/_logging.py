import logging
import sys

# Clear root logger handlers added by python -m to prevent duplicate output
logging.root.handlers.clear()

_LOG = logging.getLogger("spad")
_LOG_HANDLER = logging.StreamHandler(sys.stdout)
_LOG_HANDLER.setFormatter(logging.Formatter(
    "[%(name)s] %(levelname)-7s %(message)s"))
_LOG.addHandler(_LOG_HANDLER)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    return _LOG.getChild(name) if name else _LOG


def set_log_level(level: int | str) -> None:
    _LOG.setLevel(level)
