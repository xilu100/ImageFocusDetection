import ast
import inspect
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

_LOG_INITIALIZED = False
_COMPLETE_LOG_STREAM = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _ensure_logging_initialized():
    global _LOG_INITIALIZED, _COMPLETE_LOG_STREAM, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if _LOG_INITIALIZED:
        return

    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    complete_log_file = log_file.with_name(f"{log_file.stem}_complete.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_file
            for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    _COMPLETE_LOG_STREAM = open(complete_log_file, "a", encoding="utf-8")
    if _ORIGINAL_STDOUT is None:
        _ORIGINAL_STDOUT = sys.stdout
    if _ORIGINAL_STDERR is None:
        _ORIGINAL_STDERR = sys.stderr

    sys.stdout = _TeeStream(_ORIGINAL_STDOUT, _COMPLETE_LOG_STREAM)
    sys.stderr = _TeeStream(_ORIGINAL_STDERR, _COMPLETE_LOG_STREAM)

    _LOG_INITIALIZED = True


def _infer_save_argument_name():
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return "value"

    caller = frame.f_back.f_back
    if caller is None:
        return "value"

    try:
        frame_info = inspect.getframeinfo(caller)
        if not frame_info.code_context:
            return "value"

        code_line = frame_info.code_context[0].strip()
        tree = ast.parse(code_line)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save":
                if not node.args:
                    return "value"

                arg_src = ast.get_source_segment(code_line, node.args[0])
                if arg_src:
                    return arg_src
                break
    except (SyntaxError, ValueError, TypeError):
        pass

    fallback_info = inspect.getframeinfo(caller)
    if fallback_info.code_context:
        match = re.search(r"save\((.*?)\)", fallback_info.code_context[0])
        if match:
            return match.group(1).strip()
    return "value"


def _is_string_literal_expression(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        return False
    return isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, str)


def save(value):
    _ensure_logging_initialized()
    arg_name = _infer_save_argument_name()
    if isinstance(value, str) and _is_string_literal_expression(arg_name):
        logging.info("%s", value)
        return
    logging.info("%s = %s", arg_name, value)


def print_and_save(*args, sep=" ", end="\n", flush=False):
    _ensure_logging_initialized()
    print(*args, sep=sep, end=end, flush=flush)
    message = sep.join(str(arg) for arg in args)
    logging.info("%s", message)
