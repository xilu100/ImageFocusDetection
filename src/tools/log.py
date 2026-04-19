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
_LOG_FILE_PATH = None
_COMPLETE_LOG_FILE_PATH = None
_FILE_HANDLER = None


class TeeStream:
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


def ensure_logging_initialized():
    global _LOG_INITIALIZED, _COMPLETE_LOG_STREAM, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    global _LOG_FILE_PATH, _COMPLETE_LOG_FILE_PATH, _FILE_HANDLER
    if _LOG_INITIALIZED:
        return

    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    complete_log_file = log_file.with_name(f"{log_file.stem}_complete.log")
    _LOG_FILE_PATH = log_file
    _COMPLETE_LOG_FILE_PATH = complete_log_file
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
        _FILE_HANDLER = file_handler

    _COMPLETE_LOG_STREAM = open(complete_log_file, "a", encoding="utf-8")
    if _ORIGINAL_STDOUT is None:
        _ORIGINAL_STDOUT = sys.stdout
    if _ORIGINAL_STDERR is None:
        _ORIGINAL_STDERR = sys.stderr

    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _COMPLETE_LOG_STREAM)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _COMPLETE_LOG_STREAM)

    _LOG_INITIALIZED = True


def infer_save_argument_name():
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


def is_string_literal_expression(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        return False
    return isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, str)


def save(value):
    ensure_logging_initialized()
    arg_name = infer_save_argument_name()
    if isinstance(value, str) and is_string_literal_expression(arg_name):
        logging.info("%s", value)
        return
    logging.info("%s = %s", arg_name, value)


def print_and_save(*args, sep=" ", end="\n", flush=False):
    ensure_logging_initialized()
    print(*args, sep=sep, end=end, flush=flush)
    message = sep.join(str(arg) for arg in args)
    logging.info("%s", message)


def flush_logs():
    ensure_logging_initialized()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    if _COMPLETE_LOG_STREAM is not None:
        _COMPLETE_LOG_STREAM.flush()


def get_current_log_paths():
    ensure_logging_initialized()
    return _LOG_FILE_PATH, _COMPLETE_LOG_FILE_PATH


def close_logs():
    global _LOG_INITIALIZED, _COMPLETE_LOG_STREAM, _FILE_HANDLER
    ensure_logging_initialized()
    flush_logs()

    root_logger = logging.getLogger()
    if _FILE_HANDLER is not None:
        try:
            root_logger.removeHandler(_FILE_HANDLER)
        except Exception:
            pass
        try:
            _FILE_HANDLER.close()
        except Exception:
            pass
        _FILE_HANDLER = None

    if _COMPLETE_LOG_STREAM is not None:
        try:
            _COMPLETE_LOG_STREAM.close()
        except Exception:
            pass
        _COMPLETE_LOG_STREAM = None

    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT
    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR

    _LOG_INITIALIZED = False


class LogApi:
    def save(self, value):
        save(value)

    def print_and_save(self, *args, sep=" ", end="\n", flush=False):
        print_and_save(*args, sep=sep, end=end, flush=flush)

    def flush_logs(self):
        flush_logs()

    def get_current_log_paths(self):
        return get_current_log_paths()

    def close_logs(self):
        close_logs()


log_api = LogApi()
