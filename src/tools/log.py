import ast
import inspect
import logging
import re
import sys
from threading import Lock
from datetime import datetime
from pathlib import Path

LOG_INITIALIZED = False
COMPLETE_LOG_STREAM = None
ORIGINAL_STDOUT = None
ORIGINAL_STDERR = None
LOG_FILE_PATH = None
COMPLETE_LOG_FILE_PATH = None
FILE_HANDLER = None


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


class TimestampedLevelLineStream:
    def __init__(self, stream, level):
        self.stream = stream
        self.level = level
        self.line_start = True
        self.lock = Lock()

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        if not data:
            return 0

        with self.lock:
            for ch in data:
                if self.line_start and ch not in ("\n", "\r"):
                    prefix = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    prefix = f"{prefix} [{self.level}] "
                    self.stream.write(prefix)
                    self.line_start = False
                self.stream.write(ch)
                if ch in ("\n", "\r"):
                    self.line_start = True
        return len(data)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def close(self):
        with self.lock:
            self.stream.close()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def ensure_logging_initialized():
    global LOG_INITIALIZED, COMPLETE_LOG_STREAM, ORIGINAL_STDOUT, ORIGINAL_STDERR
    global LOG_FILE_PATH, COMPLETE_LOG_FILE_PATH, FILE_HANDLER
    if LOG_INITIALIZED:
        return

    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    complete_log_file = log_file.with_name(f"{log_file.stem}_complete.log")
    LOG_FILE_PATH = log_file
    COMPLETE_LOG_FILE_PATH = complete_log_file
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
        FILE_HANDLER = file_handler

    raw_complete_stream = open(complete_log_file, "a", encoding="utf-8")
    complete_info_stream = TimestampedLevelLineStream(raw_complete_stream, "INFO")
    complete_error_stream = TimestampedLevelLineStream(raw_complete_stream, "ERROR")
    COMPLETE_LOG_STREAM = complete_info_stream
    if ORIGINAL_STDOUT is None:
        ORIGINAL_STDOUT = sys.stdout
    if ORIGINAL_STDERR is None:
        ORIGINAL_STDERR = sys.stderr

    sys.stdout = TeeStream(ORIGINAL_STDOUT, complete_info_stream)
    sys.stderr = TeeStream(ORIGINAL_STDERR, complete_error_stream)

    LOG_INITIALIZED = True


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
    if COMPLETE_LOG_STREAM is not None:
        COMPLETE_LOG_STREAM.flush()


def get_current_log_paths():
    ensure_logging_initialized()
    return LOG_FILE_PATH, COMPLETE_LOG_FILE_PATH


def close_logs():
    global LOG_INITIALIZED, COMPLETE_LOG_STREAM, FILE_HANDLER
    ensure_logging_initialized()
    flush_logs()

    root_logger = logging.getLogger()
    if FILE_HANDLER is not None:
        try:
            root_logger.removeHandler(FILE_HANDLER)
        except Exception:
            pass
        try:
            FILE_HANDLER.close()
        except Exception:
            pass
        FILE_HANDLER = None

    if COMPLETE_LOG_STREAM is not None:
        try:
            COMPLETE_LOG_STREAM.close()
        except Exception:
            pass
        COMPLETE_LOG_STREAM = None

    if ORIGINAL_STDOUT is not None:
        sys.stdout = ORIGINAL_STDOUT
    if ORIGINAL_STDERR is not None:
        sys.stderr = ORIGINAL_STDERR

    LOG_INITIALIZED = False
