"""Logging setup for the webviewer.

Everything the webviewer writes to the terminal goes through loguru, so that a
log collector (Railway, Docker, ...) sees exactly one entry per record, on the
stream matching its severity:

* TRACE / DEBUG / INFO / SUCCESS / WARNING -> stdout (informational)
* ERROR / CRITICAL                         -> stderr (actual failures)

Railway derives the severity of an entry from the stream it arrives on, so
loguru's default of sending *everything* to stderr makes every start-up
warning and success message show up as an error. Records are also collapsed
onto a single line, because a collector splits multi-line output into one
entry per line, which scatters a single message over a dozen rows.
"""

import logging
import os
import sys

from loguru import logger

# levels which are informational rather than failures -> stdout
INFO_LEVELS = frozenset(["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING"])

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)

_configured = False


def _single_line(text):
    """Escape the line breaks in a formatted record.

    Args:
        text (str): formatted log record, possibly spanning several lines.

    Returns:
        str: the same record as a single line.
    """
    return text.rstrip("\n").replace("\r", "").replace("\n", "\\n")


def _stream_sink(stream):
    """Build a loguru sink writing single-line records to a given stream.

    Args:
        stream (io.TextIOBase): stream to write the records to.

    Returns:
        callable: sink usable by ``logger.add``.
    """

    def sink(message):
        stream.write(_single_line(message) + "\n")
        stream.flush()

    return sink


class _InterceptHandler(logging.Handler):
    """Forward records of the standard library logging to loguru.

    Third-party parts of the stack (aiohttp, wslink, trame) log through the
    standard library, and would otherwise bypass the formatting and stream
    routing set up here.
    """

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=6, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(level=None):
    """Route all webviewer output through loguru with container-friendly
    formatting. Calling this more than once is a no-op.

    Args:
        level (str | None): minimum level to emit. Defaults to the ``LOG_LEVEL``
            environment variable, or ``INFO``.
    """
    global _configured
    if _configured:
        return

    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.add(
        _stream_sink(sys.stdout),
        level=level,
        format=LOG_FORMAT,
        filter=lambda record: record["level"].name in INFO_LEVELS,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        _stream_sink(sys.stderr),
        level=level,
        format=LOG_FORMAT,
        filter=lambda record: record["level"].name not in INFO_LEVELS,
        backtrace=False,
        diagnose=False,
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    quiet_progress_bars()

    _configured = True


def quiet_progress_bars():
    """Disable tqdm progress bars when the output is not a terminal.

    The bars redraw themselves with carriage returns, which a log collector
    keeps verbatim - a single mesh conversion turns into unreadable entries
    such as ``Create nodes: 0it [00:00, ?it/s]\\rCreate nodes: 672it [...]``.
    Set ``FOURC_PROGRESS_BARS=on`` to keep them regardless.
    """
    setting = os.environ.get("FOURC_PROGRESS_BARS", "auto").lower()
    if setting == "on":
        return
    if setting == "auto" and sys.stderr.isatty():
        return

    try:
        from tqdm import std as tqdm_std
    except ImportError:  # tqdm is an indirect dependency only
        return

    original_init = tqdm_std.tqdm.__init__
    if getattr(original_init, "_fourc_silenced", False):
        return

    def __init__(self, *args, **kwargs):
        # "disable" is the 11th positional argument; only override it when the
        # caller did not pass it positionally.
        if len(args) <= 10:
            kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    __init__._fourc_silenced = True
    tqdm_std.tqdm.__init__ = __init__


def route_vtk_messages():
    """Send VTK's own warnings and errors through loguru.

    VTK writes to stderr directly, so its messages - including the harmless
    ``bad X server connection`` notice emitted while probing for a display -
    are reported as errors and carry no timestamp. Forwarding them keeps their
    real severity. Does nothing if the installed VTK does not support it.
    """
    try:
        from vtkmodules.vtkCommonCore import vtkOutputWindow
    except ImportError:  # pragma: no cover - depends on the VTK build
        return

    def _forward(level):
        def callback(caller, event, text):
            logger.log(level, f"VTK: {str(text).strip()}")

        callback.CallDataType = "string0"
        return callback

    try:
        output_window = vtkOutputWindow()
        # keep a reference, the observers die with the object otherwise
        for event, level in (
            ("ErrorEvent", "ERROR"),
            ("WarningEvent", "WARNING"),
            ("MessageEvent", "INFO"),
            ("TextEvent", "DEBUG"),
        ):
            output_window.AddObserver(event, _forward(level))
        output_window.SetDisplayModeToNever()
        vtkOutputWindow.SetInstance(output_window)
        globals()["_vtk_output_window"] = output_window
    except Exception as exc:  # pragma: no cover - depends on the VTK build
        logger.debug(f"Could not route VTK messages through the logger: {exc}")
