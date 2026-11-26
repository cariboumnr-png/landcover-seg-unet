# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Unified logging utilities for console and file destinations.

Provides a unified `Logger` class wrapping standard library logging
with support for console and delayed file handlers, structured
formatting, and convenience severity methods.

Public APIs:
    - `Logger`: Manages file and console logging handlers.
'''

# standard imports
import logging
import pathlib


# ----- public classes
class Logger:
    '''
    Handle logging messages to a file and optionally the console.

    Attributes:
        logger:
            Underlying standard library `logging.Logger` instance.
    '''

    def __init__(
        self,
        name: str | None = None,
        log_file: str | pathlib.Path | None = None,
        log_lvl: int = logging.DEBUG,
        console_lvl: int | None = logging.INFO,
        enable_file_log: bool = True
    ):
        '''
        Initialize the `Logger` instance.

        If `name` is not provided, the script file name is used, and if
        `log_file` is not provided, a `proj.log` file is created under
        the current working directory.

        Args:
            name:
                Name of the logger; defaults to script file name when
                `None`.
            log_file:
                Path to the log file.
            log_lvl:
                Logging level for the file handler.
            console_lvl:
                Logging level for the console handler; `None` disables
                console logging.
            enable_file_log:
                Whether to write text logs to file.
        '''
        if name is None:
            name = pathlib.Path(__file__).name

        if log_file is None:
            log_file = pathlib.Path.cwd() / 'logs' / 'proj.log'
        else:
            log_file = pathlib.Path(log_file)
            if not log_file.name.endswith(('.log', 'json')):
                log_file = log_file.with_suffix('.log')

        self.name = name
        self.log_file = log_file
        self.console_lvl = console_lvl
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_lvl)
        self.logger.propagate = False

        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                '%(asctime)s-%(name)s-%(levelname)s\t- %(message)s'
            )

            if enable_file_log:
                file_handler = logging.FileHandler(
                    self.log_file, delay=True
                )
                file_handler.setLevel(log_lvl)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            if console_lvl is not None:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_lvl)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        self._level_dispatch = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical,
        }

    def log(
        self,
        level: str,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log a message with the specified severity level.

        Args:
            level:
                Severity level (`'debug'`, `'info'`, `'warning'`,
                `'error'`, or `'critical'`).
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        if skip_log:
            return
        log_method = self._level_dispatch.get(
            level.lower(), self.logger.info
        )
        log_method(message, exc_info=exc_info)

    def debug(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log a debug-level message.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        self.log('debug', message, skip_log=skip_log, exc_info=exc_info)

    def info(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log an info-level message.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        self.log('info', message, skip_log=skip_log, exc_info=exc_info)

    def warning(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log a warning-level message.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        self.log('warning', message, skip_log=skip_log, exc_info=exc_info)

    def error(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log an error-level message.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        self.log('error', message, skip_log=skip_log, exc_info=exc_info)

    def critical(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''
        Log a critical-level message.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
            exc_info:
                Whether to include exception traceback information.
        '''
        self.log('critical', message, skip_log=skip_log, exc_info=exc_info)

    def exception(
        self,
        message: str,
        skip_log: bool = False
    ) -> None:
        '''
        Log an error-level message with exception traceback.

        Args:
            message:
                Message text to log.
            skip_log:
                Whether to skip emitting the log record.
        '''
        self.log('error', message, skip_log=skip_log, exc_info=True)

    def log_sep(
        self,
        sep: str = '=',
        ln: int = 90
    ) -> None:
        '''
        Log a repeated separator line.

        Args:
            sep:
                Character or string repeated to form the separator line.
            ln:
                Number of repetitions for the separator line.
        '''
        self.log('info', sep * ln)

    def on_close(self) -> None:
        '''Hook invoked by subclasses when closing the logger.'''

    def close(self) -> None:
        '''Close active log handlers and invoke `on_close`.'''
        self.on_close()
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    @staticmethod
    def abs_path(fp: str | pathlib.Path) -> str:
        '''
        Resolve input path to its absolute path string.

        Args:
            fp:
                File or directory path to resolve.

        Returns:
            str:
                Absolute filesystem path string.
        '''
        return str(pathlib.Path(fp).resolve())
