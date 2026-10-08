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
    - `Logger`: manages file and console logging handlers.
'''

# standard imports
import logging
import os


# ----- public classes
class Logger:
    '''
    A class to handle logging messages to a file and optionally to the
    console.

    Attributes:
        logger (logging.Logger): The logger instance.
    '''

    def __init__(
        self,
        name: str | None = None,
        log_file: str | None = None,
        log_lvl: int = logging.DEBUG,
        console_lvl: int | None = logging.INFO,
        enable_file_log: bool = True
    ):
        '''
        Initializes the Logger instance.

        If `name` is not provided, the script file name with be used and
        if `log_file` is not provided, a proj.log file will be created
        at the current working directory.

        Args:
            name (str, optional): Name of the logger. If None use the
                script name.
            log_file (str, optional): Path to the log file.
            log_lvl (int, optional): Logging level for the file handler.
            console_lvl (int, optional): Logging level for the console
                handler. If None, console logging is disabled.
            enable_file_log (bool, optional): Whether to write text logs
            to file.
        '''
        if name is None:
            name = os.path.basename(__file__)

        if log_file is None:
            log_file = os.path.join(os.getcwd(), 'logs', 'proj.log')
        elif not log_file.endswith('.log'):
            root, _ = os.path.splitext(log_file)
            log_file = f'{root}.log'

        self.name = name
        self.log_file = log_file
        self.console_lvl = console_lvl
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

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
        Logs a message with the specified logging level.
        Args:
            level (str): The logging level ('debug', 'info', 'warning',
                'error', 'critical').
            message (str): The message to log.
            skip_log (bool, optional): Flag whether to log or not.
            exc_info (bool, optional): Flag whether to include traceback.
        '''
        if skip_log:
            return
        log_method = self._level_dispatch.get(level.lower(), self.logger.info)
        log_method(message, exc_info=exc_info)

    def debug(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''Log a debug-level message.'''
        self.log('debug', message, skip_log=skip_log, exc_info=exc_info)

    def info(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''Log an info-level message.'''
        self.log('info', message, skip_log=skip_log, exc_info=exc_info)

    def warning(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''Log a warning-level message.'''
        self.log('warning', message, skip_log=skip_log, exc_info=exc_info)

    def error(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''Log an error-level message.'''
        self.log('error', message, skip_log=skip_log, exc_info=exc_info)

    def critical(
        self,
        message: str,
        skip_log: bool = False,
        exc_info: bool = False
    ) -> None:
        '''Log a critical-level message.'''
        self.log('critical', message, skip_log=skip_log, exc_info=exc_info)

    def exception(
        self,
        message: str,
        skip_log: bool = False
    ) -> None:
        '''Log an error-level message with exception traceback.'''
        self.log('error', message, skip_log=skip_log, exc_info=True)

    def log_sep(
        self,
        sep: str = '=',
        ln: int = 90
    ) -> None:
        '''
        Log a separator with a length of repeated string.

        Args:
            sep (str, optional): Repeats to form a separator line
                (default: `'='`).
            ln (int, optional): Length of the line (default: 90).
        '''
        self.log('info', sep * ln)

    def on_close(self) -> None:
        '''Hook for subclasses to execute code when close() is called.'''

    def close(self) -> None:
        '''Closes the file handler.'''
        self.on_close()
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
