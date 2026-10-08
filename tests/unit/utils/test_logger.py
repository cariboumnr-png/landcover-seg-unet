# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Unit tests for the custom logger (logger.py).'''

# standard imports
import logging
import os
# local imports
import landseg.utils.logger as logger


def test_logger_initialization(tmp_path):
    '''
    Given: A target file path and logging level.
    When: Initializing a Logger instance.
    Then: Return a valid Logger instance with expected defaults.
    '''
    log_file = tmp_path / 'test_init.log'
    log_inst = logger.Logger(
        name='test_logger',
        log_file=str(log_file),
        log_lvl=logging.INFO
    )

    assert log_inst.name == 'test_logger'
    assert log_inst.log_file == str(log_file)

    log_inst.close()


def test_logger_disabled_console(tmp_path):
    '''
    Given: A console logging level of None.
    When: Initializing a Logger instance.
    Then: No StreamHandler is attached to the logger.
    '''
    log_file = tmp_path / 'test_silent.log'
    log_inst = logger.Logger(
        name='test_silent_logger',
        log_file=str(log_file),
        console_lvl=None
    )

    stream_handlers = [
        h for h in log_inst.logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 0

    log_inst.close()


def test_logger_convenience_methods(tmp_path):
    '''
    Given: A Logger instance with file logging enabled at DEBUG level.
    When: Calling debug, info, warning, error, and critical methods.
    Then: Write all messages with appropriate level markers to file.
    '''
    log_file = tmp_path / 'test_levels.log'
    log_inst = logger.Logger(
        name='test_levels',
        log_file=str(log_file),
        log_lvl=logging.DEBUG,
        enable_file_log=True
    )

    log_inst.debug('debug message')
    log_inst.info('info message')
    log_inst.warning('warning message')
    log_inst.error('error message')
    log_inst.critical('critical message')
    log_inst.close()

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'DEBUG' in content and 'debug message' in content
    assert 'INFO' in content and 'info message' in content
    assert 'WARNING' in content and 'warning message' in content
    assert 'ERROR' in content and 'error message' in content
    assert 'CRITICAL' in content and 'critical message' in content


def test_logger_exception_method(tmp_path):
    '''
    Given: An active exception context.
    When: Calling logger.exception.
    Then: Write error message and exception traceback to file.
    '''
    log_file = tmp_path / 'test_exception.log'
    log_inst = logger.Logger(
        name='test_exc',
        log_file=str(log_file),
        log_lvl=logging.ERROR,
        enable_file_log=True
    )

    try:
        raise ValueError('sample error')
    except ValueError:
        log_inst.exception('An error occurred')

    log_inst.close()

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'ERROR' in content
    assert 'An error occurred' in content
    assert 'ValueError: sample error' in content


def test_logger_skip_log(tmp_path):
    '''
    Given: A Logger instance with file logging enabled.
    When: Calling convenience methods with skip_log=True.
    Then: Messages with skip_log=True are omitted from file.
    '''
    log_file = tmp_path / 'test_skip.log'
    log_inst = logger.Logger(
        name='test_skip',
        log_file=str(log_file),
        log_lvl=logging.INFO,
        enable_file_log=True
    )

    log_inst.info('persisted message', skip_log=False)
    log_inst.info('skipped message', skip_log=True)
    log_inst.close()

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'persisted message' in content
    assert 'skipped message' not in content


def test_logger_writes_to_file(tmp_path):
    '''
    Given: A log path with file logging enabled.
    When: Logging messages of varying levels.
    Then: Write eligible level messages to file and exclude debugs.
    '''
    log_file = tmp_path / 'test_write.log'
    log_inst = logger.Logger(
        name='test_writer',
        log_file=str(log_file),
        log_lvl=logging.INFO,
        enable_file_log=True
    )

    log_inst.log('info', 'Test message info')
    log_inst.log('debug', 'Test message debug')
    log_inst.close()

    assert os.path.exists(log_file)

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'Test message info' in content
    assert 'Test message debug' not in content


def test_logger_separator(tmp_path):
    '''
    Given: A Logger instance.
    When: Invoking log_sep.
    Then: Output a line separator containing the character repeated.
    '''
    log_file = tmp_path / 'test_sep.log'
    log_inst = logger.Logger(
        name='test_sep',
        log_file=str(log_file),
        log_lvl=logging.INFO
    )

    log_inst.log_sep(sep='*', ln=10)
    log_inst.close()

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert '**********' in content
