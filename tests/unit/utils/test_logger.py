# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Unit tests for the custom logger (logger.py).
'''

# standard imports
import logging
import os
# local imports
import landseg.utils.logger as logger

def test_logger_initialization(tmp_path):
    '''Test logger initialization and correct attribute assignments.'''
    log_file = tmp_path / 'test_init.log'
    log_inst = logger.Logger(
        name='test_logger',
        log_file=str(log_file),
        log_lvl=logging.INFO
    )

    assert log_inst.name == 'test_logger'
    assert log_inst.log_file == str(log_file)
    assert not log_inst.silent

    log_inst.close()

def test_logger_silent_property(tmp_path):
    '''Test the silent property when console_lvl is None.'''
    log_file = tmp_path / 'test_silent.log'
    log_inst = logger.Logger(
        name='test_silent_logger',
        log_file=str(log_file),
        console_lvl=None
    )

    assert log_inst.silent is True

    log_inst.close()

def test_logger_writes_to_file(tmp_path):
    '''Test that messages are written to the log file correctly.'''
    log_file = tmp_path / 'test_write.log'
    log_inst = logger.Logger(
        name='test_writer',
        log_file=str(log_file),
        log_lvl=logging.INFO,
        enable_file_log=True
    )

    log_inst.log('info', 'Test message info')
    # should not be logged as log_lvl is INFO
    log_inst.log('debug', 'Test message debug')
    log_inst.close()

    assert os.path.exists(log_file)

    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'Test message info' in content
    assert 'Test message debug' not in content

def test_logger_get_child(tmp_path):
    '''Test child logger creation and handler propagation.'''
    log_file = tmp_path / 'test_child.log'
    parent = logger.Logger(
        name='parent',
        log_file=str(log_file),
        log_lvl=logging.INFO
    )
    child = parent.get_child('child')

    assert child.name == 'parent.child'
    assert child.log_file == parent.log_file
    assert child.console_lvl == parent.console_lvl

    child.log('info', 'Message from child')
    parent.close()

    assert os.path.exists(log_file)
    with open(log_file, 'r', encoding='UTF-8') as f:
        content = f.read()

    assert 'parent.child-INFO' in content
    assert 'Message from child' in content

def test_logger_separator(tmp_path):
    '''Test that logger correctly logs separators.'''
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
