# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods

'''Fixtures for testing `landseg.session.engine.runtime.executor` module.'''

# third-party imports
import pytest
# local imports
import landseg.session.engine.runtime.tasks.loss.builder as loss_builder
import landseg.session.engine.runtime.tasks.metrics.segmentation.builder as metrics_builder
import landseg.session.engine.runtime.tasks.heads.specs as headspecs


@pytest.fixture
def mock_hspecs(dataspecs):
    return headspecs.build_headspecs(dataspecs, alpha_fn='inverse')

    # defined in dataspecs fixture @unit/conftest.py
    # includes two heads as:
    # class_counts={
    #     'head_1': [100, 200],
    #     'head_2': [50, 150, 250],
    # },
    # logits_adjust={
    #     'head_1': [0.2, 0.1],
    #     'head_2': [0.1, 0.1, 0.1],
    # },
    # head_parent={'head_1': None, 'head_2': None},
    # head_parent_cls={'head_1': None, 'head_2': None},


@pytest.fixture
def mock_hlosses(mock_hspecs, session_config):
    return loss_builder.build_headlosses(
        mock_hspecs,
        config=session_config.engine_tasks.loss_configs,
        ignore_index=255,
        spectral_band_indices=None
    )


@pytest.fixture
def mock_hmetrics(mock_hspecs):
    return metrics_builder.build_headmetrics(
        mock_hspecs,
        ignore_index=255
    )
