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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

'''
Fixtures for testing `landseg.session.engine.runtime.batch` module.
'''

# standard imports
import dataclasses
# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.session as session_schema
import landseg.session.engine.runtime.tasks.loss.builder as loss_builder
import landseg.session.engine.runtime.tasks.metrics as metrics
import landseg.session.engine.runtime.tasks.heads as headspecs

# aliases
field = dataclasses.field


@pytest.fixture
def mock_hspecs(dataspecs):
    # see dataspecs fixture @unit/conftest.py
    return headspecs.build_headspecs(dataspecs, alpha_fn='inverse')


@pytest.fixture
def mock_hlosses(mock_hspecs):
    return loss_builder.build_headlosses(
        mock_hspecs,
        config=session_schema._LossTypesConfig(),
        ignore_index=255,
        spectral_band_indices=None
    )


@pytest.fixture
def mock_hmetrics(mock_hspecs):
    return metrics.build_headmetrics(
        mock_hspecs,
        ignore_index=255
    )


@pytest.fixture
def mock_constraint():
    def _create(
        name: str = 'rule_1',
        source_head: str = 'head_1',
        trigger_val: int = 1,
        target_head: str = 'head_2',
        forbidden: list[int] | None = None
    ):
        if forbidden is None:
            forbidden = [2]
        return session_schema._MTLConstraints(
            name=name,
            source_head=source_head,
            trigger_val=trigger_val,
            target_head=target_head,
            forbidden=forbidden
    )
    return _create
