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

'''Unit tests for batch engine builder (session/engine/batch/builder.py).'''

# local imports
import landseg.session.engine.batch as batch_mod


# ----- `build_batch_engine` tests
def test_build_batch_engine_success(
    dataspecs,
    mock_dataloaders,
    mock_model,
    session_config,
):
    '''
    Given: Valid dataspecs, dataloaders, model, and engine_exec config.
    When: Calling `build_batch_engine`.
    Then: Return instantiated `BatchEngine`.
    '''
    batch_engine = batch_mod.build_batch_engine(
        dataspecs=dataspecs,
        dataloaders=mock_dataloaders,
        model=mock_model,
        config=session_config.engine_exec,
        device='cpu',
    )

    assert isinstance(batch_engine, batch_mod.BatchEngine)
    assert batch_engine.model is mock_model
    assert batch_engine.state is not None
