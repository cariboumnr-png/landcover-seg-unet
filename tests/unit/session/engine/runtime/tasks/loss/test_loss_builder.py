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

'''Unit tests for loss builder module (builder.py).'''

# local imports
import landseg.session.engine.runtime.tasks.heads.specs as head_specs
import landseg.session.engine.runtime.tasks.loss.builder as loss_builder
import landseg.session.engine.runtime.tasks.loss.composite as composite_loss


def test_build_headlosses(dataspecs, session_config):
    '''
    Given: `HeadSpecs` describing prediction heads and a
        `CompositeLossConfig`.
    When: `build_headlosses` is called.
    Then: Return a `HeadLosses` instance mapping each head name to a
        `CompositeLoss`.
    '''
    hspecs = head_specs.build_headspecs(dataspecs, alpha_fn='inverse')

    hlosses = loss_builder.build_headlosses(
        hspecs,
        config=session_config.engine_tasks.loss_configs,
        ignore_index=255,
        spectral_band_indices=[0, 1]
    )

    assert isinstance(hlosses, loss_builder.HeadLosses)
    assert len(hlosses) == 2
    assert isinstance(hlosses.as_dict()['head_1'], composite_loss.CompositeLoss)
    assert isinstance(hlosses.as_dict()['head_2'], composite_loss.CompositeLoss)
    assert isinstance(hlosses['head_1'], composite_loss.CompositeLoss)
    assert isinstance(hlosses['head_2'], composite_loss.CompositeLoss)
