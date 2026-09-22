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

'''Unit tests for loss builder module (builder.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.tasks.loss.builder as builder
import landseg.session.engine.tasks.loss.composite as composite
import landseg.session.engine.tasks.loss.primitives as primitives


def test_build_headlosses(mock_hspecs, session_config):
    '''
    Given: `HeadSpecs` describing prediction heads and a
        `CompositeLossConfig`.
    When: `build_headlosses` is called.
    Then: Return a `HeadLosses` instance mapping each head name to a
        `CompositeLoss`.
    '''
    hlosses = builder.build_headlosses(
        mock_hspecs,
        config=session_config.engine.engine_tasks.loss_configs,
        ignore_index=255,
        spectral_band_indices=[0, 1]
    )

    assert isinstance(hlosses, builder.HeadLosses)
    assert len(hlosses) == 2
    assert isinstance(hlosses.as_dict()['head_1'], composite.CompositeLoss)
    assert isinstance(hlosses.as_dict()['head_2'], composite.CompositeLoss)
    assert isinstance(hlosses['head_1'], composite.CompositeLoss)
    assert isinstance(hlosses['head_2'], composite.CompositeLoss)


def test_build_headlosses_with_per_head_taxonomy(mock_hspecs, session_config):
    '''
    Given: `HeadSpecs` where head_1 has similarity_matrix and head_2 has None.
    When: `build_headlosses` is called with ecological weight > 0.
    Then: `EcologicalSimilarityLoss` is attached only to head_1.
    '''
    mock_hspecs['head_1'].similarity_matrix = torch.eye(2)
    mock_hspecs['head_2'].similarity_matrix = None

    cfg = session_config.engine.engine_tasks.loss_configs
    cfg.focal.weight = 0.5
    cfg.dice.weight = 0.5
    cfg.spectral.weight = 0.0
    cfg.tv.weight = 0.0
    cfg.ecological.weight = 0.2

    hlosses = builder.build_headlosses(
        mock_hspecs,
        config=cfg,
        ignore_index=255,
    )

    head_1_losses = hlosses['head_1'].losses
    head_2_losses = hlosses['head_2'].losses

    assert any(
        isinstance(l, primitives.EcologicalSimilarityLoss)
        for l in head_1_losses
    )
    assert not any(
        isinstance(l, primitives.EcologicalSimilarityLoss)
        for l in head_2_losses
    )


def test_build_headlosses_with_explicit_matrix_override(
    mock_hspecs, session_config
):
    '''
    Given: `HeadSpecs` with no similarity matrices and explicit matrix override.
    When: `build_headlosses` is called with ecological_similarity_matrix.
    Then: All heads receive `EcologicalSimilarityLoss` with the override matrix.
    '''
    mock_hspecs['head_1'].similarity_matrix = None
    mock_hspecs['head_2'].similarity_matrix = None

    cfg = session_config.engine.engine_tasks.loss_configs
    cfg.ecological.weight = 0.2

    hlosses = builder.build_headlosses(
        mock_hspecs,
        config=cfg,
        ignore_index=255,
        ecological_similarity_matrix=torch.eye(2)
    )

    assert any(
        isinstance(l, primitives.EcologicalSimilarityLoss)
        for l in hlosses['head_1'].losses
    )
    assert any(
        isinstance(l, primitives.EcologicalSimilarityLoss)
        for l in hlosses['head_2'].losses
    )
