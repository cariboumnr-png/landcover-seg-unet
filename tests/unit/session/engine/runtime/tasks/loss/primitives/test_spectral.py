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

'''Unit tests for Spectral smoothness loss module (spectral.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.spectral as spectral


def test_spectral_loss_invalid_neighbour():
    '''
    Given: Invalid neighbour parameter value.
    When: Instantiating `SpectralSmoothnessLoss`.
    Then: Raise `ValueError`.
    '''
    with pytest.raises(ValueError, match='Neighbourhood must be 4 or 8.'):
        _ = spectral.SpectralSmoothnessLoss(
            alpha=1.0,
            neighbour=6,
            spectral_bands=None,
            ignore_index=255
        )


def test_spectral_loss_forward_4_neighbour():
    '''
    Given: Logits, targets, and features.
    When: `forward` is called on `SpectralSmoothnessLoss` with 4-neighborhood offsets.
    Then: Return a valid scalar smoothness loss tensor.
    '''
    loss_module = spectral.SpectralSmoothnessLoss(
        alpha=1.0, neighbour=4, spectral_bands=None, ignore_index=255
    )
    # B=1, C=2, H=3, W=3
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    loss = loss_module(logits, targets, masks=None, features=features)

    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_spectral_loss_forward_8_neighbour():
    '''
    Given: Logits, targets, and features.
    When: `forward` is called on `SpectralSmoothnessLoss` with 8-neighborhood offsets.
    Then: Return a valid scalar smoothness loss tensor.
    '''
    loss_module = spectral.SpectralSmoothnessLoss(
        alpha=1.0, neighbour=8, spectral_bands=None, ignore_index=255
    )
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    loss = loss_module(logits, targets, masks=None, features=features)

    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_spectral_loss_validation():
    '''
    Given: Invalid shapes or missing features.
    When: `forward` is called on `SpectralSmoothnessLoss`.
    Then: Raise `ValueError`.
    '''
    loss_mod = spectral.SpectralSmoothnessLoss(
        alpha=1.0,
        neighbour=4,
        spectral_bands=None,
        ignore_index=255
    )
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    # missing features
    with pytest.raises(ValueError, match='Features are required'):
        _ = loss_mod(
            logits,
            targets,
            masks=None,
            features=None
        )

    # wrong logits dimensions
    with pytest.raises(ValueError, match='Expected logits with shape'):
        _ = loss_mod(
            torch.randn((1, 2, 3)),
            targets,
            masks=None,
            features=features
        )

    # wrong targets dimensions
    with pytest.raises(ValueError, match='Expected targets with shape'):
        _ = loss_mod(
            logits,
            torch.zeros((1, 3)),
            masks=None,
            features=features
        )

    # wrong features dimensions
    with pytest.raises(ValueError, match='Expected features with shape'):
        _ = loss_mod(
            logits,
            targets,
            masks=None,
            features=torch.randn((1, 4, 3))
        )

    # mismatch batch size
    with pytest.raises(ValueError, match=r'Batch.*logits.*targets'):
        _ = loss_mod(
            logits,
            torch.zeros((2, 3, 3), dtype=torch.long),
            masks=None,
            features=features
        )

    with pytest.raises(ValueError, match=r'Batch.*features.*logits'):
        _ = loss_mod(
            logits,
            targets,
            masks=None,
            features=torch.randn((2, 4, 3, 3))
        )

    # mismatch spatial size
    with pytest.raises(ValueError, match=r'Spatial.*logits.*targets'):
        _ = loss_mod(
            logits,
            torch.zeros((1, 4, 4), dtype=torch.long),
            masks=None,
            features=features
        )

    with pytest.raises(ValueError, match=r'Spatial.*features.*logits'):
        _ = loss_mod(
            logits,
            targets,
            masks=None,
            features=torch.randn((1, 4, 4, 4))
        )


def test_spectral_loss_spectral_bands():
    '''
    Given: Specific spectral_bands indices.
    When: `forward` is called.
    Then: Correctly subset features channel dimension.
    '''
    # only use features from channels 1 and 3
    loss_module = spectral.SpectralSmoothnessLoss(
        alpha=1.0, neighbour=4, spectral_bands=[1, 3], ignore_index=255
    )
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    loss = loss_module(logits, targets, masks=None, features=features)

    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_spectral_loss_masks_and_ignore():
    '''
    Given: Targets containing ignore_index and a down-weighting mask.
    When: `forward` is called on `SpectralSmoothnessLoss`.
    Then: Correctly apply mask and ignore weights during neighborhood
        pairwise computation.
    '''
    loss_mod = spectral.SpectralSmoothnessLoss(
        alpha=1.0,
        neighbour=4,
        spectral_bands=None,
        ignore_index=255
    )
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    # test with masks
    mask = torch.ones((1, 3, 3), dtype=torch.bool)
    loss_unmasked = loss_mod(logits, targets, masks=None, features=features)
    _ = loss_mod(logits, targets, masks={0.5: mask}, features=features)

    # since all weights are scaled down uniformly by 0.5,
    # loss sum and denom sum scale by same factor.
    # but let's test a mask that zeros out some pixels.
    zero_mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    zero_mask[0, 1, 1] = True  # downweights center pixel
    loss_zero_masked = loss_mod(
        logits,
        targets,
        masks={0.0: zero_mask},
        features=features
    )

    assert loss_unmasked.item() != loss_zero_masked.item()

    # test with ignore_index
    targets[0, 1, 1] = 255
    loss_ignore = loss_mod(logits, targets, masks=None, features=features)
    assert loss_ignore.item() != loss_unmasked.item()
