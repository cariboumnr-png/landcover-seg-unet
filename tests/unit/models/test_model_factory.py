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

# pylint: disable=protected-access

'''Unit tests for top-level model factory and validation (factory.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.models.factory as factory
import landseg.models.frames as frames


# ----- build_multihead_unet tests
def test_build_multihead_unet(
    dataspecs,
    mock_backbone_config_factory,
    mock_domain_config_factory,
):
    '''
    Given: A DataSpec and conditioning configurations.
    When: Building a multihead UNet model.
    Then: Return a valid instance of MultiHeadBaseModel.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    conditioning_cfg = {
        'concat': mock_domain_config_factory(use_ids=True, ids_embd_dims=4),
        'film': mock_domain_config_factory(use_vec=True, vec_proj_dims=4),
    }

    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config=conditioning_cfg,
    )

    assert isinstance(model, frames.MultiHeadBaseModel)


# ----- _validate_model_build validation tests
def test_validate_model_build_success(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A correctly constructed model.
    When: Validating the build integrity.
    Then: Complete successfully without raising any exceptions.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    factory._validate_model_build(model, dataspecs)


def test_validate_model_build_output_not_dict(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model whose forward pass outputs a list instead of a dict.
    When: Validating the build integrity.
    Then: Raise a RuntimeError alerting output type mismatch.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.forward = lambda *args, **kwargs: [torch.randn(2, 2, 256, 256)]

    with pytest.raises(RuntimeError, match='Expected dict'):
        factory._validate_model_build(model, dataspecs)


def test_validate_model_build_head_mismatch(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model with output keys mismatching dataspec heads.
    When: Validating the build integrity.
    Then: Raise a RuntimeError indicating head mismatch.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.forward = lambda *args, **kwargs: {
        'wrong_head': torch.randn(2, 2, 256, 256)
    }

    with pytest.raises(RuntimeError, match='Head mismatch'):
        factory._validate_model_build(model, dataspecs)


def test_validate_model_build_invalid_ndim(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model whose outputs are not 4-dimensional tensors.
    When: Validating the build integrity.
    Then: Raise a RuntimeError indicating non-BCHW shape.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.forward = lambda *args, **kwargs: {'head1': torch.randn(2, 2, 256)}

    with pytest.raises(RuntimeError, match='must be BCHW'):
        factory._validate_model_build(model, dataspecs)


def test_validate_model_build_batch_mismatch(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model output whose batch size is incorrect.
    When: Validating the build integrity.
    Then: Raise a RuntimeError indicating batch dimension mismatch.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.forward = lambda *args, **kwargs: {
        'head1': torch.randn(3, 2, 256, 256)
    }

    with pytest.raises(RuntimeError, match='Batch mismatch in head'):
        factory._validate_model_build(model, dataspecs, batch_size=2)


def test_validate_model_build_channel_mismatch(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model output whose channel dimension is incorrect.
    When: Validating the build integrity.
    Then: Raise a RuntimeError indicating class channels mismatch.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.forward = lambda *args, **kwargs: {
        'head1': torch.randn(2, 3, 256, 256)
    }

    with pytest.raises(RuntimeError, match='Channel mismatch in head'):
        factory._validate_model_build(model, dataspecs)


def test_validate_model_build_non_finite(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A model output containing non-finite values like NaN.
    When: Validating the build integrity.
    Then: Raise a RuntimeError.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = factory.build_multihead_unet(
        patch_size=256,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config={},
    )

    nan_tensor = torch.randn(2, 2, 256, 256)
    nan_tensor[0, 0, 0, 0] = float('nan')
    model.forward = lambda *args, **kwargs: {'head1': nan_tensor}

    with pytest.raises(RuntimeError, match='Non-finite values in head'):
        factory._validate_model_build(model, dataspecs)
