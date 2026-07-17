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

'''Unit tests for backbone factory (factory.py).'''

# third-party imports
import pytest
# local imports
import landseg.models.backbones.factory as factory
import landseg.models.backbones.unet.body as body


# ----- build_unet_backbone tests
def test_build_unet_backbone_unet(mock_backbone_config_factory):
    '''
    Given: UNet backbone type config.
    When: Running build_unet_backbone.
    Then: Return a valid UNet body backbone with targeted output channels.
    '''
    cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    backbone = factory.build_unet_backbone(in_ch=3, input_size=256, config=cfg)

    assert isinstance(backbone, body.UNet)
    assert backbone.spatial_divisor == 16
    assert backbone.out_channels == 16


def test_build_unet_backbone_unetpp(mock_backbone_config_factory):
    '''
    Given: UNet++ backbone type config.
    When: Running build_unet_backbone.
    Then: Return a valid UNetPP body backbone.
    '''
    cfg = mock_backbone_config_factory(body_type='unetpp', base_ch=16)
    backbone = factory.build_unet_backbone(in_ch=3, input_size=256, config=cfg)

    assert isinstance(backbone, body.UNetPP)
    assert backbone.spatial_divisor == 16


def test_build_unet_backbone_unetppp(mock_backbone_config_factory):
    '''
    Given: UNet+++ backbone type config.
    When: Running build_unet_backbone.
    Then: Return a valid UNetPPP body backbone.
    '''
    cfg = mock_backbone_config_factory(body_type='unetppp', base_ch=16)
    backbone = factory.build_unet_backbone(in_ch=3, input_size=256, config=cfg)

    assert isinstance(backbone, body.UNetPPP)
    assert backbone.spatial_divisor == 16


def test_build_unet_backbone_invalid_body(mock_backbone_config_factory):
    '''
    Given: An invalid backbone body type.
    When: Running build_unet_backbone.
    Then: Raise a ValueError.
    '''
    cfg = mock_backbone_config_factory(body_type='invalid', base_ch=16)

    with pytest.raises(ValueError, match='Invalid backbone body: invalid'):
        factory.build_unet_backbone(in_ch=3, input_size=256, config=cfg)


def test_build_unet_backbone_indivisible_size(mock_backbone_config_factory):
    '''
    Given: An input spatial size indivisible by the divisor (16).
    When: Running build_unet_backbone.
    Then: Raise a ValueError.
    '''
    cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)

    with pytest.raises(ValueError, match='is not divisible by'):
        factory.build_unet_backbone(in_ch=3, input_size=15, config=cfg)
