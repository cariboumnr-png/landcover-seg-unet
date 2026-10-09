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

# pylint: disable=protected-access

'''
Unit tests for data preparation configuration schemas.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.data.preparation as prep_sec


# ----- `DataPreparationConfig` tests
def test_preparation_defaults_and_validation():
    '''
    Given: A default `DataPreparationConfig` instance.
    When: Calling `DataPreparationConfig.validate()`.
    Then: Initialize default partition ratios and rebuild flags.
    '''
    dt = prep_sec.DataPreparationConfig()
    dt.validate()

    assert dt.rebuild is False
    assert dt.partition.val_ratio == 0.1
    assert dt.partition.test_ratio == 0.0
    assert dt.hydration.max_skew_rate == 10.0


def test_preparation_features_validation():
    '''
    Given: `DataPreparationConfig` with valid and invalid features types.
    When: `DataPreparationConfig.validate()` runs.
    Then: Accept string or list values, or raise ValueError.
    '''
    # valid features configurations
    valid = prep_sec.DataPreparationConfig(
        features={'sentinel2': 'rgb_nir', 'spectral': ['ndvi']}
    )
    valid.validate()

    # invalid value type
    invalid = prep_sec.DataPreparationConfig(features={'topo': 123})
    with pytest.raises(ValueError, match='Expected type'):
        invalid.validate()


# ----- `CatalogView` tests
def test_catalog_view_validation():
    '''
    Given: `CatalogView` with valid and out-of-range thresholds.
    When: `CatalogView.validate()` is called.
    Then: Validate pixel threshold boundaries or raise ValueError.
    '''
    catalog = prep_sec.DatasetViewConfig(valid_pxs={'image': 0.8, 'label': 0.95})
    catalog.validate()

    invalid_catalog = prep_sec.DatasetViewConfig(valid_pxs={'image': 1.5})
    with pytest.raises(ValueError, match='valid threshold'):
        invalid_catalog.validate()


# ----- `Partition` tests
def test_partition_validation():
    '''
    Given: `Partition` instances with valid and invalid split ratios.
    When: `Partition.validate()` is executed.
    Then: Accept valid ratio bounds [0.0, 1.0] or raise ConfigValidationError.
    '''
    partition = prep_sec.Partition(val_ratio=0.2, test_ratio=0.1)
    partition.validate()

    with pytest.raises(base.ConfigValidationError):
        prep_sec.Partition(val_ratio=-0.1).validate()

    with pytest.raises(base.ConfigValidationError):
        prep_sec.Partition(test_ratio=1.5).validate()


# ----- `Scoring` & `Hydration` tests
def test_scoring_and_hydration_validation():
    '''
    Given: `Scoring` and `Hydration` sub-configuration objects.
    When: `.validate()` is called with valid or negative boundaries.
    Then: Pass valid params or raise ConfigValidationError for negatives.
    '''
    scoring = prep_sec.Scoring(alpha=0.5, beta=0.5)
    scoring.validate()

    with pytest.raises(base.ConfigValidationError):
        prep_sec.Scoring(alpha=-1.0).validate()

    hydration = prep_sec.Hydration(max_skew_rate=5.0)
    hydration.validate()

    with pytest.raises(base.ConfigValidationError):
        prep_sec.Hydration(max_skew_rate=-2.0).validate()
