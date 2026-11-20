# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Unit tests for GDAL cloud runtime configuration profiles.'''

# standard imports
import os
# third-party imports
import rasterio
# local imports
import landseg.geopipe.rasters.env as rasters_env


# ----- `configure_gdal_defaults` tests
def test_configure_gdal_defaults_populates_missing_keys(monkeypatch):
    '''
    Given: Clean process environment without GDAL cloud keys.
    When: Invoking `configure_gdal_defaults(override=False)`.
    Then: All defaults are applied to `os.environ` and returned.
    '''
    for key in rasters_env.DEFAULT_GDAL_CONFIG:
        monkeypatch.delenv(key, raising=False)

    applied = rasters_env.configure_gdal_defaults(override=False)

    for key, expected_val in rasters_env.DEFAULT_GDAL_CONFIG.items():
        assert applied[key] == expected_val
        assert os.environ[key] == expected_val


def test_configure_gdal_defaults_preserves_existing_without_override(
    monkeypatch,
):
    '''
    Given: Pre-existing custom GDAL environment variable.
    When: Invoking `configure_gdal_defaults(override=False)`.
    Then: Custom value is preserved and not overwritten.
    '''
    monkeypatch.setenv('GDAL_CACHEMAX', '256MB')
    applied = rasters_env.configure_gdal_defaults(override=False)
    assert applied['GDAL_CACHEMAX'] == '256MB'
    assert os.environ['GDAL_CACHEMAX'] == '256MB'


def test_configure_gdal_defaults_overwrites_with_override_true(
    monkeypatch,
):
    '''
    Given: Pre-existing custom GDAL environment variable.
    When: Invoking `configure_gdal_defaults(override=True)`.
    Then: Custom value is overwritten with project default.
    '''
    monkeypatch.setenv('GDAL_CACHEMAX', '256MB')
    applied = rasters_env.configure_gdal_defaults(override=True)
    assert applied['GDAL_CACHEMAX'] == '512MB'
    assert os.environ['GDAL_CACHEMAX'] == '512MB'


# ----- `gdal_cloud_env` tests
def test_gdal_cloud_env_context_applies_options():
    '''
    Given: Cloud GDAL context options.
    When: Entering `gdal_cloud_env` block.
    Then: Active rasterio environment reflects configured options.
    '''
    with rasters_env.gdal_cloud_env() as env:
        assert env.options['GDAL_DISABLE_READDIR_ON_OPEN'] == 'EMPTY_DIR'
        assert env.options['GDAL_NUM_THREADS'] == 'ALL_CPUS'
        assert env.options['GDAL_CACHEMAX'] == 512 * 1024 * 1024


def test_gdal_cloud_env_accepts_overrides():
    '''
    Given: Explicit overrides passed to `gdal_cloud_env`.
    When: Entering context block.
    Then: Overrides supersede defaults while maintaining other options.
    '''
    with rasters_env.gdal_cloud_env(GDAL_CACHEMAX='1024MB') as env:
        assert env.options['GDAL_CACHEMAX'] == 1024 * 1024 * 1024
        assert env.options['GDAL_DISABLE_READDIR_ON_OPEN'] == 'EMPTY_DIR'
