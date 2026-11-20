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

'''
GDAL runtime environment and configuration profiles.

Provides centralized defaults and context management for GDAL and
rasterio configuration options, optimized for cloud object storage
and Unity Catalog Volumes.

Public APIs:
    - `DEFAULT_GDAL_CONFIG`: Default cloud-optimized GDAL options.
    - `configure_gdal_defaults`: Apply cloud defaults to environment.
    - `gdal_cloud_env`: Context manager applying cloud GDAL options.
'''

# standard imports
import contextlib
import os
import typing
# third-party imports
import rasterio
import rasterio.env


# ----- configuration constants
DEFAULT_GDAL_CONFIG: typing.Final[dict[str, str]] = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'GDAL_NUM_THREADS': 'ALL_CPUS',
    'GDAL_CACHEMAX': '512MB',
}


# ----- public functions
def configure_gdal_defaults(
    override: bool = False,
) -> dict[str, str]:
    '''
    Configure process-level GDAL environment defaults for cloud I/O.

    Applies default settings to `os.environ` if not already set,
    including disabling recursive directory scanning on cloud mounts,
    enabling multi-threaded compression, and expanding cache size.

    Args:
        override:
            Whether to overwrite existing environment variable values.

    Returns:
        dict[str, str]:
            Dictionary of active GDAL configuration options.
    '''
    applied: dict[str, str] = {}
    for key, value in DEFAULT_GDAL_CONFIG.items():
        if override or key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]
    return applied


@contextlib.contextmanager
def gdal_cloud_env(
    **overrides: typing.Any,
) -> typing.Iterator[rasterio.env.Env]:
    '''
    Context manager applying cloud-optimized GDAL configuration.

    Merges `DEFAULT_GDAL_CONFIG` with any provided override options.

    Args:
        **overrides:
            Custom GDAL option overrides for this context.

    Yields:
        rasterio.env.Env:
            Active rasterio environment context.
    '''
    opts: dict[str, typing.Any] = {**DEFAULT_GDAL_CONFIG, **overrides}
    if 'GDAL_CACHEMAX' in opts:
        opts['GDAL_CACHEMAX'] = _parse_cache_max(opts['GDAL_CACHEMAX'])
    with rasterio.env.Env(**opts) as env:
        yield env


# ----- private helpers
def _parse_cache_max(val: str | int) -> int:
    '''Convert string cache size representations to integer bytes.'''
    if isinstance(val, int):
        return val
    cleaned = str(val).strip().upper()
    if cleaned.endswith('GB'):
        return int(float(cleaned[:-2]) * 1024 * 1024 * 1024)
    if cleaned.endswith('MB'):
        return int(float(cleaned[:-2]) * 1024 * 1024)
    if cleaned.endswith('KB'):
        return int(float(cleaned[:-2]) * 1024)
    return int(cleaned)
