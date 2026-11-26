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
Top-level namespace for `landseg.geopipe.harmonize`.

Exposes raster harmonization, manifest compilation, and logging APIs
via lazy resolution to keep import order simple and circular-free.

Public APIs:
    - `HarmonizationContext`: Execution context for harmonization.
    - `HarmonizationLogger`: Logger tracking ETL progress and summary.
    - `build_harmonization_context`: Load reference context from report.
    - `run_data_harmonization`: Pipeline runner.
    - `validators`: Preflight validation namespace.
'''

# pylint: disable=import-outside-toplevel

# standard imports
import pathlib
import typing
# local imports
import landseg._lazy as _lazy


class _Validators:
    '''Validation namespace for harmonization preflight checks.'''

    @staticmethod
    def grid_reference(grid_report_path: pathlib.Path):
        '''
        Load and validate world grid reference.

        Args:
            grid_report_path:
                Path to the persisted world grid JSON report.

        Returns:
            tuple[GridLayout, str, pathlib.Path]:
                Tuple of loaded `GridLayout`, grid ID, and grid path.
        '''
        import landseg.geopipe.harmonize.context as context
        return context.load_grid_reference(grid_report_path)

    @staticmethod
    def manifest_contract(manifest_path: pathlib.Path):
        '''
        Validate input dataset manifest (JSON or YAML).

        Args:
            manifest_path:
                Path to the dataset manifest file.

        Returns:
            tuple[ManifestEntry, ...]:
                Tuple of validated and normalized manifest entries.
        '''
        import landseg.geopipe.harmonize.manifest.builder as builder
        return builder.DatasetManifestBuilder.from_file(manifest_path).compile()

validators = _Validators()


__all__ = [
    'HarmonizationContext',
    'HarmonizationLogger',
    'build_harmonization_context',
    'run_data_harmonization',
    'validators',
]


# for static check
if typing.TYPE_CHECKING:
    from .context import (
        HarmonizationContext,
        build_harmonization_context,
    )
    from .logger import HarmonizationLogger
    from .pipeline import run_data_harmonization


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.context': [
        'HarmonizationContext',
        'build_harmonization_context',
    ],
    '.logger': ['HarmonizationLogger'],
    '.pipeline': ['run_data_harmonization'],
})
