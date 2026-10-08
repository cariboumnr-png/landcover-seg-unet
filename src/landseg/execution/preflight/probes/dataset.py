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
Dataset partition and feature contract diagnostic probes.

Inspects train/validation/test split ratios and feature/target contracts.

Public APIs:
    - `split_ratios`: Validate dataset partition split ratio proportions.
    - `dataset_targets`: Inspect configured features and target mappings.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def raw_dataset(
    config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''Inspect raw input raster dataset manifest'''
    try:
        manifest_fp = config.data.harmonization.dataset_manifest
        dirp = os.path.dirname(manifest_fp)
        ctrl = artifacts.Controller[list].load_json_or_fail(manifest_fp)
        ctrl.hash(overwrite=False) # hash upon first open

        manifest = ctrl.fetch()
        if not isinstance(manifest, list): # simple check
            raise RuntimeError('Raw dataset manifest JSON not read as a list')
        n = len(manifest)
        if n == 0:
            raise RuntimeError('Raw dataset manifest JSON appears to be empty')
        return schema.ProbeResult(
            pid=pid or 'source_dataset_manifest',
            category='Dataset',
            status=schema.ProbeStatus.PASS,
            message=f'Found {n} rasters available for harmonization at: {dirp}',
            details={'dataset_manifest_filepath': manifest_fp}
        )

    except (artifacts.ArtifactError, RuntimeError) as e:
        return schema.ProbeResult(
            pid=pid or 'source_dataset_manifest',
            category='Dataset',
            status=schema.ProbeStatus.FAIL,
            message='Raw dataset manifest JSON cannot be read or is invalid',
            details={'error': str(e)}
        )
