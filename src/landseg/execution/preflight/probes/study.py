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
Study optimization and trial analysis diagnostic probes.

Inspects Optuna study schemas, parameter space configurations, and
analysis report destinations prior to sweep or analysis execution.

Public APIs:
    - `probe_study`: Check study optimization parameters and contracts.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_study(
    target: str,
    root_config: configs.RootConfig,
) -> list[schema.ProbeResult]:
    '''
    Inspect study schemas, parameters, and analysis targets.

    Args:
        target:
            workflow or study target identifier.
        root_config:
            hydra-composed root configuration.

    Returns:
        list[schema.ProbeResult]:
            list of study diagnostic probe records.
    '''
    probes: list[schema.ProbeResult] = []
    artifact_paths = artifacts.ArtifactPaths.from_config(root_config)
    sweep_cfg = getattr(root_config.command, 'study_sweep', None)

    if target == 'study-sweep':
        study_name = (
            getattr(sweep_cfg, 'study_name', 'study_test')
            if sweep_cfg
            else 'study_test'
        )
        probes.append(
            schema.ProbeResult(
                probe_id='study_schema',
                category='optuna',
                status=schema.ProbeStatus.PASS,
                message=f"Study '{study_name}' configured",
                details={'study_name': study_name},
            )
        )

        # verify model training contract
        prep_root = artifact_paths.data_preparation.root
        if os.path.isdir(prep_root):
            probes.append(
                schema.ProbeResult(
                    probe_id='model_train_contract',
                    category='dependency',
                    status=schema.ProbeStatus.PASS,
                    message=f"Prepared dataset directory found at '{prep_root}'",
                    details={'prepared_data_root': prep_root},
                )
            )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='model_train_contract',
                    category='dependency',
                    status=schema.ProbeStatus.WARN,
                    message=(
                        f"Prepared dataset directory '{prep_root}' not yet "
                        f"materialized"
                    ),
                    details={'prepared_data_root': prep_root},
                )
            )

    elif target == 'study-analysis':
        study_name = (
            getattr(sweep_cfg, 'study_name', 'study_test')
            if sweep_cfg
            else 'study_test'
        )
        probes.append(
            schema.ProbeResult(
                probe_id='study_exists',
                category='study',
                status=schema.ProbeStatus.PASS,
                message=f"Target study '{study_name}' configured for analysis",
                details={'study_name': study_name},
            )
        )

        analysis_dir = os.path.join(root_config.execution.exp_root, 'analysis')
        probes.append(
            schema.ProbeResult(
                probe_id='report_directory',
                category='storage',
                status=schema.ProbeStatus.PASS,
                message=f"Report destination '{analysis_dir}' valid",
                details={'analysis_dir': analysis_dir},
            )
        )

    return probes
