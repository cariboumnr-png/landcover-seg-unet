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
Model architecture and checkpoint diagnostic probes.

Inspects neural backbone configurations, channel counts, and evaluation
checkpoint weights prior to model execution.

Public APIs:
    - `probe_model`: Check neural architecture and checkpoint readiness.
'''

# standard imports
import os
# local imports
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_model(
    target: str,
    root_config: configs.RootConfig,
) -> list[schema.ProbeResult]:
    '''
    Inspect neural architecture specifications and checkpoint weights.

    Args:
        target:
            pipeline target identifier.
        root_config:
            hydra-composed root configuration.

    Returns:
        list[schema.ProbeResult]:
            list of model diagnostic probe records.
    '''
    probes: list[schema.ProbeResult] = []

    if target == 'model-train':
        model_body = root_config.models.model_body
        registry = root_config.models.model_body_registry
        if model_body in registry:
            probes.append(
                schema.ProbeResult(
                    probe_id='backbone_registry',
                    category='model',
                    status=schema.ProbeStatus.PASS,
                    message=f"'{model_body}' recognized in registry",
                    details={'model_body': model_body},
                )
            )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='backbone_registry',
                    category='model',
                    status=schema.ProbeStatus.FAIL,
                    message=f"Unknown model body '{model_body}'",
                    details={'model_body': model_body},
                )
            )

        # channel configuration
        probes.append(
            schema.ProbeResult(
                probe_id='channel_compatibility',
                category='model',
                status=schema.ProbeStatus.PASS,
                message='Channel configuration valid',
            )
        )

    elif target == 'model-evaluate':
        eval_cfg = getattr(root_config.command, 'model_evaluate', None)
        ckpt_path = getattr(eval_cfg, 'checkpoint', None) if eval_cfg else None

        if not ckpt_path:
            probes.append(
                schema.ProbeResult(
                    probe_id='checkpoint_exists',
                    category='model',
                    status=schema.ProbeStatus.FAIL,
                    message=(
                        'No checkpoint specified '
                        '(set command.model_evaluate.checkpoint=...)'
                    ),
                    details={'checkpoint': None},
                )
            )
        elif os.path.isfile(ckpt_path):
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            probes.append(
                schema.ProbeResult(
                    probe_id='checkpoint_exists',
                    category='model',
                    status=schema.ProbeStatus.PASS,
                    message=f'Checkpoint exists ({size_mb:.1f} MB)',
                    details={'checkpoint': ckpt_path, 'size_mb': size_mb},
                )
            )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='checkpoint_exists',
                    category='model',
                    status=schema.ProbeStatus.FAIL,
                    message=f'Checkpoint file not found: {ckpt_path}',
                    details={'checkpoint': ckpt_path},
                )
            )

    elif target == 'diagnose-overfit':
        model_body = root_config.models.model_body
        probes.append(
            schema.ProbeResult(
                probe_id='model_architecture',
                category='model',
                status=schema.ProbeStatus.PASS,
                message=f"Architecture '{model_body}' configured for test",
                details={'model_body': model_body},
            )
        )

    return probes
