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

# pylint: disable=too-many-branches

'''
Pipeline execution
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows as workflows


COMMANDS = typing.Literal[
    'default',
    'world-grid',
    'data-harmonize',
    'data-ingest',
    'data-prepare',
    'diagnose-overfit',
    'model-evaluate',
    'model-train',
    'batch-ingest',
    'preflight',
    'study-analysis',
    'study-sweep'
]


# ----- public functions
def execute_pipeline(root_config: configs.RootConfig) -> typing.Any:
    '''Run the selected CLI pipeline with resolved configuration.'''
    command = root_config.command.name
    results = None

    match command:
        case 'default':
            workflows.execute_default_action(root_config)

        case 'preflight':
            results = _dispatch_preflight(root_config)

        case 'world-grid':
            pipelines.WorldGridGeneration(root_config).run()

        case 'data-harmonize':
            pipelines.DataHarmonization(root_config).run()

        case 'data-ingest':
            pipelines.DataIngestion(root_config).run()

        case 'data-prepare':
            pipelines.DataPreparation(root_config).run()

        case 'model-train':
            pipelines.ModelTraining(root_config).run()

        case 'model-evaluate':
            pipelines.ModelEvaluation(root_config).run()

        case 'diagnose-overfit':
            workflows.execute_diagnose_overfit(root_config)

        case 'batch-ingest':
            workflows.execute_batch_ingest(root_config)

        case 'study-analysis':
            workflows.execute_study_analysis(root_config)

        case 'study-sweep':
            results = workflows.execute_study_sweep(root_config)

        case _:
            raise KeyError(f'Unknown command: {command}; allowed: {COMMANDS}')

    return results


# ----- private helpers
def _dispatch_preflight(root_config: configs.RootConfig) -> typing.Any:
    '''Dispatch pre-flight checks for target pipeline(s).'''
    target = root_config.command.preflight.target
    pipeline_runners: dict[str, typing.Callable[[], typing.Any]] = {
        'world-grid': lambda: (
            pipelines.WorldGridGeneration(root_config).preflight()
        ),
        'data-harmonize': lambda: (
            pipelines.DataHarmonization(root_config).preflight()
        ),
        'data-ingest': lambda: (
            pipelines.DataIngestion(root_config).preflight()
        ),
        'data-prepare': lambda: (
            pipelines.DataPreparation(root_config).preflight()
        ),
        'model-train': lambda: (
            pipelines.ModelTraining(root_config).preflight()
        ),
        'model-evaluate': lambda: (
            pipelines.ModelEvaluation(root_config).preflight()
        ),
    }

    if target in pipeline_runners:
        return pipeline_runners[target]()

    if target == 'all':
        return [runner_fn() for runner_fn in pipeline_runners.values()]

    allowed = sorted(list(pipeline_runners.keys()) + ['all'])
    raise KeyError(
        f'Target "{target}" not supported for pipeline preflight; '
        f'allowed: {allowed}'
    )
