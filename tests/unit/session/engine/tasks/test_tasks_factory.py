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

'''Unit tests for engine tasks factory (tasks/factory.py).'''

# local imports
import landseg.session.engine.tasks.builder as task_factory
import landseg.session.engine.tasks.heads as heads
import landseg.session.engine.tasks.loss as loss
import landseg.session.engine.tasks.metrics as metrics
import landseg.session.engine.tasks.regularization as regularization


def test_build_engine_tasks_success(dataspecs, session_config):
    '''
    Given: Valid `DataSpecs` and `TasksConfig`.
    When: Calling `build_engine_tasks`.
    Then: Return populated `EngineTasks` containing all components.
    '''
    tasks = task_factory.build_engine_tasks(
        dataspecs,
        session_config.engine_tasks
    )

    assert isinstance(tasks, task_factory.EngineTasks)
    assert isinstance(tasks.headspecs, heads.HeadSpecs)
    assert isinstance(tasks.headlosses, loss.HeadLosses)
    assert isinstance(tasks.headmetrics, metrics.HeadMetrics)
    assert isinstance(
        tasks.multihead_regularization,
        regularization.ConsistencyRegularizer
    )
    assert isinstance(
        tasks.multihead_metrics,
        metrics.MTLMetricsAggregator
    )


def test_build_engine_tasks_with_constraints(
    dataspecs, session_config, mock_constraint
):
    '''
    Given: Task config populated with multi-task constraints.
    When: Calling `build_engine_tasks`.
    Then: Constraints are compiled and wired into regularization.
    '''
    session_config.engine_tasks.mtl_constraints = [mock_constraint()]
    tasks = task_factory.build_engine_tasks(
        dataspecs,
        session_config.engine_tasks
    )

    assert tasks.multihead_regularization.constraints is not None
    assert len(tasks.multihead_regularization.constraints) == 1
