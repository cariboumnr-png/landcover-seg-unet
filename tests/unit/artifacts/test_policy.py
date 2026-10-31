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

'''Unit tests for `landseg.artifacts.policy`.'''

# local imports
import landseg.artifacts.policy as policy_mod


# ----- `LifecyclePolicy` tests
def test_lifecycle_policy_members():
    '''
    Given: The `LifecyclePolicy` enumeration.
    When: Accessing enum members and member count.
    Then: Verify member names and total count match expected policies.
    '''
    assert policy_mod.LifecyclePolicy.LOAD_ONLY.name == 'LOAD_ONLY'
    assert policy_mod.LifecyclePolicy.LOAD_OR_FAIL.name == 'LOAD_OR_FAIL'
    assert policy_mod.LifecyclePolicy.BUILD_IF_MISSING.name == 'BUILD_IF_MISSING'
    assert policy_mod.LifecyclePolicy.REBUILD.name == 'REBUILD'
    assert policy_mod.LifecyclePolicy.REBUILD_IF_STALE.name == 'REBUILD_IF_STALE'

    # verify total count of enum members
    assert len(policy_mod.LifecyclePolicy) == 5
