# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

'''Project-wide unit test suite for lazy-loaded module resolution.'''

# standard imports
import importlib
import pkgutil
import types
# third-party imports
import pytest
# local imports
import landseg


def _discover_lazy_modules() -> list[types.ModuleType]:
    '''Discover all package modules in `landseg` that define `__getattr__`.'''
    modules = [landseg] if '__getattr__' in landseg.__dict__ else []
    prefix = landseg.__name__ + '.'
    for _, modname, ispkg in pkgutil.walk_packages(landseg.__path__, prefix):
        if ispkg:
            try:
                mod = importlib.import_module(modname)
                if '__getattr__' in mod.__dict__ and hasattr(mod, '__all__'):
                    modules.append(mod)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    return modules


LAZY_MODULES = _discover_lazy_modules()


@pytest.mark.parametrize(
    'module',
    LAZY_MODULES,
    ids=lambda m: m.__name__
)
def test_package_lazy_imports(module: types.ModuleType):
    '''
    Given: Any package module in `landseg` implementing lazy `__getattr__`.
    When: Accessing every symbol declared in `__all__`.
    Then: Every attribute resolves cleanly and invalid attribute raises AttributeError.
    '''
    assert hasattr(module, '__all__'), f'{module.__name__} missing __all__'

    for symbol in module.__all__:
        attr = getattr(module, symbol, None)
        assert attr is not None, (
            f'Failed to resolve symbol {symbol!r} in {module.__name__}'
        )

    with pytest.raises(AttributeError, match='has no attribute'):
        _ = getattr(module, '_non_existent_invalid_symbol_')
