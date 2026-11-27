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

'''Architectural boundary and path-ownership invariant checks (ADR-0070).'''

# standard imports
import ast
import pathlib

# ----- module constants
_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'src' / 'landseg'
_NOTEBOOKS_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'notebooks'


# ----- boundary tests
def test_visual_does_not_import_artifacts():
    '''
    Given: All Python modules inside `src/landseg/visual/`.
    When: Inspecting AST import nodes.
    Then: Zero imports reference `landseg.artifacts`.
    '''
    violations: list[str] = []
    for py_fpath in sorted((_SRC_ROOT / 'visual').rglob('*.py')):
        tree = ast.parse(py_fpath.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('landseg.artifacts'):
                        violations.append(f'{py_fpath}: {alias.name}')
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ''
                if mod.startswith('landseg.artifacts'):
                    violations.append(f'{py_fpath}: from {mod}')
    assert violations == []


def test_no_glob_in_visual_mlops_or_model_train_notebook():
    '''
    Given: `visual`, `mlops`, and `notebooks/02_model_train.ipynb`.
    When: Scanning source text for `glob(`.
    Then: Zero ad-hoc globbing remains in any of those targets.
    '''
    targets = [
        *sorted((_SRC_ROOT / 'visual').rglob('*.py')),
        *sorted((_SRC_ROOT / 'mlops').rglob('*.py')),
        _NOTEBOOKS_ROOT / '02_model_train.ipynb',
    ]
    hits = [
        str(p)
        for p in targets
        if 'glob(' in p.read_text(encoding='utf-8')
    ]
    assert hits == []


def test_checkpoint_globs_confined_to_artifacts_and_session():
    '''
    Given: All source files under `src/landseg/`.
    When: Scanning for literal `'*_best.pt'` or `'**/*.pt'` patterns.
    Then: Occurrences are confined to `artifacts` and `session`.
    '''
    allowed_roots = {
        _SRC_ROOT / 'artifacts',
        _SRC_ROOT / 'session',
    }
    patterns = ('*_best.pt', '**/*.pt')
    violations: list[str] = []
    for py_fpath in sorted(_SRC_ROOT.rglob('*.py')):
        if any(py_fpath.is_relative_to(root) for root in allowed_roots):
            continue
        content = py_fpath.read_text(encoding='utf-8')
        for pat in patterns:
            if pat in content:
                violations.append(f'{py_fpath}: {pat}')
    assert violations == []
