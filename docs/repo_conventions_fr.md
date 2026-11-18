# Hygiène du dépôt et conventions d'ingénierie

Ce document établit les normes de codage unifiées, les frontières architecturales, les lignes directrices de documentation et les pratiques d'hygiène de dépôt pour le projet `landseg`.

Ces conventions s'appliquent à l'ensemble des contributions — qu'elles soient rédigées par des développeurs humains ou par des agents IA autonomes — afin de garantir que le code source demeure maintenable, cohérent, lisible et conforme aux exigences juridiques.

---

## Table des matières

1. [Droit d'auteur de la Couronne et conformité légale](#1-droit-dauteur-de-la-couronne-et-conformité-légale)
2. [Formatage du code et style structurel](#2-formatage-du-code-et-style-structurel)
   - [Limites de longueur de ligne](#limites-de-longueur-de-ligne)
   - [Organisation des imports](#organisation-des-imports)
   - [Espacement vertical et lignes vides](#espacement-vertical-et-lignes-vides)
   - [Délimiteurs de section et ordonnancement des scripts](#délimiteurs-de-section-et-ordonnancement-des-scripts)
   - [Formatage des commentaires](#formatage-des-commentaires)
3. [Spécifications de documentation et docstrings](#3-spécifications-de-documentation-et-docstrings)
   - [Docstrings au niveau des modules](#docstrings-au-niveau-des-modules)
   - [Fonctions et méthodes publiques](#fonctions-et-méthodes-publiques)
   - [Dataclasses et fonctions auxiliaires privées](#dataclasses-et-fonctions-auxiliaires-privées)
4. [Protocoles de typage et analyse statique](#4-protocoles-de-typage-et-analyse-statique)
   - [Protocoles de typage](#protocoles-de-typage)
   - [Philosophie Pylint et qualité du code](#philosophie-pylint-et-qualité-du-code)
5. [Architecture et normes de test](#5-architecture-et-normes-de-test)
   - [Frontière tests unitaires vs intégration](#frontière-tests-unitaires-vs-intégration)
   - [Pas de « Mock Theater »](#pas-de--mock-theater-)
   - [Portée et emplacement des fixtures](#portée-et-emplacement-des-fixtures)
   - [Contrats Pathlib](#contrats-pathlib)
   - [Centralisation des tests d'imports paresseux](#centralisation-des-tests-dimports-paresseux)
   - [Exécution ciblée des tests](#exécution-ciblée-des-tests)
6. [Hygiène du dépôt et flux de contribution](#6-hygiène-du-dépôt-et-flux-de-contribution)
   - [Exécution en environnement virtuel](#exécution-en-environnement-virtuel)
   - [Espace de travail temporaire et d'analyse (`sandbox/`)](#espace-de-travail-temporaire-et-danalyse-sandbox)
   - [Hygiène des diffs et modifications ciblées](#hygiène-des-diffs-et-modifications-ciblées)
   - [Dossiers de décision d'architecture (ADR)](#dossiers-de-décision-darchitecture-adr)
   - [Conventions de messages de validation Git](#conventions-de-messages-de-validation-git)
   - [Descriptions de requêtes de tirage (Pull Requests - PR)](#descriptions-de-requêtes-de-tirage-pull-requests---pr)
7. [Liste de contrôle de référence rapide](#7-liste-de-contrôle-de-référence-rapide)

---

## 1. Droit d'auteur de la Couronne et conformité légale

Chaque fichier source Python (`.py`) de ce dépôt doit débuter par le bloc d'en-tête textuel de 20 lignes relatif au droit d'auteur de la Couronne de l'Ontario (*Ontario Crown Copyright*).

* **Exigence stricte** : Le bloc d'en-tête doit demeurer **entièrement intact et textuellement conforme** aux lignes 1 à 20 de chaque fichier.
* **Aucune modification autorisée** : Ne modifiez pas la formulation, ne reformatez pas les bordures, n'ajustez pas les années et ne retirez aucun élément du texte de licence.

```python
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
```

---

## 2. Formatage du code et style structurel

### Limites de longueur de ligne

* **Lignes de code** : Ne doivent pas dépasser **80 caractères**.
  * *Exception pragmatique* : Si une ligne dépasse 80 caractères de seulement quelques caractères (p. ex. 81 à 83 caractères) et qu'un retour à la ligne nuirait fortement à la lisibilité, laissez-la telle quelle au lieu d'introduire des césures artificielles.
* **Docstrings et commentaires** : Ne doivent pas dépasser **72 caractères**.
  * Les docstrings et commentaires doivent être coupés proprement afin de garantir une lisibilité optimale dans les terminaux et les comparateurs de différences côte à côte (*side-by-side diff*).

### Organisation des imports

Les imports doivent être organisés en exactement trois catégories, séparées par des commentaires de catégorie explicites et en minuscules :

1. `# standard imports`
2. `# third-party imports`
3. `# local imports`

#### Règles et contraintes :
* **Aucune ligne vide entre les catégories** : Le commentaire d'en-tête de catégorie succède immédiatement au bloc d'import précédent sans aucune ligne vide intermédiaire.
* **Aucun alias pour `np` ou `pd`** : N'utilisez jamais d'alias d'import abrégés tels que `import numpy as np` ou `import pandas as pd`. Utilisez toujours l'import direct du module : `import numpy` et `import pandas`.
* **Espacement final après les imports** :
  * Si les imports sont suivis par une **fonction de premier niveau, une fixture, une classe ou un délimiteur de section** : Laissez **exactement deux lignes vides**.
  * Si les imports sont suivis par une **variable globale ou un alias au niveau module** : Laissez **exactement une ligne vide** (puis laissez deux lignes vides après la variable avant toute fonction ou délimiteur de section).

#### Exemple :
```python
# standard imports
import pathlib
import typing
# third-party imports
import numpy
import pandas
import torch
# local imports
import landseg.configs as configs
import landseg.utils as utils


# ----- public functions
def compute_metrics(...) -> dict:
    ...
```

### Espacement vertical et lignes vides

* **Définitions de premier niveau** : Laissez **exactement deux lignes vides** entre les classes, fonctions, fixtures et délimiteurs de section de premier niveau.
* **Méthodes de classe** : Laissez **exactement une ligne vide** entre les méthodes au sein d'une même définition de classe.
* **Espacement consécutif aux docstrings** : Ne laissez **aucune ligne vide** immédiatement après la docstring d'une fonction ou méthode avant la première ligne de code exécutable.

```python
def example_function(value: int) -> int:
    '''
    Calculate transformed output.

    Args:
        value:
            Input integer value.

    Returns:
        int:
            Transformed value.
    '''
    result = value * 2
    return result
```

### Délimiteurs de section et ordonnancement des scripts

Les délimiteurs de section établissent des repères visuels standardisés et facilitent la navigation au sein des modules métier et des suites de tests.

* **Format du délimiteur** : `# ----- `[Symbol]` [description/action en minuscules]`
  * Si le premier mot de la description est une classe ou un symbole de code, entourez-le d'apostrophes inversées (*backticks*) et conservez sa casse exacte (p. ex., `# ----- `HeadManager` initialization`).
* **Espacement du délimiteur** :
  * Précédé de **exactement deux lignes vides**.
  * Suivi de **zéro ligne vide** (la définition ciblée commence immédiatement à la ligne suivante).

#### Ordonnancement des sections dans les scripts métier :
Le cas échéant, les modules Python métier doivent organiser leurs sections selon la séquence suivante (les alias de typage succèdent immédiatement aux imports) :

1. `# ----- typing aliases`
2. `# ----- public types`
3. `# ----- private types`
4. `# ----- public dataclasses`
5. `# ----- private dataclasses`
6. `# ----- public classes`
7. `# ----- private classes`
8. `# ----- public functions`
9. `# ----- private helpers`

### Formatage des commentaires

* **Mots standards** : Les commentaires doivent débuter par une lettre minuscule (p. ex., `# verify shape`, `# standard imports`).
* **Identifiants de code et noms de classes** : Lorsqu'un commentaire fait référence à une classe, une fonction ou un type au niveau module, préservez sa casse d'origine et entourez-le d'apostrophes inversées (*backticks*) :
```
# `LayerNorm` standard deviation check
# dispatch to `execute_command`
```
* **Espacement** : Placez toujours exactement un espace entre le caractère `#` et le texte du commentaire.

---

## 3. Spécifications de documentation et docstrings

Toutes les docstrings doivent respecter la limite de longueur de 72 caractères par ligne.

### Docstrings au niveau des modules

Chaque module Python requiert une docstring au niveau racine, positionnée immédiatement sous l'en-tête du droit d'auteur de la Couronne.

* Structure :
  1. Un résumé concis en une seule ligne.
  2. Un ou plusieurs paragraphes courts expliquant le contexte, l'architecture et l'objectif du module.
  3. Une section `Public APIs:` répertoriant les symboles exportés ainsi qu'une brève description.

```python
'''
Programmatic API entry surface.

Provides the primary orchestrator entry points for invoking training,
evaluation, preflight readiness checks, and intake pipelines.

Public APIs:
    - `run`: Run execution pipeline with resolved configuration.
    - `run_experiment`: Run full experiment lifecycle.
    - `run_preflight`: Run pre-flight diagnostic validation checks.
'''
```

### Fonctions et méthodes publiques

Les fonctions publiques appliquent les docstrings de style Google avec les arguments formatés sur de nouvelles lignes indentées :

* Structure :
  1. Résumé en une ligne, suivi d'un paragraphe descriptif optionnel.
  2. Bloc `Args:` où chaque nom de paramètre est suivi d'un deux-points, avec sa description placée sur une nouvelle ligne indentée.
  3. Bloc `Returns:` avec l'annotation de type et sa description sur une nouvelle ligne indentée.
  4. Bloc `Raises:` (le cas échéant) documentant les exceptions pouvant être levées.

```python
def configure_stage(target: str, strict: bool = False) -> StageConfig:
    '''
    Configure execution stage with target parameters.

    Resolves pipeline recipes and validates compatibility with
    registered stage manifests.

    Args:
        target:
            Identifier of target execution pipeline or stage.
        strict:
            Whether to enforce strict schema validation.

    Returns:
        StageConfig:
            Resolved and validated stage configuration instance.

    Raises:
        ValueError:
            If target identifier is unregistered or invalid.
    '''
    ...
```

### Dataclasses et fonctions auxiliaires privées

Les dataclasses, les fonctions auxiliaires privées et les utilitaires de configuration de test doivent employer une docstring concise en une seule ligne :

```python
@dataclasses.dataclass(frozen=True)
class GridCoordinate:
    '''Immutable spatial grid coordinate representation.'''
    x: int
    y: int


def _resolve_cache_key(prefix: str, identifier: str) -> str:
    '''Compute deterministic cache key from input metadata.'''
    ...
```

---

## 4. Protocoles de typage et analyse statique

### Protocoles de typage

Lors de la définition d'interfaces au moyen de `typing.Protocol` :

* **Agencement compact** : Maintenez les définitions de classes de protocole compactes, avec **aucune ligne vide** au sein du corps de la classe.
* **Aucune docstring sur les signatures** : Ne placez aucune docstring sur les déclarations de méthodes ou de propriétés individuelles au sein du protocole.
* **Propriétés d'attributs** : Définissez les attributs du protocole à l'aide de déclarations `@property` avec annotations de type de retour explicites.
* **Directive Pylint** : Incluez `# pylint: disable=missing-function-docstring` près du sommet du fichier lorsque les définitions de protocole omettent les docstrings de méthodes.

```python
class SessionConfigShape(typing.Protocol):
    '''Interface for session construction configuration.'''
    @property
    def dataloader(self) -> data.DataLoaderConfig: ...
    @property
    def engine(self) -> engine_mod.EngineConfigShape: ...
    @property
    def orchestration(self) -> orchestration_mod.OrchestrationConfigShape: ...
```

### Philosophie Pylint et qualité du code

* **Pas de « course au score » ni d'abus de pragmas** : Ne forcez pas artificiellement un score Pylint de 10.0 en multipliant les balises `# pylint: disable` à travers les fichiers du projet.
* **Désactivations pragmatiques uniquement** : N'utilisez les directives de désactivation que lorsqu'elles sont rigoureusement justifiées (p. ex., `# pylint: disable=protected-access` dans les tests unitaires inspectant l'état interne, ou `# pylint: disable=missing-function-docstring` pour les protocoles).
* **Résoudre la cause fondamentale** : Corrigez les problèmes sous-jacents de typage, d'architecture ou de modularité plutôt que de masquer des diagnostics valides du linter.

---

## 5. Architecture et normes de test

### Frontière tests unitaires vs intégration

La suite de tests impose une séparation structurelle stricte :

* `tests/unit/` :
  * Tests unitaires rapides, isolés et déterministes.
  * Valident la logique métier pure, les transformations mathématiques, les schémas de configuration et les contrats d'étapes isolées.
  * **Interdit dans les tests unitaires** : Opérations d'I/O disque lourdes, dispatch de pipelines multi-étapes, requêtes réseau externes ou orchestration inter-sous-systèmes.
* `tests/integration/` :
  * Workflows de bout en bout et pipelines d'exécution multi-étapes.
  * Valident l'exécution des pipelines, la création d'artefacts sur disque et l'orchestration à l'exécution.
  * Doivent être alimentés par des données synthétiques canoniques générées via `landseg.testing.dummy_data`.

### Pas de « Mock Theater »

* Évitez d'affirmer sur des appels de méthodes privées ou de construire des chaînes d'assertions de mocks profondes (évitez par exemple `assert_called_once_with` sur des fonctions auxiliaires internes).
* Testez les **contrats publics observables** : valeurs de retour, modifications d'état, artefacts générés sur disque et invariants système.
* Utilisez des mocks principalement aux véritables frontières du système (p. ex. accélérateurs matériels, stockage infonuagique distant, appels de sous-processus).

### Portée et emplacement des fixtures

* **Fixtures partagées** : Doivent résider strictement dans les fichiers `conftest.py` selon la hiérarchie appropriée :
  * `tests/conftest.py` : Fixtures au niveau racine.
  * `tests/unit/conftest.py` : Fixtures transversales aux tests unitaires.
  * `tests/unit/<sous-système>/conftest.py` : Fixtures partagées propres à un sous-système.
  * `tests/integration/conftest.py` : Fixtures de données pour les tests d'intégration.
* **Aucune fixture locale au script** : Ne définissez **PAS** de fonctions `@pytest.fixture` au sein des fichiers `test_*.py` individuels.
* **Fonctions auxiliaires privées de configuration** : Convertissez la logique de configuration locale ou de génération d'objets en fonctions auxiliaires privées préfixées par un tiret bas (`_`), placées sous la section `# ----- private helpers` en bas du script de test.

### Contrats Pathlib

* Utilisez systématiquement `pathlib.Path` à travers l'ensemble des fixtures, entrées de test, opérations sur fichiers et validations de chemins.
* Évitez de convertir les chemins en chaînes de caractères (`str`), sauf en cas d'exigence formelle d'une interface externe tierce ou lors de la sérialisation JSON.

### Centralisation des tests d'imports paresseux

* N'ajoutez pas de tests unitaires individuels pour vérifier la résolution des imports paresseux dans les modules `__init__.py` individuels.
* Les imports paresseux à l'échelle du package sont testés de manière centralisée dans `tests/unit/test_lazy_imports.py`.

### Exécution ciblée des tests

* Lors des phases de développement itératif ou de refactorisation, exécutez uniquement les tests unitaires ciblés directement liés aux modules modifiés.
* L'exécution complète de la suite de tests est réservée aux validations d'étapes majeures, aux vérifications préalables à la validation finale et aux pipelines d'intégration continue (CI).

---

## 6. Hygiène du dépôt et flux de contribution

### Exécution en environnement virtuel

Le projet requiert Python 3.12+ et fonctionne au sein d'un environnement virtuel dédié (`.venv`).

* Tous les tests, formatages et exécutions de scripts doivent utiliser les exécutables de l'environnement virtuel :
  * **Windows** : `.\.venv\Scripts\python.exe` et `.\.venv\Scripts\pytest`
  * **Linux / macOS** : `.venv/bin/python` et `.venv/bin/pytest`

### Espace de travail temporaire et d'analyse (`sandbox/`)

Afin de prévenir toute pollution du dépôt et tout enregistrement accidentel de fichiers intermédiaires :

* Le répertoire `sandbox/` est ignoré par Git (`/sandbox/*` dans `.gitignore`).
* Placez tous les scripts d'investigation temporaires, notebooks exploratoires, sessions de débogage ad hoc et artefacts expérimentaux dans `sandbox/`.
* Ne créez jamais de répertoires temporaires non suivis et ne laissez aucun script ad hoc à la racine du dépôt.
* Vous pouvez également effectuer des vérifications ponctuelles au moyen de commandes en ligne de commande éphémères (p. ex., `python -c "..."`).

### Hygiène des diffs et modifications ciblées

* Veillez à ce que les pull requests et commits demeurent strictement limités à la fonctionnalité, au correctif ou à l'objectif de refactorisation précis.
* **Évitez les modifications gratuites** : N'effectuez pas de reformatages massifs, d'ajustements d'espaces blancs ou de modifications de docstrings sur des fichiers existants non ciblés.
* Préservez la clarté de l'historique Git et la pertinence de l'annotation des modifications (*git blame*).

### Dossiers de décision d'architecture (ADR)

Toute nouvelle conception architecturale ou modification structurelle majeure requiert un ADR dans `docs/ADRs/`.

* **ADR proposés** :
  * Nom de fichier : `docs/ADRs/ADR-XXXX-<identifiant-descriptif>.md`
  * En-tête : `**Status:** Proposed`
  * Temps des verbes : Prospectif (« we will... », « will be introduced... », « will execute... »).
* **ADR acceptés** :
  * Une fois la proposition intégralement implémentée, validée et fusionnée :
  * En-tête : `**Status:** Accepted` (ou `**Status:** Accepted — Implemented`)
  * Temps des verbes : Passé (« we have... », « was introduced... »).

### Conventions de messages de validation Git

Les messages de validation doivent être clairs, descriptifs et structurés comme suit :

* **En-tête** : Un résumé concis à l'impératif sur une seule ligne (maximum 72 caractères).
* **Corps** : Liste à puces détaillant les modifications clés, formatée selon les règles suivantes :
  * Chaque puce débute par `* `.
  * Le texte de la puce commence par une lettre minuscule (sauf s'il débute par un identifiant de code entre apostrophes inversées).
  * Tout symbole de code, nom de classe, fonction ou module doit être encadré d'apostrophes inversées (p. ex., `* add `resolve_target_reclass` to `landseg.geopipe``).

#### Exemple :
```text
refactor dataset manifest assembly pipeline

* introduce `ManifestBuilder` protocol in `landseg.geopipe`
* update `resolve_target_reclass` to accept optional category mapping
* migrate manifest serialization to use `pathlib.Path` contracts
* add unit tests for schema validation in `tests/unit/geopipe`
```

### Descriptions de requêtes de tirage (Pull Requests - PR)

* **Aucune URI locale absolue** : N'incluez jamais de chemins d'accès locaux ou d'URI d'environnement absolues (p. ex. `file:///...`) dans le corps des PR ou les historiques de commits. Utilisez des chemins relatifs au dépôt (p. ex., `docs/ADRs/...`, `src/landseg/...`).
* **Description structurée** : Structurez la description de la PR avec :
  1. **Titre** : Résumé concis de la modification.
  2. **Résumé exécutif** : Énoncé du problème et solution apportée.
  3. **Modifications clés** : Liste à puces des changements regroupés par sous-système.
  4. **Vérification et suite de tests** : Commandes exécutées et confirmation du passage des tests.

---

## 7. Liste de contrôle de référence rapide

Avant de soumettre du code, vérifiez le respect de cette liste de contrôle :

| Point de contrôle | Exigence | Référence |
| :--- | :--- | :--- |
| **Droit d'auteur de la Couronne** | En-tête textuel de 20 lignes aux lignes 1 à 20 de chaque fichier `.py` | [Section 1](#1-droit-dauteur-de-la-couronne-et-conformité-légale) |
| **Imports** | 3 catégories, en-têtes en minuscules, aucune ligne vide entre catégories, pas d'alias `np`/`pd` | [Section 2](#organisation-des-imports) |
| **Longueur de ligne** | Code $\le$ 80 caractères ; docstrings et commentaires $\le$ 72 caractères | [Section 2](#limites-de-longueur-de-ligne) |
| **Espacement vertical** | 2 lignes vides au niveau racine ; 1 entre méthodes ; 0 après docstrings | [Section 2](#espacement-vertical-et-lignes-vides) |
| **Délimiteurs de section** | `# ----- `[Symbol]` [action]`, 2 lignes vides avant, 0 après | [Section 2](#délimiteurs-de-section-et-ordonnancement-des-scripts) |
| **Commentaires** | Débutent par une minuscule ou `` `Symbol` ``, 1 espace après `#` | [Section 2](#formatage-des-commentaires) |
| **Docstrings** | Docstring de module avec `Public APIs:`; docstrings de fonctions au style Google | [Section 3](#3-spécifications-de-documentation-et-docstrings) |
| **Protocoles de typage** | Compacts, sans ligne vide, stubs `@property`, pragma pylint en début de fichier | [Section 4](#protocoles-de-typage) |
| **Pylint** | Pas de course au score ni d'abus de pragmas ; architecture saine d'abord | [Section 4](#philosophie-pylint-et-qualité-du-code) |
| **Frontière de test** | Tests unitaires rapides dans `tests/unit/` ; pipelines bout en bout dans `tests/integration/` | [Section 5](#frontière-tests-unitaires-vs-intégration) |
| **Fixtures** | Fixtures partagées dans `conftest.py` ; utilitaires de test dans fonctions `_helper` | [Section 5](#portée-et-emplacement-des-fixtures) |
| **Gestion des chemins** | Utilisation systématique de `pathlib.Path` plutôt que des chaînes brutes | [Section 5](#contrats-pathlib) |
| **Espace temporaire** | Scripts exploratoires/débogage placés dans le répertoire ignoré `sandbox/` | [Section 6](#espace-de-travail-temporaire-et-danalyse-sandbox) |
| **Hygiène des diffs** | Modifications strictement ciblées ; pas de reformatages massifs non sollicités | [Section 6](#hygiène-des-diffs-et-modifications-ciblées) |
| **Message de commit** | En-tête sur une ligne, puces `* ` en minuscules, symboles entre apostrophes inversées | [Section 6](#conventions-de-messages-de-validation-git) |
| **Description de PR** | Chemins relatifs au dépôt, aucune URI locale `file:///`, corps structuré | [Section 6](#descriptions-de-requêtes-de-tirage-pull-requests---pr) |
