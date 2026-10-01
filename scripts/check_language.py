"""Vérifie que rien de français ne part dans l'interface.

LA REGLE.

Le code et sa documentation sont en français, pour l'auteur. Tout ce qui
s'affiche est en anglais, parce que le jury Space Apps est international et
que le dépôt est public. Les deux cohabitent dans les mêmes fichiers, ce qui
rend la règle facile à enfreindre sans s'en apercevoir : une étiquette de
commande oubliée, un nom de colonne, un message d'erreur.

CE QUI EST CONTROLE.

Les chaînes littérales des fichiers qui alimentent la page, docstrings mises à
part puisque ce sont des commentaires. On y cherche des mots-outils français,
qui ne peuvent pas apparaître par hasard dans une phrase anglaise.

    py scripts/check_language.py

Ce contrôle est grossier par construction. Il attrape la phrase oubliée, pas
la traduction maladroite ; c'est déjà ce qui manquait, et un contrôle exact
coûterait plus cher qu'il ne rapporte.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Les fichiers dont les litteraux partent a l'ecran.
WATCHED = [
    ROOT / "app" / "main.py",
    ROOT / "flame" / "dashboard" / "catalogue.py",
    ROOT / "flame" / "dashboard" / "coverage.py",
    ROOT / "flame" / "dashboard" / "findings.py",
    ROOT / "flame" / "dashboard" / "proximity.py",
    ROOT / "flame" / "dashboard" / "questions.py",
    ROOT / "flame" / "dashboard" / "registry.py",
]

# Les chargeurs sont un cas a part. Leurs constantes de niveau module partent
# dans les tables que le tableau de bord affiche, donc elles comptent ; mais
# leur `main()` imprime un compte rendu en francais a l'intention de l'auteur,
# et leurs exceptions s'adressent au developpeur. Ni l'un ni l'autre n'atteint
# l'ecran. On ne regarde donc que le niveau module, et toute chaine qui part a
# l'affichage doit y etre nommee.
MODULE_LEVEL_ONLY = [
    ROOT / "flame" / "loaders" / "psi115.py",
]

# Les gabarits de vue, ou tout le texte visible est du HTML.
TEMPLATES = sorted((ROOT / "flame" / "viz" / "templates").glob("*.html"))

# Mots-outils francais. Deux exclusions volontaires : « plus » et « mode »
# sont des mots anglais courants, et les retenir noyait les vrais cas sous des
# faux positifs. Le mot doit etre isole, pas colle par un trait d'union, sinon
# « sans-serif » declenche a chaque feuille de style.
FRENCH = re.compile(
    r"(?<![-\w])("
    r"les|des|une|est|sont|pour|avec|dans|cette|chaque|aucun|aucune|"
    r"essai|essais|carburant|flamme|flammes|goutte|gouttelette|donnees|"
    r"pression|resultat|resultats|mesure|mesures|nous|qui|que|sur|par|"
    r"tres|jamais|toujours|seulement|depuis|entre|vers|selon|ainsi|"
    r"lorsque|parce|donc|mais|puis|alors|comme|apres|avant|sans|sous|"
    # Le vocabulaire du domaine. Les mots-outils seuls laissaient passer
    # « brule en microgravite », qui n'en contient aucun.
    r"brule|brulee|brulees|brulent|bruleur|eteint|eteinte|eteindre|"
    r"microgravite|gravite|fumee|suie|melange|melanges|echantillon|"
    r"echantillons|materiau|materiaux|valeur|valeurs|essayee|essayees|"
    r"couverture|vitesse|diametre|duree|largeur|longueur|epaisseur"
    r")(?![-\w])",
    re.IGNORECASE,
)

# Les commentaires HTML portent nos notes, comme les docstrings en Python.
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Le <script> d'un gabarit melange code et texte ; ses commentaires sont
# francais par regle, donc on les retire avant de chercher.
JS_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
# Un attribut CSS ou une propriete ne sont pas du texte.
CSS_BLOCK = re.compile(r"<style>.*?</style>", re.DOTALL | re.IGNORECASE)


def python_strings(path: Path, module_level_only: bool = False) -> list[tuple[int, str]]:
    """Les littéraux d'un module, docstrings exclues.

    `module_level_only` ne regarde que les affectations de niveau module, en
    sautant le corps des fonctions et des classes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    if module_level_only:
        roots = [
            node
            for node in tree.body
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ]
    else:
        roots = list(tree.body)

    docstrings = set()
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    docstrings.add(id(first.value))
    # La docstring du module elle-meme n'est pas dans `roots` quand on filtre,
    # mais elle y est sinon.
    first = tree.body[0] if tree.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        docstrings.add(id(first.value))

    found = []
    for root in roots:
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and len(node.value) > 8
            ):
                found.append((node.lineno, node.value))
    return found


def template_text(path: Path) -> list[tuple[int, str]]:
    """Le texte d'un gabarit, commentaires et CSS retirés."""
    raw = path.read_text(encoding="utf-8")
    stripped = CSS_BLOCK.sub(
        lambda m: "\n" * m.group(0).count("\n"), raw
    )
    stripped = HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), stripped)
    stripped = JS_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), stripped)
    return [
        (number, line)
        for number, line in enumerate(stripped.splitlines(), start=1)
        if line.strip()
    ]


def main() -> int:
    problems = 0

    for path, module_only in [(p, False) for p in WATCHED] + [
        (p, True) for p in MODULE_LEVEL_ONLY
    ]:
        for line, text in python_strings(path, module_level_only=module_only):
            match = FRENCH.search(text)
            if match:
                problems += 1
                short = " ".join(text.split())[:78]
                print(f"{path.relative_to(ROOT)}:{line}  «{match.group(0)}»  {short}")

    for path in TEMPLATES:
        for line, text in template_text(path):
            match = FRENCH.search(text)
            if match:
                problems += 1
                short = " ".join(text.split())[:78]
                print(f"{path.relative_to(ROOT)}:{line}  «{match.group(0)}»  {short}")

    print()
    if problems:
        print(f"{problems} chaine(s) francaise(s) dans l'interface.")
        return 1
    print("Interface entierement en anglais.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
