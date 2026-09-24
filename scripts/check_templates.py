"""Vérifie la syntaxe JavaScript des gabarits de vue.

POURQUOI CE SCRIPT EXISTE.

`cool_flame_live.html` a porté pendant plusieurs jours une erreur de syntaxe
silencieuse. Un heredoc de shell avait doublé un antislash, si bien que le
fichier contenait :

    + '+0.50) : les constituants s\\'evaporent a des rythmes differents, et '

En JavaScript, ce `\\` est un antislash échappé, donc l'apostrophe qui suit
FERME la chaîne et `evaporent` devient un identifiant. Une erreur de syntaxe
dans un module tue le module entier : la vue des flammes froides ne
s'affichait pas du tout, et rien, ni côté Python ni dans Streamlit, ne le
signalait.

Python vérifie sa propre syntaxe à l'import ; le JavaScript embarqué dans un
gabarit HTML n'est vérifié par personne. Ce script comble ce trou. Il extrait
le `<script type="module">` de chaque gabarit et le passe à `node --check`.

    py scripts/check_templates.py

Node n'est pas une dépendance du projet : s'il est absent, le script le dit et
sort proprement plutôt que d'échouer.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "flame" / "viz" / "templates"

# On ne verifie que le module ; les <script type="application/json"> ne sont
# pas du code, et le gabarit y porte un marqueur de substitution.
MODULE = re.compile(
    r'<script type="module">(.*?)</script>', re.DOTALL | re.IGNORECASE
)

# `import ... from 'three'` n'est resolvable que par la carte d'import du
# navigateur. `node --check` analyse sans resoudre, donc l'import passe ; on le
# neutralise quand meme pour que l'intention reste lisible.
BARE_IMPORT = re.compile(r"^\s*import\s+.*?from\s+'three';\s*$", re.MULTILINE)


def check(path: Path, node: str) -> list[str]:
    """Retourne les erreurs trouvées dans un gabarit, vide si tout va bien."""
    html = path.read_text(encoding="utf-8")
    blocks = MODULE.findall(html)
    if not blocks:
        return []

    problems: list[str] = []
    for number, source in enumerate(blocks, start=1):
        source = BARE_IMPORT.sub("const THREE = {};", source)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".mjs", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(source)
            temporary = Path(handle.name)
        try:
            result = subprocess.run(
                [node, "--check", str(temporary)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                problems.append(f"bloc {number} : " + " | ".join(detail[:4]))
        finally:
            temporary.unlink(missing_ok=True)
    return problems


def main() -> int:
    node = shutil.which("node")
    if node is None:
        print("node introuvable, verification sautee.")
        return 0

    failures = 0
    for path in sorted(TEMPLATES.glob("*.html")):
        problems = check(path, node)
        if problems:
            failures += 1
            print(f"ECHEC  {path.name}")
            for line in problems:
                print(f"       {line}")
        else:
            print(f"ok     {path.name}")

    print()
    if failures:
        print(f"{failures} gabarit(s) en erreur.")
        return 1
    print("Tous les gabarits sont syntaxiquement valides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
