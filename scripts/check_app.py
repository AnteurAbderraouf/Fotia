"""Exécute le tableau de bord hors navigateur et signale toute exception.

POURQUOI.

Une erreur dans `app/main.py` ne se voit qu'en ouvrant la page, en cliquant
sur le bon niveau et en choisissant le bon module. Trois niveaux, six
questions et cinq onglets font beaucoup de chemins à parcourir à la main, et
le chemin qu'on oublie est justement celui qui casse.

`AppTest` de Streamlit rejoue le script comme le ferait le navigateur, sans
navigateur. On visite ici CHAQUE niveau, CHAQUE question et CHAQUE onglet, et
on échoue au premier tracé d'exception.

    py scripts/check_app.py

Ce n'est pas un test unitaire : il ne vérifie pas ce qui est affiché, il
vérifie que la page se rend sans lever. C'est déjà ce qui manquait.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from flame.dashboard.questions import QUESTIONS  # noqa: E402

APP = ROOT / "app" / "main.py"

# Les modeles s'entrainent au premier rendu ; le defaut de 3 s ne suffit pas.
TIMEOUT = 240


def run(label: str, **state) -> bool:
    """Rend la page dans un état donné et rapporte les exceptions."""
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    if app.exception:
        print(f"ECHEC  {label}")
        for problem in app.exception:
            head = (problem.message or "").strip().splitlines()
            print(f"       {problem.type}: {head[0] if head else ''}")
            for line in (problem.stack_trace or [])[-4:]:
                print(f"       {line.rstrip()}")
        return False
    print(f"ok     {label}")
    return True


def main() -> int:
    failures = 0

    # Niveau 1 : les quatre constats.
    failures += not run("01 findings", level="01")

    # Niveau 2 : chaque question, y compris le module descriptif qui ne passe
    # pas par le meme chemin de code que les autres.
    for key in QUESTIONS:
        failures += not run(f"02 {key}", level="02", question=key)

    # Le module descriptif visite aussi le cas « issues explicites seules ».
    failures += not run("02 materials, stated only", level="02", question="materials")

    # Niveau 3 : les cinq onglets sont rendus ensemble par st.tabs, donc un
    # seul passage les couvre tous.
    failures += not run("03 verification", level="03")

    print()
    if failures:
        print(f"{failures} etat(s) en erreur.")
        return 1
    print("Tous les etats du tableau de bord se rendent sans exception.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
