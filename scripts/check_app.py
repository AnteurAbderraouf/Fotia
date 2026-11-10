"""Exécute la console Fotia hors navigateur et signale toute erreur.

POURQUOI.

Une erreur ne se voit qu'en ouvrant la page avec le bon réglage. `AppTest` de
Streamlit rejoue le script sans navigateur ; on le fait pour chaque lieu,
chaque carburant, chaque diluant et les cas limites, et on échoue au premier
tracé d'exception.

CE QU'ON NE PEUT PAS VOIR AINSI. La console est un composant
`st.components.v2` : AppTest ne l'exécute pas, il ne voit que les données
envoyées au navigateur. On vérifie donc ces DONNEES, qui sont tout ce qui
s'affiche :
  * la page se rend sans exception ;
  * aucun mot français dans les chaînes envoyées (les CSV de data/context sont
    rédigés en français, un champ qui passerait se verrait ici) ;
  * chaque prédiction porte l'un des trois badges, au vocabulaire fixe.
Le rendu visuel, lui, reste à regarder dans un navigateur.

    py scripts/check_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from flame.dashboard import console  # noqa: E402
from flame.dashboard.console import visible_presets  # noqa: E402
from flame.dashboard.why import untranslated  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from check_language import FRENCH  # noqa: E402

APP = ROOT / "app" / "main.py"
BADGES = {"Measured here", "Between tests", "Never tested"}

# Le modele s'entraine au premier rendu ; le defaut de 3 s ne suffit pas.
TIMEOUT = 240


def _strings(value, key: str = ""):
    """Toutes les chaînes d'une structure, sauf les URL (pas du texte lu)."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, k)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v, key)
    elif isinstance(value, str) and key not in {"url", "key", "fuel", "preset",
                                                 "diluent", "kind", "verdict"}:
        yield value


def check_payload(label: str, conditions: dict) -> bool:
    """Ce que la console recevrait pour ce réglage : français, badge."""
    data = console.payload(conditions)
    problems = []
    for text in _strings(data):
        match = FRENCH.search(text)
        if match:
            problems.append(f"«{match.group(0)}»  {' '.join(text.split())[:70]}")
    pred = data["result"]["prediction"]
    if pred is not None and pred["badge"] not in BADGES:
        problems.append(f"badge inconnu : {pred['badge']}")
    if pred is None and not data["result"]["error"]:
        problems.append("ni prediction ni message d'erreur")
    if problems:
        print(f"ECHEC  {label}")
        for line in problems[:8]:
            print(f"       {line}")
        return False
    return True


def run(label: str, **changes) -> bool:
    """Rend la page avec un réglage donné, puis vérifie ses données."""
    conditions = console.sanitize({**console.initial(), **changes})
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    app.session_state[console.STATE] = conditions
    app.run()
    if app.exception:
        print(f"ECHEC  {label}")
        for problem in app.exception:
            head = (problem.message or "").strip().splitlines()
            print(f"       {problem.type}: {head[0] if head else ''}")
            for line in (problem.stack_trace or [])[-4:]:
                print(f"       {line.rstrip()}")
        return False
    if not check_payload(label, conditions):
        return False
    print(f"ok     {label}")
    return True


def preset(key: str) -> dict:
    """Les valeurs d'un lieu, comme le navigateur les appliquerait."""
    p = next(x for x in console.presets_payload() if x["key"] == key)
    out = {"preset": key}
    for name in ("pressure_atm", "o2_frac", "gravity_g"):
        if p[name] is not None:
            out[name] = p[name]
    return out


def main() -> int:
    failures = 0

    # Chaque ligne de data/context doit avoir son texte anglais.
    gaps = untranslated()
    if gaps:
        print(f"ECHEC  why.py : sans texte anglais {gaps}")
        failures += 1
    else:
        print("ok     why.py : chaque ligne de contexte a son texte anglais")

    # Lieux masques (Lune, Mars, les deux Apollo) : jamais envoyes comme
    # preregle ; Lune et Mars non plus comme point de la carte.
    data = console.payload(console.initial())
    leaks = [p["key"] for p in data["presets"] if p["key"] in console.MASKED_PRESETS]
    leaks += [a["id"] for a in data["atmospheres"] if a["id"] in console.MASKED_ATMOSPHERES]
    if leaks:
        print(f"ECHEC  lieux masques : encore envoyes au navigateur {leaks}")
        failures += 1
    else:
        print("ok     lieux masques : rien n'est envoye au navigateur")

    failures += not run("premier ecran")
    for p in visible_presets():
        failures += not run(f"lieu {p.key}", **preset(p.key))
    # Hors de tout essai, regle a la main (les lieux Apollo sont masques,
    # mais 100 % d'O2 reste atteignable au curseur). Les trois carburants,
    # dont le n-decane qui n'a aucun essai et passe par un autre chemin du
    # moteur (pas d'extinction modelisee).
    pure_o2 = {"preset": "custom", "pressure_atm": 0.34, "o2_frac": 1.0, "gravity_g": 0.0}
    for fuel in ["Methanol", "Heptane", "n-decane"]:
        failures += not run(f"100 % O2 hors essais, {fuel}", fuel=fuel, **pure_o2)
    failures += not run("diluant co2", diluent="co2", diluent_frac=0.3)
    failures += not run("diluant he, methanol", fuel="Methanol", diluent="he",
                        diluent_frac=0.3)
    # Melange impossible : O2 + diluant > 100 %, la page ne doit pas planter.
    failures += not run("gaz > 100 %", diluent="he", diluent_frac=0.4, **pure_o2)
    failures += not run("goutte hors plage", d0_mm=5.8)
    failures += not run("pression hors carte", pressure_atm=4.5, preset="custom")
    # Ce que le navigateur pourrait envoyer de faux : ramene a des valeurs sures.
    failures += not run("valeurs invalides", fuel="kerosene", o2_frac="abc",
                        diluent="argon", d0_mm=float("nan"))

    print()
    if failures:
        print(f"{failures} etat(s) en erreur.")
        return 1
    print("Tous les etats de la console se rendent sans erreur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
