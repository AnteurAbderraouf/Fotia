"""Reconstruit toutes les tables nettoyées, de zéro.

    python scripts/build_all.py

`data/processed/` est entièrement jetable : ce script le régénère à
l'identique depuis `combustion_science/`, qui n'est jamais modifié. Si un
loader échoue, le script continue et signale l'échec à la fin plutôt que de
s'arrêter au premier problème — on veut savoir combien de tables sont
touchées, pas seulement la première.
"""

from __future__ import annotations

import importlib
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOADERS = [
    ("psi69", "Suppression — gouttelettes, CO2 et helium"),
    ("psi39", "Flammes froides — gouttelettes ISS"),
    ("psi117", "Flammes froides — donnees derriere figures publiees"),
    ("psi159", "Sustainment — brûleur a gaz, normal et inverse"),
    ("psi101", "Detection — fumee et detecteurs ISS"),
    ("psi107", "Suie — point de fumee"),
    ("psi25", "Materiaux solides — catalogue BASS-II"),
    ("psi99", "Microgravite contre Terre — carte d'affichage"),
    ("psi142", "Reference au sol (1g) — limites d'extinction"),
    ("psi115", "Champs CFD — gravite contre microgravite"),
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    produced: list[str] = []

    for name, description in LOADERS:
        print(f"\n{'─' * 70}\n{name:8} {description}\n{'─' * 70}")
        try:
            module = importlib.import_module(f"flame.loaders.{name}")
            captured = io.StringIO()
            with redirect_stdout(captured):
                module.main()
            output = captured.getvalue()
            for line in output.splitlines():
                if ".csv" in line:
                    print("  " + line.strip())
                    produced.append(line.strip().split()[0])
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  ECHEC — voir le detail en fin de rapport")

    print(f"\n{'═' * 70}")
    print(f"{len(LOADERS) - len(failures)}/{len(LOADERS)} loaders ont abouti")
    print(f"{len(produced)} fichiers ecrits dans data/processed/")
    print(f"{'═' * 70}")

    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
