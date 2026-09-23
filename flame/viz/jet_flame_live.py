"""La flamme de jet de PSI-107 — trois dimensions mesurees, un profil suppose.

CE QUI EST MESURE, ET CE QUI NE L'EST PAS.

PSI-107 releve trois grandeurs geometriques par essai :

    nozzle_mm         le diametre de la buse, donc la base de la flamme
    flame_width_mm    sa largeur MAXIMALE
    smoke_point_mm    sa longueur au point de fumee

Aucune colonne ne dit OU la flamme atteint sa largeur maximale le long de son
axe. La silhouette dessinee entre ces trois points est donc une CONVENTION :
celle, connue, d'une flamme de diffusion laminaire, qui s'evase depuis la buse
puis se referme en pointe.

C'est dit dans l'interface, et ce n'est pas un detail : une silhouette
plausible presentee comme une mesure serait exactement le genre de survente
que ce projet evite partout ailleurs.

CE QUI SE COMPARE VALABLEMENT, ce sont les TAILLES, et elles sont mesurees.
Une flamme courte signifie un carburant qui fume beaucoup : le propylene
lache a 27 mm en moyenne contre 71 pour le propane. C'est ce qu'on attend d'un
alcene face a un alcane, et c'est le resultat du module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flame.loaders.psi107 import load

TEMPLATE = Path(__file__).parent / "templates" / "jet_flame_live.html"


def payload() -> dict:
    df = load()
    usable = df[
        df[["smoke_point_mm", "flame_width_mm", "nozzle_mm", "fuel"]]
        .notna()
        .all(axis=1)
    ]
    tests = [
        {
            "fuel": row["fuel"],
            "fraction": round(float(row["fuel_fraction"]), 2)
            if pd.notna(row["fuel_fraction"])
            else 1.0,
            "nozzle": round(float(row["nozzle_mm"]), 3),
            "width": round(float(row["flame_width_mm"]), 3),
            "length": round(float(row["smoke_point_mm"]), 2),
            "coflow": round(float(row["coflow_velocity_cm_s"]), 2)
            if pd.notna(row["coflow_velocity_cm_s"])
            else 0.0,
        }
        for _, row in usable.sort_values(["fuel", "smoke_point_mm"]).iterrows()
    ]
    return {"tests": tests}


def page() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*__PAYLOAD__*/", json.dumps(payload(), separators=(",", ":")))


def main() -> None:
    data = payload()
    frame = pd.DataFrame(data["tests"])
    print(f"{len(frame)} essais avec les trois dimensions\n")
    print(
        frame.groupby("fuel")
        .agg(essais=("length", "size"), longueur=("length", "mean"),
             largeur=("width", "mean"), buse=("nozzle", "mean"))
        .round(2)
        .to_string()
    )
    ratio = frame["length"] / frame["width"]
    print(f"\n  rapport longueur/largeur : {ratio.min():.1f} a {ratio.max():.1f}")
    print("  le PROFIL n'est pas mesure : la silhouette est une convention")

    destination = Path("data/figures/psi107_jet_flame3d.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page(), encoding="utf-8")
    print(f"\n  {destination}  ({destination.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
