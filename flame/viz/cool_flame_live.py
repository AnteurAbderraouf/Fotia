"""Les trois coquilles de PSI-39 — et pourquoi une seule s'anime.

CE QUE LA VUE MONTRE.

Une flamme de gouttelette en microgravité est sphérique, donc trois diamètres
mesurés donnent trois coquilles emboîtées exactes :

    bleu     la gouttelette au depart
    orange   le diametre auquel la flamme CHAUDE s'est eteinte
    violet   le diametre auquel la flamme FROIDE s'est eteinte

C'est la question du module rendue visible : entre la coquille orange et la
violette, il ne se passe rien de visible à l'œil, et pourtant la gouttelette
continue de brûler.

POURQUOI L'ANIMATION EST RÉSERVÉE AU DODÉCANE PUR.

Sur PSI-69, la loi en d² reproduit les durées mesurées avec une corrélation de
0.995, ce qui autorise à animer la trajectoire. Sur PSI-39, la même
vérification donne 0.400, et le détail explique pourquoi :

    n-dodecane (corps PUR)        correlation +0.889   62 % a 20 % pres
    dodecane75 / iso-dodecane25   +0.501               19 %
    dodecane60 / iso-dodecane40   +0.026               33 %

La loi tient sur le corps pur et s'effondre à mesure que la part de mélange
augmente. C'est de la combustion multi-composants : les constituants
s'évaporent à des rythmes différents, et une constante unique ne décrit plus
rien. Sept des huit gouttelettes dont le diamètre d'extinction DÉPASSE le
diamètre initial sont d'ailleurs des mélanges — elles ont gonflé avant de
brûler, par chauffage interne.

L'animation n'est donc proposée que là où la loi a été vérifiée. Sur les
mélanges, les trois coquilles restent affichées, exactes et mesurées ; seule
la trajectoire entre elles serait inventée, et l'interface le dit.

DEUX PHASES, PAS UNE. Là où l'animation joue, elle enchaîne deux régimes : la
gouttelette se consume à k_hot jusqu'à l'extinction de la flamme chaude, puis
à k_cool, plus lent, sous la flamme froide. Les deux constantes sont mesurées
essai par essai.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flame.loaders.psi39 import load

TEMPLATE = Path(__file__).parent / "templates" / "cool_flame_live.html"

# Carburants ou la loi en d2 a ete verifiee comme acceptable.
ANIMATABLE = ["n-dodecane"]


def payload() -> dict:
    df = load()
    usable = df[
        df[["d0_mm", "dext_hot_mm", "k_hot_mm2_s", "fuel", "pressure_atm"]]
        .notna()
        .all(axis=1)
    ].copy()
    usable["dext_cool_mm"] = usable["dext_cool_mm"].fillna(0.0)
    usable["k_cool_mm2_s"] = usable["k_cool_mm2_s"].fillna(0.0)

    tests = [
        {
            "fuel": row["fuel"],
            "d0": round(float(row["d0_mm"]), 3),
            "dh": round(float(row["dext_hot_mm"]), 3),
            "dc": round(float(row["dext_cool_mm"]), 3),
            "kh": round(float(row["k_hot_mm2_s"]), 4),
            "kc": round(float(row["k_cool_mm2_s"]), 4),
            "p": round(float(row["pressure_atm"]), 3),
        }
        for _, row in usable.sort_values(["fuel", "pressure_atm"]).iterrows()
    ]
    return {"tests": tests, "animatable": ANIMATABLE}


def page() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*__PAYLOAD__*/", json.dumps(payload(), separators=(",", ":")))


def main() -> None:
    data = payload()
    frame = pd.DataFrame(data["tests"])
    print(f"{len(frame)} essais exploitables\n")
    print(frame.groupby("fuel").agg(
        essais=("d0", "size"),
        avec_flamme_froide=("dc", lambda s: int((s > 0).sum())),
        gonflement=("dh", "size"),
    ).to_string())
    swollen = frame[frame["dh"] > frame["d0"]]
    print(f"\n  gouttelettes gonflees (dext_hot > d0) : {len(swollen)}")
    print(f"  animables (loi en d2 verifiee)         : "
          f"{int(frame['fuel'].isin(ANIMATABLE).sum())}")

    destination = Path("data/figures/psi39_cool_flame3d.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page(), encoding="utf-8")
    print(f"\n  {destination}  ({destination.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
