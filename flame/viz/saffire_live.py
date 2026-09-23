"""Les bandes de Saffire-II, brûlées en orbite et au sol.

CE QUE LA VUE MONTRE, ET POURQUOI ELLE EST EXACTE.

Chaque échantillon de PSI-99 est une bande de 29 cm sur 5. NASA a relevé la
longueur réellement brûlée dans chaque régime de gravité. La portion sombre
est donc une mesure, pas une estimation : deux bandes identiques, côte à côte,
avec ce qui a brûlé de chaque côté.

C'est le contraste du projet rendu immédiat. Le silicone ne brûle PAS en
microgravité — quatre échantillons sur quatre à zéro — alors qu'à 1 g deux
d'entre eux brûlent entièrement.

« COMPLETE » EST TRADUIT EN LONGUEUR TOTALE. NASA écrit « Complete » plutôt
qu'un nombre quand l'échantillon a brûlé entièrement. La portion brûlée vaut
alors toute la bande, ce qui est la lecture littérale du mot. Le relevé NASA
d'origine reste affiché à côté pour qu'on puisse vérifier.

L'ÉPAISSEUR EST EXAGÉRÉE À L'ÉCRAN. Les bandes font entre 0.27 et 10 mm
d'épaisseur pour 290 mm de long : à l'échelle, elles seraient invisibles. Le
facteur appliqué est constant et ne sert qu'à rendre l'objet visible ; les
longueurs et largeurs, elles, sont à l'échelle.

NEUF ÉCHANTILLONS. Cette vue ne se modélise pas et ne se généralise pas. Elle
montre neuf essais réels, et c'est tout ce qu'elle prétend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flame.loaders.psi99 import load

TEMPLATE = Path(__file__).parent / "templates" / "saffire_live.html"
DEFAULT_LENGTH_CM = 29.0


def _burn_length(row: pd.Series, prefix: str, total: float) -> float:
    """Longueur brûlée en cm, « Complete » valant la bande entière."""
    measured = row.get(f"{prefix}_burn_length_cm")
    if pd.notna(measured):
        return round(float(measured), 2)
    qualitative = str(row.get(f"{prefix}_burn_length_qualitative") or "").lower()
    if "complete" in qualitative:
        return total
    return 0.0


def payload() -> dict:
    df = load()
    samples = []
    for _, row in df.iterrows():
        length = float(row["length_cm"]) if pd.notna(row["length_cm"]) else DEFAULT_LENGTH_CM
        samples.append(
            {
                "id": row["sample_id"],
                "material": row["material"],
                "length": round(length, 1),
                "width": round(float(row["width_cm"]), 1)
                if pd.notna(row["width_cm"])
                else 5.0,
                "thickness": round(float(row["thickness_mm"]), 2)
                if pd.notna(row["thickness_mm"])
                else 0,
                "ugBurn": _burn_length(row, "ug", length),
                "g1Burn": _burn_length(row, "g1", length),
                "ugRaw": row["ug_burn_length_raw"],
                "g1Raw": row["g1_burn_length_raw"],
                "ugRate": row["ug_spread_rate_raw"]
                if pd.notna(row["ug_spread_rate_raw"])
                else "",
                "g1Rate": row["g1_spread_rate_raw"]
                if pd.notna(row["g1_spread_rate_raw"])
                else "",
            }
        )
    return {"samples": samples}


def page() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*__PAYLOAD__*/", json.dumps(payload(), separators=(",", ":")))


def main() -> None:
    data = payload()
    frame = pd.DataFrame(data["samples"])
    print(f"{len(frame)} echantillons\n")
    view = frame[["id", "material", "length", "ugBurn", "g1Burn", "ugRaw", "g1Raw"]]
    view.columns = ["ech.", "materiau", "bande cm", "brule ug", "brule 1g",
                    "releve ug", "releve 1g"]
    print(view.to_string(index=False))

    contrast = frame[(frame["ugBurn"] == 0) & (frame["g1Burn"] > 0)]
    print(
        f"\n  {len(contrast)} echantillons ne brulent PAS en microgravite mais "
        f"brulent a 1 g :\n  {', '.join(contrast['id'])}"
    )

    destination = Path("data/figures/psi99_saffire3d.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page(), encoding="utf-8")
    print(f"\n  {destination}  ({destination.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
