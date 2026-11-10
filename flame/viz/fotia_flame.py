"""La flamme de Fotia — une page autonome qui anime une `Prediction`.

Tout le calcul vit dans `flame/models/hybrid.py`. La page ne recoit que le
resultat : diametres, K, duree, badge, liste des grandeurs extrapolees.

Ce qui est a l'echelle : la gouttelette (loi en d2). Ce qui ne l'est pas :
l'enveloppe de flamme. Son rapport au diametre et son allongement avec la
gravite sont des choix de dessin, pas des mesures ; la page l'ecrit.
"""

from __future__ import annotations

import json
from pathlib import Path

from flame.models.hybrid import LABELS, Prediction

TEMPLATE = Path(__file__).parent / "templates" / "fotia_flame.html"

# Choix de dessin, assumes et affiches comme tels.
ENVELOPE_RATIO = 4.0      # rayon de l'enveloppe / diametre de la goutte
MAX_STRETCH = 2.2         # allongement a 1 g

BADGES = {
    "inside": ("Measured here", "#35b37e"),
    "extrapolation": ("Between tests", "#e0a327"),
    "outside": ("Never tested", "#d9534f"),
}


def payload(prediction: Prediction, d0_mm: float, gravity_g: float) -> dict:
    label, colour = BADGES[prediction.verdict]
    final = (
        prediction.dext_mm
        if prediction.extinguishes and prediction.dext_mm is not None
        else 0.0
    )
    final = min(max(final, 0.0), d0_mm)
    return {
        "d0": round(d0_mm, 4),
        "dfinal": round(final, 4),
        "k": round(prediction.k_mm2_s, 5),
        "duration": round(max(prediction.duration_s, 0.1), 3),
        "envelope": ENVELOPE_RATIO,
        "stretch": round(1 + (MAX_STRETCH - 1) * min(max(gravity_g, 0.0), 1.0) ** 0.5, 3),
        "badge": {"label": label, "colour": colour},
        "extrapolated": [LABELS[name] for name in prediction.extrapolated],
    }


def page(prediction: Prediction, d0_mm: float, gravity_g: float) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload(prediction, d0_mm, gravity_g), separators=(",", ":"))
    return html.replace("/*__PAYLOAD__*/", data)
