"""Composant animé de la gouttelette, pour intégration dans Streamlit.

La trajectoire affichée est la loi en d² — voir `flame/models/droplet_burn.py`
pour sa validation sur les durées mesurées (corrélation 0.995). Ce n'est pas
une interpolation entre deux bornes.

POURQUOI THREE.JS ICI ET PAS PLOTLY. Mesuré sur ce projet : en 3D, plotly
impose une reconstruction complète de la scène à chaque image, et même une
rotation de caméra sans données restait saccadée. Three.js met à jour une
matrice d'échelle sur une géométrie déjà en mémoire GPU. L'utilisateur a
demandé une vision continue, sans blancs ni coupures : c'est la condition
pour l'obtenir.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "templates" / "droplet_threejs.html"


def page(initial_mm: float, final_mm: float, rate_mm2_s: float) -> str:
    """Page autonome animant d(t) = sqrt(d0^2 - K t)."""
    from flame.models.droplet_burn import burn_duration

    duration = burn_duration(initial_mm, final_mm, rate_mm2_s)
    payload = {
        "d0": round(float(initial_mm), 4),
        "dext": round(float(final_mm), 4),
        "k": round(float(rate_mm2_s), 5),
        "duration": round(float(duration), 4) or 1.0,
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*__PAYLOAD__*/", json.dumps(payload, separators=(",", ":")))
