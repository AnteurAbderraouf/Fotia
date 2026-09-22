"""Panneau interactif autonome — curseurs et modeles dans la meme page.

Streamlit relance tout le script a chaque mouvement de curseur, ce qui recree
l'iframe du composant 3D : blanc, puis saut. Les trois modeles du module
Suppression etant lineaires, ils tiennent en un produit scalaire et partent
dans le navigateur. Les curseurs y vivent aussi, plus rien ne repasse par le
serveur, et la gouttelette glisse d'un etat a l'autre.

Voir flame/dashboard/export.py pour le contenu exact envoye.
"""

from __future__ import annotations

from pathlib import Path

from flame.dashboard.export import as_json
from flame.dashboard.predict import Bundle

TEMPLATE = Path(__file__).parent / "templates" / "droplet_live.html"


def page(bundle: Bundle) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*__PAYLOAD__*/", as_json(bundle))
