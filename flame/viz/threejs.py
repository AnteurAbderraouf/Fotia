"""Rendu Three.js — le panache en 3D, animé sans reconstruire la scène.

POURQUOI QUITTER PLOTLY POUR CETTE VUE.

Plotly rend très bien une scène 3D fixe qu'on manipule à la souris, mais son
système d'animation impose `redraw` en 3D : chaque image reconstruit le graphe
de scène en JavaScript, même quand rien ne change. Mesuré sur ce projet, une
rotation de caméra de 48 images ne contenant AUCUNE donnée restait lente.

Three.js travaille à l'envers. La géométrie est envoyée une fois sur le GPU,
et chaque image ne met à jour qu'une matrice de caméra ou un tampon déjà en
mémoire. C'est ce que fait un moteur de jeu, et c'est fluide sur une carte
intégrée.

CE QUI EST EXPORTÉ, ET POURQUOI C'EST SI PETIT.

  * Les contours, et non les surfaces. Une isosurface de révolution est
    entièrement décrite par sa courbe génératrice — 110 couples (x, rayon).
    La révolution se fait dans le navigateur, où elle est gratuite. On exporte
    48 niveaux pour que le balayage soit continu, ce qui fait environ 60 Ko
    contre 3,4 Mo pour la version plotly.

  * Le plan de coupe devient une TEXTURE, pas un maillage. Plotly le rendait
    avec 14 400 sommets ; ici c'est une image plaquée sur quatre sommets. Le
    GPU ne voit presque plus rien à dessiner.

    Le champ est rééchantillonné sur une grille régulière avant d'en faire une
    image. C'est nécessaire : le maillage NASA est étiré d'un facteur 3.8 en x
    et 1.5 en y, et plaquer une texture sans corriger cet étirement
    déformerait le panache.

TOUT EST EMBARQUÉ DANS LA PAGE. Les données sont inscrites dans le HTML et la
texture en URI base64, parce qu'un `fetch` depuis un fichier local est bloqué
par la politique d'origine des navigateurs. La page s'ouvre donc d'un
double-clic, sans serveur.

CE QUE LA PAGE MONTRE, ET NE MONTRE PAS. Même règle que partout ailleurs :
le champ est STATIONNAIRE. Le balayage fait varier le seuil de découpe, la
rotation déplace l'observateur. Aucune des deux n'est une combustion filmée,
PSI-115 ne contient pas de temps.
"""

from __future__ import annotations

import base64
import io as _io
import json
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.colors import Normalize
from PIL import Image

from flame.loaders.psi115 import load_mesh
from flame.viz.plume3d import _check_axisymmetry, _isocontour, _load_field

LEVELS = 48
CONTOUR_POINTS = 110
# Resolution de la texture du plan de coupe. Le champ fait 960x240 ; a cette
# taille l'oeil ne distingue plus la difference, et le poids est divise par 4.
TEXTURE_WIDTH = 640
TEXTURE_HEIGHT = 160
LOWEST_LEVEL_FRACTION = 0.025
DECIMALS = 4

TEMPLATE = Path(__file__).parent / "templates" / "plume_threejs.html"


def _resample(curve: np.ndarray, count: int) -> np.ndarray:
    """Ramène une courbe à un nombre fixe de points, ordonnés selon l'axe.

    Le nombre doit être le même pour tous les niveaux : le balayage passe d'un
    contour au suivant, et deux courbes de longueurs différentes ne se
    correspondraient pas point à point.
    """
    order = np.argsort(curve[:, 0])
    ordered = curve[order]
    target = np.linspace(ordered[0, 0], ordered[-1, 0], count)
    radii = np.interp(target, ordered[:, 0], np.abs(ordered[:, 1]))
    return np.column_stack([target, radii])


def build_contours(case: str, variable: str) -> dict:
    """Les contours de révolution, un par niveau de seuil."""
    axis_x, axis_y = load_mesh()
    field = _load_field(case, variable)
    _check_axisymmetry(field, case)

    finite = field[np.isfinite(field)]
    ceiling = float(np.percentile(finite, 99.9))
    floor = LOWEST_LEVEL_FRACTION * float(np.nanmax(finite))

    levels, curves = [], []
    for level in np.linspace(ceiling, floor, LEVELS):
        paths = _isocontour(field, axis_x, axis_y, float(level))
        if not paths:
            continue
        curve = _resample(max(paths, key=len), CONTOUR_POINTS)
        levels.append(round(float(level), 3))
        curves.append(np.round(curve, DECIMALS).tolist())

    if not curves:
        raise ValueError(f"{case} : aucun contour sur la plage de seuils")

    return {
        "levels": levels,
        "curves": curves,
        "points": CONTOUR_POINTS,
        "xRange": [float(axis_x.min()), float(axis_x.max())],
        "yRange": [float(axis_y.min()), float(axis_y.max())],
    }


def build_texture(case: str, variable: str, colormap: str = "magma") -> str:
    """Le champ en image, rééchantillonné sur une grille régulière.

    Le rééchantillonnage n'est pas cosmétique : le maillage NASA est étiré,
    et plaquer directement les valeurs sur un rectangle régulier écraserait
    le panache là où les mailles sont fines et l'étirerait ailleurs.
    """
    axis_x, axis_y = load_mesh()
    field = _load_field(case, variable)

    uniform_x = np.linspace(axis_x.min(), axis_x.max(), TEXTURE_WIDTH)
    uniform_y = np.linspace(axis_y.min(), axis_y.max(), TEXTURE_HEIGHT)

    # Interpolation separable : d'abord chaque ligne le long de x, puis
    # chaque colonne le long de y. Licite car la grille est separable.
    along_x = np.empty((field.shape[0], TEXTURE_WIDTH))
    for row in range(field.shape[0]):
        along_x[row] = np.interp(uniform_x, axis_x, field[row])
    regular = np.empty((TEXTURE_HEIGHT, TEXTURE_WIDTH))
    for column in range(TEXTURE_WIDTH):
        regular[:, column] = np.interp(uniform_y, axis_y, along_x[:, column])

    finite = regular[np.isfinite(regular)]
    normalise = Normalize(vmin=float(finite.min()), vmax=float(np.percentile(finite, 99.5)))
    # matplotlib 3.9 a retire cm.get_cmap au profit du registre colormaps.
    coloured = matplotlib.colormaps[colormap](normalise(np.nan_to_num(regular)))
    image = Image.fromarray((coloured[:, :, :3] * 255).astype(np.uint8))

    buffer = _io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_page(
    destination: Path,
    variable: str = "Smoke",
    burner_size_cm: int = 8,
) -> Path:
    """Assemble la page autonome : données inscrites, texture embarquée."""
    micro_case = f"{burner_size_cm}cmNoGravity"
    earth_case = f"{burner_size_cm}cmGravity"

    payload = {
        "variable": variable,
        "burner": burner_size_cm,
        "microgravity": build_contours(micro_case, variable),
        "textures": {
            "microgravity": build_texture(micro_case, variable),
            "earth": build_texture(earth_case, variable),
        },
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__PAYLOAD__*/", json.dumps(payload, separators=(",", ":")))
    destination.write_text(html, encoding="utf-8")
    return destination


def main() -> None:
    output = Path("data/figures")
    output.mkdir(parents=True, exist_ok=True)

    destination = write_page(output / "psi115_threejs.html")
    size_kb = destination.stat().st_size / 1024
    print(f"  {destination}  ({size_kb:.0f} Ko)")

    contours = build_contours("8cmNoGravity", "Smoke")
    print(
        f"\n  {len(contours['levels'])} niveaux de seuil exportes, "
        f"{CONTOUR_POINTS} points chacun"
        f"\n  seuils de {contours['levels'][0]:.2f} a {contours['levels'][-1]:.2f}"
        f"\n  texture du plan : {TEXTURE_WIDTH}x{TEXTURE_HEIGHT}, rééchantillonnée"
        " sur grille reguliere"
    )
    print(
        f"\n  a comparer aux 3382 Ko de la version plotly, pour un balayage"
        f"\n  de {len(contours['levels'])} niveaux au lieu de 14."
    )


if __name__ == "__main__":
    main()
