"""Animations — ce qui bouge, et ce qui ne bouge pas.

LA CONTRAINTE, POSÉE D'ABORD.

PSI-115 ne contient aucune série temporelle. Les cinq cas sont des solutions
STATIONNAIRES : un seul état d'équilibre par cas, pas une évolution. Il
n'existe donc rien à animer au sens d'un film de combustion, et une flamme qui
vacillerait à l'écran serait inventée de bout en bout.

Ce module anime donc des choses qui existent réellement :

    sweep    le SEUIL d'affichage balaie du cœur dense vers le bord du
             panache. Chaque image est une isosurface réelle du même champ
             immobile. On voit la structure interne, densité par densité.

    orbit    la CAMÉRA tourne autour d'une scène figée. Rien dans les données
             ne bouge, seul le point de vue change. C'est de la présentation
             assumée, pas de l'information nouvelle.

Dans les deux cas le titre dit lequel des deux est à l'écran, et rappelle que
le champ est stationnaire. Une animation est le format le plus facile à
confondre avec une mesure dynamique : autant l'écrire sur l'image.

POIDS. Une animation stocke toutes ses images dans la page. La résolution est
donc volontairement plus basse que pour la vue fixe de `plume3d`, et les
coordonnées sont arrondies : le fichier passe d'une dizaine de mégaoctets à
deux ou trois, sans différence visible à l'écran.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from flame.loaders.psi115 import load_mesh
from flame.viz.plume3d import (
    FIELD_STYLES,
    PLANE_STEP,
    SWEEP_DEGREES,
    _check_axisymmetry,
    _isocontour,
    _load_field,
    _plane,
)

# Resolution reduite : chaque image est stockee dans la page.
FRAMES = 14
ANIM_POINTS = 110
ANIM_ANGLES = 40
# Arrondi des coordonnees ecrites dans le HTML. Quatre decimales sur des
# valeurs de l'ordre de l'unite, c'est bien au-dela de la precision d'affichage.
DECIMALS = 4
ORBIT_FRAMES = 48
# Seuil le plus bas du balayage, en fraction du maximum du champ.
LOWEST_LEVEL_FRACTION = 0.025


def _resample(curve: np.ndarray, count: int) -> np.ndarray:
    if len(curve) <= count:
        return curve
    keep = np.linspace(0, len(curve) - 1, count).astype(int)
    return curve[keep]


def _revolved_surface(curve: np.ndarray, colorscale: str) -> go.Surface:
    """Révolution d'un contour, arrondie pour alléger la page."""
    x = curve[:, 0]
    radius = np.abs(curve[:, 1])
    theta = np.radians(np.linspace(0, SWEEP_DEGREES, ANIM_ANGLES))

    return go.Surface(
        x=np.round(np.tile(x[:, None], (1, ANIM_ANGLES)), DECIMALS),
        y=np.round(radius[:, None] * np.cos(theta)[None, :], DECIMALS),
        z=np.round(radius[:, None] * np.sin(theta)[None, :], DECIMALS),
        surfacecolor=np.round(np.tile(x[:, None], (1, ANIM_ANGLES)), DECIMALS),
        colorscale=colorscale,
        showscale=False,
        opacity=0.93,
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.2),
        hoverinfo="skip",
    )


def sweep(variable: str = "Smoke", burner_size_cm: int = 8) -> go.Figure:
    """Balayage du seuil d'isosurface, du cœur dense vers le bord.

    Le champ ne change jamais. Ce qui change est la valeur à laquelle on
    découpe la surface : image après image, on descend du cœur le plus dense
    vers la limite extérieure du panache.
    """
    label, colorscale, _ = FIELD_STYLES[variable]
    axis_x, axis_y = load_mesh()
    case = f"{burner_size_cm}cmNoGravity"
    field = _load_field(case, variable)
    _check_axisymmetry(field, case)

    finite = field[np.isfinite(field)]
    # Du plus dense vers le plus tenu : on part du coeur et on ouvre.
    #
    # La borne basse ne peut pas etre un percentile du champ entier : plus de
    # 60 % des cellules valent exactement zero (tout ce qui est hors panache),
    # donc le percentile tombe a 0 et le contour epouse alors le bord du
    # domaine au lieu du panache. On la fixe a une petite fraction du maximum,
    # ce qui designe toujours une vraie limite de panache.
    ceiling = float(np.percentile(finite, 99.9))
    floor = LOWEST_LEVEL_FRACTION * float(np.nanmax(finite))
    levels = np.linspace(ceiling, floor, FRAMES)

    frames, labels = [], []
    first_surface = None
    for level in levels:
        paths = _isocontour(field, axis_x, axis_y, float(level))
        if not paths:
            continue
        curve = _resample(max(paths, key=len), ANIM_POINTS)
        surface = _revolved_surface(curve, colorscale)
        if first_surface is None:
            first_surface = surface
        name = f"{level:.2f}"
        labels.append(name)
        frames.append(go.Frame(data=[surface], traces=[0], name=name))

    if first_surface is None:
        raise ValueError(f"{case} : aucun contour sur la plage de seuils demandee")

    figure = go.Figure(
        data=[first_surface, _plane(field, axis_x, axis_y, colorscale, True)],
        frames=frames,
    )

    figure.update_layout(
        title=dict(
            text=(
                f"PSI-115 — {label}, microgravite, bruleur {burner_size_cm} cm"
                "<br><sub>LE CHAMP NE BOUGE PAS. Ce qui varie est le seuil de decoupe"
                " de l'isosurface : chaque image montre une densite differente du"
                " MEME etat stationnaire,<br>du coeur dense vers le bord du panache."
                " Aucune donnee temporelle n'existe dans PSI-115."
                "<br>Simulation numerique, unites normalisees.</sub>"
            ),
            x=0.01,
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="x (u. norm.)",
            yaxis_title="y (u. norm.)",
            zaxis_title="z (u. norm.)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.02,
                xanchor="left",
                buttons=[
                    dict(
                        label="Lecture",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=170, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.15,
                y=0.02,
                len=0.75,
                currentvalue=dict(prefix="seuil d'isosurface : ", font=dict(size=13)),
                steps=[
                    dict(
                        label=name,
                        method="animate",
                        args=[
                            [name],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                    )
                    for name in labels
                ],
            )
        ],
        margin=dict(l=0, r=0, t=120, b=70),
        height=760,
        template="plotly_dark",
    )
    return figure


def orbit(variable: str = "Smoke", burner_size_cm: int = 8) -> go.Figure:
    """Rotation de caméra autour d'une scène entièrement figée.

    Aucune donnée ne change d'une image à l'autre : seule la position de
    l'observateur tourne. C'est de la présentation, assumée comme telle.
    """
    label, colorscale, level = FIELD_STYLES[variable]
    axis_x, axis_y = load_mesh()
    case = f"{burner_size_cm}cmNoGravity"
    field = _load_field(case, variable)
    _check_axisymmetry(field, case)

    curve = _resample(
        max(_isocontour(field, axis_x, axis_y, level), key=len), ANIM_POINTS
    )
    surface = _revolved_surface(curve, colorscale)
    plane = _plane(field, axis_x, axis_y, colorscale, True)

    radius, height = 2.1, 0.75
    frames = [
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=radius * np.cos(angle),
                        y=radius * np.sin(angle),
                        z=height,
                    )
                )
            ),
            name=f"{int(np.degrees(angle))}",
        )
        for angle in np.linspace(0, 2 * np.pi, ORBIT_FRAMES, endpoint=False)
    ]

    figure = go.Figure(data=[surface, plane], frames=frames)
    figure.update_layout(
        title=dict(
            text=(
                f"PSI-115 — {label}, microgravite, bruleur {burner_size_cm} cm"
                f"<br><sub>SEULE LA CAMERA TOURNE. Isosurface fixe a {level:g},"
                " champ stationnaire : rien dans les donnees ne bouge."
                "<br>Simulation numerique, unites normalisees.</sub>"
            ),
            x=0.01,
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="x (u. norm.)",
            yaxis_title="y (u. norm.)",
            zaxis_title="z (u. norm.)",
            aspectmode="data",
            camera=dict(eye=dict(x=radius, y=0, z=height)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.02,
                xanchor="left",
                buttons=[
                    dict(
                        label="Lecture",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=70, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        margin=dict(l=0, r=0, t=110, b=60),
        height=760,
        template="plotly_dark",
    )
    return figure


def main() -> None:
    output = Path("data/figures")
    output.mkdir(parents=True, exist_ok=True)

    for name, builder in [("sweep", sweep), ("orbit", orbit)]:
        figure = builder(variable="Smoke", burner_size_cm=8)
        destination = output / f"psi115_anim_{name}.html"
        figure.write_html(destination, include_plotlyjs="cdn", auto_play=False)
        print(
            f"  {destination}  ({destination.stat().st_size / 1024:.0f} Ko, "
            f"{len(figure.frames)} images)"
        )

    print(
        "\n  sweep : le seuil de decoupe varie, le champ non."
        "\n  orbit : la camera tourne, rien d'autre."
        "\n  Aucune des deux n'est une combustion filmee — PSI-115 n'a pas de temps."
    )


if __name__ == "__main__":
    main()
