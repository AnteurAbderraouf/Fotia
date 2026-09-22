"""La flamme de gouttelette, reliée aux commandes du tableau de bord.

POURQUOI UNE SPHÈRE EST ICI UNE REPRÉSENTATION EXACTE.

Une flamme de gouttelette en microgravité EST sphérique — c'est la raison même
pour laquelle ces essais se font en orbite. Sans flottabilité, pas de
convection naturelle, donc pas de panache et aucune direction privilégiée : la
flamme entoure la goutte comme une coquille.

Dessiner une sphère du diamètre voulu n'est donc ni une illustration ni une
simulation, c'est l'énoncé géométrique d'une taille.

CE QUE LA VUE MONTRE.

    sphere externe, translucide   le diametre initial, regle par l'utilisateur
    coquille pleine               le diametre d'extinction PREDIT
    halo autour                   l'incertitude du modele, +/- son erreur
                                  moyenne en validation croisee

Le halo n'est pas décoratif. Une prédiction affichée sans son incertitude se
lit comme une certitude, et l'erreur du modèle vaut 0.35 mm sur des diamètres
qui vont de 0.7 à 4.6 mm : ce n'est pas négligeable, donc ça se voit.

CE QUE LA VUE NE MONTRE PAS. Aucune transition entre les deux états. PSI-69
relève deux instants — la taille au départ, la taille à l'extinction — et rien
entre les deux. Une goutte qui rétrécirait progressivement à l'écran serait
inventée.

LA COQUILLE D'EXTINCTION EST INTÉRIEURE À LA SPHÈRE DE DÉPART, et c'est normal :
la goutte brûle et rétrécit avant que la flamme ne lâche. Le rapport vaut 0.70
en moyenne sur les essais réels.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

RESOLUTION = 40
# Revolution partielle : la decoupe laisse voir les coquilles interieures.
SWEEP_DEGREES = 250

INITIAL = "#3987e5"
PREDICTED = "#f2a649"
UNCERTAINTY = "#8a6d1f"
SURFACE = "#0d0d0d"
MUTED_INK = "#898781"
SECONDARY_INK = "#c3c2b7"


def _shell(diameter: float, colour: str, opacity: float, name: str) -> go.Surface:
    """Une coquille sphérique, ouverte pour laisser voir l'intérieur."""
    radius = max(diameter, 1e-6) / 2
    theta = np.radians(np.linspace(0, SWEEP_DEGREES, RESOLUTION))
    phi = np.linspace(0, np.pi, RESOLUTION)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    return go.Surface(
        x=radius * np.sin(phi_grid) * np.cos(theta_grid),
        y=radius * np.sin(phi_grid) * np.sin(theta_grid),
        z=radius * np.cos(phi_grid),
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.62, diffuse=0.8, specular=0.2),
        hovertemplate=f"{name}<br>%{{text}} mm<extra></extra>",
        text=[[f"{diameter:.2f}"] * RESOLUTION] * RESOLUTION,
    )


def figure(
    initial_mm: float,
    predicted_mm: float | None,
    error_mm: float = 0.0,
    extinction_chance: float | None = None,
) -> go.Figure:
    """La gouttelette au départ et la coquille d'extinction prédite.

    predicted_mm à None signifie qu'aucun diamètre n'est affiché — parce que le
    modèle n'a rien à dire, ou parce que la flamme n'est pas censée s'éteindre.
    """
    traces = [_shell(initial_mm, INITIAL, 0.2, "diametre initial")]

    if predicted_mm is not None:
        upper = min(predicted_mm + error_mm, initial_mm)
        lower = max(predicted_mm - error_mm, 0.0)
        if error_mm > 0:
            traces.append(_shell(upper, UNCERTAINTY, 0.16, "incertitude haute"))
            traces.append(_shell(lower, UNCERTAINTY, 0.16, "incertitude basse"))
        traces.append(_shell(predicted_mm, PREDICTED, 0.95, "extinction predite"))

    scene = go.Figure(traces)
    extent = initial_mm / 2 * 1.15

    subtitle = "En microgravite une flamme de gouttelette est spherique."
    if extinction_chance is not None and extinction_chance < 0.5:
        subtitle += (
            f" Attention : le modele ne donne que {extinction_chance:.0%} de"
            " chances d'extinction — ce diametre decrit un cas peu probable."
        )

    scene.update_layout(
        title=dict(
            text=(
                "Gouttelette et coquille d'extinction"
                f"<br><sub>{subtitle}</sub>"
            ),
            x=0.02,
            font=dict(size=14, color=SECONDARY_INK),
        ),
        scene=dict(
            xaxis=dict(range=[-extent, extent], title="mm", color=MUTED_INK),
            yaxis=dict(range=[-extent, extent], title="mm", color=MUTED_INK),
            zaxis=dict(range=[-extent, extent], title="mm", color=MUTED_INK),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
        paper_bgcolor=SURFACE,
        margin=dict(l=0, r=0, t=64, b=0),
        height=420,
        showlegend=False,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    return scene
