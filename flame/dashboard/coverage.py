"""Couverture du plan d'expérience — montrer où les essais existent vraiment.

LE PROBLÈME QUE CE MODULE RÉSOUT.

Le voyant de distance dit « vous êtes hors du domaine » une fois qu'on y est.
C'est nécessaire mais frustrant : on déplace des curseurs à l'aveugle et on
tombe sur un avertissement sans savoir où aller.

Ce module retourne la question. Il montre AVANT de choisir où les essais se
trouvent, pour qu'on puisse viser.

CE QU'IL RÉVÈLE, ET QUI N'EST PAS ÉVIDENT.

Le domaine testé n'est pas une boîte. Sur PSI-69, les 14 valeurs d'oxygène et
les 14 valeurs de CO2 ouvrent 196 combinaisons ; **31 ont été essayées**, soit
16 %. Le motif est celui d'un plan « un facteur à la fois » : la colonne sans
CO2 est complète sur les 14 niveaux d'oxygène, puis chaque niveau d'oxygène
n'a été croisé qu'avec un ou deux niveaux de CO2.

Conséquence contre-intuitive : **choisir deux valeurs testées séparément ne
garantit pas que leur combinaison l'ait été.** On peut être sur une valeur
d'oxygène essayée 25 fois et une valeur de CO2 essayée 12 fois, et se trouver
dans une case que personne n'a jamais visitée.

CHOIX D'AFFICHAGE. Zéro essai n'est pas une petite magnitude, c'est une
absence : les cases vides prennent la couleur du fond et une bordure fine,
plutôt que le bas de la rampe de couleur. Sinon « jamais testé » se lirait
comme « un peu testé ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Rampe séquentielle à teinte unique, du référentiel de palette. Sur fond
# sombre, c'est le pas le plus SOMBRE qui se fond dans la surface : la
# magnitude croissante va donc vers le clair.
BLUE_RAMP = ["#184f95", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"]
SURFACE = "#1a1a19"
GRIDLINE = "#2c2c2a"
MUTED_INK = "#898781"
SECONDARY_INK = "#c3c2b7"
MARKER = "#eda100"

AXIS_LABELS = {
    "o2_frac": "oxygene (fraction molaire)",
    "co2_frac": "CO2 ajoute (fraction molaire)",
    "he_frac": "helium ajoute (fraction molaire)",
    "pressure_atm": "pression (atm)",
    "d0_mm": "diametre initial (mm)",
}


def tested_values(frame: pd.DataFrame, feature: str) -> list[float]:
    """Les valeurs réellement essayées pour une variable, triées."""
    return sorted(float(v) for v in frame[feature].dropna().unique())


def coverage_table(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Nombre d'essais par croisement de deux variables."""
    return pd.crosstab(frame[y], frame[x])


def coverage_stats(frame: pd.DataFrame, x: str, y: str) -> dict:
    table = coverage_table(frame, x, y)
    filled = int((table > 0).sum().sum())
    return {
        "cells": int(table.size),
        "filled": filled,
        "share": filled / table.size if table.size else 0.0,
        "rows": table.shape[0],
        "columns": table.shape[1],
    }


def coverage_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    current: tuple[float, float] | None = None,
) -> go.Figure:
    """Carte des combinaisons réellement essayées, avec la position courante."""
    table = coverage_table(frame, x, y)
    counts = table.to_numpy(dtype=float)
    # Les cases vides recoivent NaN et non zero : elles se fondent dans le fond
    # au lieu de prendre la teinte la plus faible de la rampe.
    counts_display = np.where(counts > 0, counts, np.nan)

    labels_x = [f"{v:g}" for v in table.columns]
    labels_y = [f"{v:g}" for v in table.index]

    figure = go.Figure(
        go.Heatmap(
            z=counts_display,
            x=labels_x,
            y=labels_y,
            colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
            xgap=2,
            ygap=2,
            hovertemplate=(
                f"{AXIS_LABELS.get(x, x)} %{{x}}<br>"
                f"{AXIS_LABELS.get(y, y)} %{{y}}<br>"
                "<b>%{z:.0f} essais</b><extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="essais", side="right"),
                thickness=11,
                outlinewidth=0,
                tickfont=dict(color=MUTED_INK, size=11),
                title_font=dict(color=SECONDARY_INK, size=11),
            ),
        )
    )

    if current is not None:
        # La position courante est reportee sur la case la plus proche, les
        # axes etant categoriels : ce sont les valeurs essayees, pas un
        # continuum.
        nearest_x = min(table.columns, key=lambda v: abs(float(v) - current[0]))
        nearest_y = min(table.index, key=lambda v: abs(float(v) - current[1]))
        count = int(table.loc[nearest_y, nearest_x])
        figure.add_scatter(
            x=[f"{nearest_x:g}"],
            y=[f"{nearest_y:g}"],
            mode="markers",
            marker=dict(
                symbol="square-open",
                size=26,
                color=MARKER,
                line=dict(width=2.5, color=MARKER),
            ),
            name="position",
            hovertemplate=(
                f"position courante<br><b>{count} essai(s)</b> dans cette case"
                "<extra></extra>"
            ),
            showlegend=False,
        )

    stats = coverage_stats(frame, x, y)
    figure.update_layout(
        title=dict(
            text=(
                f"{stats['filled']} combinaisons essayees sur {stats['cells']} "
                f"({stats['share']:.0%})"
            ),
            font=dict(size=13, color=SECONDARY_INK),
            x=0,
        ),
        xaxis=dict(
            title=dict(text=AXIS_LABELS.get(x, x), font=dict(size=11, color=MUTED_INK)),
            type="category",
            tickfont=dict(size=10, color=MUTED_INK),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text=AXIS_LABELS.get(y, y), font=dict(size=11, color=MUTED_INK)),
            type="category",
            tickfont=dict(size=10, color=MUTED_INK),
            showgrid=False,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        margin=dict(l=0, r=0, t=34, b=0),
        height=330,
    )
    return figure


def nearest_real_test(
    frame: pd.DataFrame, query: dict, features: list[str]
) -> pd.Series:
    """L'essai réel le plus proche, pour s'y placer d'un clic.

    Les variables catégorielles filtrent, les numériques mesurent — même
    règle que dans l'index de proximité.
    """
    eligible = frame
    for name in features:
        if name in query and not pd.api.types.is_numeric_dtype(frame[name]):
            match = frame[name] == query[name]
            if match.any():
                eligible = eligible[match]

    numeric = [
        name
        for name in features
        if name in query and pd.api.types.is_numeric_dtype(frame[name])
    ]
    values = eligible[numeric].to_numpy(dtype=float)
    spread = np.nanstd(frame[numeric].to_numpy(dtype=float), axis=0)
    spread[spread == 0] = 1.0
    target = np.array([float(query[name]) for name in numeric])

    distances = np.sqrt((((values - target) / spread) ** 2).sum(axis=1))
    return eligible.iloc[int(np.argmin(distances))]
