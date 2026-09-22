"""Flammes de gouttelettes en 3D — la sphère est la mesure, pas une illustration.

POURQUOI UNE SPHÈRE EST ICI UNE REPRÉSENTATION EXACTE.

Une flamme de gouttelette en microgravité EST sphérique. C'est la raison même
pour laquelle ces essais se font en orbite : sans flottabilité, pas de
convection naturelle, donc pas de panache et pas de direction privilégiée. La
flamme entoure la goutte comme une coquille.

Dessiner une sphère du diamètre mesuré n'est donc pas une illustration ni une
simulation : c'est l'énoncé géométrique de la mesure. `dext_mm` est le
diamètre auquel la flamme visible s'est éteinte ; la coquille affichée a ce
diamètre-là, relevé sur la vidéo de l'essai.

CE QUE LA FIGURE MONTRE. Pour des essais réels choisis sur toute la plage de
CO2, la gouttelette initiale et la coquille d'extinction, à l'échelle l'une
par rapport à l'autre. Un « diamètre d'extinction de 3.5 mm » cesse d'être un
nombre et devient une taille qu'on compare à l'œil.

CE QUE LA FIGURE N'EST PAS. Ni une animation de combustion, ni un rendu de
flamme. Aucune donnée temporelle n'existe dans PSI-69 : seuls le diamètre
initial et le diamètre d'extinction sont relevés, soit deux instants. Toute
transition affichée entre les deux serait inventée.

UNE PRÉCISION SUR L'ÉCHELLE DES DIAMÈTRES. `dext_mm` est plus petit que
`d0_mm` : la goutte a brûlé et rétréci avant que la flamme ne lâche. La
coquille d'extinction est donc INTÉRIEURE à la sphère de départ. Ce n'est pas
une erreur de tracé, c'est ce que la combustion fait.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from flame.loaders.psi69 import load

RESOLUTION = 36
# Revolution partielle : la decoupe laisse voir la sphere interieure.
SWEEP_DEGREES = 260


def _sphere(
    diameter: float, centre: float, colour: str, name: str, opacity: float
) -> go.Surface:
    """Une demi-coquille de révolution, décentrée sur l'axe x."""
    radius = diameter / 2
    theta = np.radians(np.linspace(0, SWEEP_DEGREES, RESOLUTION))
    phi = np.linspace(0, np.pi, RESOLUTION)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    return go.Surface(
        x=centre + radius * np.sin(phi_grid) * np.cos(theta_grid),
        y=radius * np.sin(phi_grid) * np.sin(theta_grid),
        z=radius * np.cos(phi_grid),
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.25),
        hovertemplate=f"{name}<br>diametre {diameter:.2f} mm<extra></extra>",
    )


def select_burns(count: int = 5) -> pd.DataFrame:
    """Essais réels couvrant la plage de CO2, à carburant constant.

    Le carburant est fixé au méthanol et l'hélium à zéro, pour retirer deux
    effets du tableau.

    CE QU'ON NE PEUT PAS FAIRE, ET IL FAUT LE DIRE. Une vraie série contrôlée
    demanderait de ne faire varier QUE le CO2, à oxygène et pression
    constants. Le plan d'expérience de FLEX-1 ne le permet pas : à O2 = 0.21
    et 1 atm, seuls deux niveaux de CO2 ont été essayés, 0 % et 15 %. Au-delà,
    la teneur en CO2 ne monte qu'en compagnie d'un oxygène ou d'une pression
    différents — le même confondement que celui trouvé sur le modèle.

    Cette figure est donc une GALERIE D'ESSAIS RÉELS, pas une série contrôlée.
    Chaque sphère porte ses conditions complètes, et les écarts de taille ne
    s'attribuent pas au seul CO2.
    """
    df = load()
    usable = df[
        df["dext_mm"].notna()
        & df["d0_mm"].notna()
        & (df["fuel"] == "Methanol")
        & (df["he_frac"] == 0)
    ].copy()

    if usable.empty:
        raise ValueError("aucun essai methanol avec les deux diametres releves")

    usable = usable.sort_values("co2_frac")
    picks = np.linspace(0, len(usable) - 1, min(count, len(usable))).astype(int)
    return usable.iloc[picks]


def figure(count: int = 5) -> go.Figure:
    """Gouttelette de départ et coquille d'extinction, essai par essai."""
    burns = select_burns(count)
    spacing = float(burns["d0_mm"].max()) * 1.6

    scene = go.Figure()
    annotations = []
    for index, (_, burn) in enumerate(burns.iterrows()):
        centre = index * spacing
        scene.add_trace(
            _sphere(burn["d0_mm"], centre, "#5B8FF9", "gouttelette initiale", 0.22)
        )
        scene.add_trace(
            _sphere(burn["dext_mm"], centre, "#F2A649", "diametre d'extinction", 0.95)
        )
        annotations.append(
            dict(
                x=centre,
                y=0,
                z=-burns["d0_mm"].max() * 0.75,
                text=(
                    f"CO2 {burn['co2_frac']:.0%}<br>"
                    f"O2 {burn['o2_frac']:.2f}<br>"
                    f"{burn['pressure_atm']:.2f} atm<br>"
                    f"d0 {burn['d0_mm']:.2f} mm<br>"
                    f"dext {burn['dext_mm']:.2f} mm"
                ),
                showarrow=False,
                font=dict(size=11),
            )
        )

    scene.update_layout(
        title=dict(
            text=(
                "PSI-69 / FLEX-1 — gouttelettes de methanol, effet du CO2"
                "<br><sub>Bleu : gouttelette au depart. Orange : diametre auquel la flamme"
                " visible s'est eteinte, releve sur la video de l'essai.<br>"
                "En microgravite une flamme de gouttelette est spherique : la sphere est"
                " la mesure, pas une illustration.<br>"
                "GALERIE D'ESSAIS REELS, PAS UNE SERIE CONTROLEE : a O2 et pression"
                " constants, FLEX-1 n'a essaye que deux teneurs en CO2 (0 % et 15 %)."
                "<br>Les conditions varient donc d'une sphere a l'autre, lire les"
                " etiquettes avant de conclure.</sub>"
            ),
            x=0.01,
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="essais, par teneur en CO2 croissante (mm)",
            yaxis_title="mm",
            zaxis_title="mm",
            aspectmode="data",
            camera=dict(eye=dict(x=0.2, y=-2.2, z=0.9)),
            annotations=annotations,
        ),
        margin=dict(l=0, r=0, t=110, b=0),
        height=650,
        template="plotly_dark",
        showlegend=False,
    )
    return scene


def main() -> None:
    output = Path("data/figures")
    output.mkdir(parents=True, exist_ok=True)

    burns = select_burns()
    print("essais retenus (methanol, helium nul, CO2 croissant) :\n")
    view = burns[["co2_frac", "o2_frac", "pressure_atm", "d0_mm", "dext_mm", "test_end"]]
    print(view.to_string(index=False))

    shrink = (1 - burns["dext_mm"] / burns["d0_mm"]).mean()
    print(
        f"\n  la goutte a perdu en moyenne {shrink:.0%} de son diametre"
        "\n  avant que la flamme ne lache : la coquille d'extinction est"
        "\n  interieure a la sphere de depart, et c'est normal."
    )

    destination = output / "psi69_droplet3d.html"
    figure().write_html(destination, include_plotlyjs="cdn")
    print(f"\n  {destination}  ({destination.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
