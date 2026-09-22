"""Panache de fumée en 3D — et pourquoi seule la microgravité s'y prête.

LE POINT DE CETTE VISUALISATION.

En microgravité, rien ne désigne une direction privilégiée. Le panache est un
corps de révolution : la tranche calculée est symétrique autour de l'axe, à la
précision numérique près (asymétrie mesurée : 0.0002). La faire tourner autour
de y = 0 reconstruit donc le volume que la simulation représentait déjà — c'est
l'opération inverse de l'hypothèse de calcul, pas une extrapolation.

Sur Terre, la gravité désigne une direction. Le panache chaud monte, il dévie,
et la symétrie axiale disparaît : l'asymétrie mesurée passe à 0.03-0.11 selon
la variable. Le faire tourner autour de y = 0 produirait un tore qui ne
correspond à rien de physique.

Autrement dit, l'impossibilité de dessiner le cas terrestre comme le cas
spatial N'EST PAS une limite de l'outil, c'est le résultat. C'est pourquoi les
deux panneaux ne sont pas représentés de la même façon : à gauche un volume,
à droite le plan de calcul tel qu'il est. Les représenter pareil aurait été
plus joli et faux.

`_check_axisymmetry` refuse de révolutionner un champ asymétrique. Une règle
inscrite dans le code ne s'oublie pas ; une règle écrite en commentaire, si.

CE QUE CE N'EST PAS. Pas une animation, pas de la turbulence, pas une
simulation refaite ici. Ces champs sont stationnaires et il n'existe aucune
série temporelle : tout mouvement affiché serait inventé.

ÉCHELLE. Coordonnées normalisées, pas métriques — x de 0 à 40, y de -3.409 à
3.409, convention de normalisation non documentée par NASA. Les formes et les
rapports sont justes, une longueur absolue ne l'est pas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from flame.loaders.psi115 import FIELDS_DIR, load_mesh

ANGLES = 72
# Un contour brut porte ~1500 sommets, six fois plus fin que ce que l'oeil
# distingue, et produit une page de 12 Mo. On le ramene a :
CONTOUR_POINTS = 260
PLANE_STEP = 4
# Revolution sur 300 degres et non 360 : la decoupe laisse voir l'interieur.
SWEEP_DEGREES = 300
# Au-dela de ce niveau d'asymetrie, une revolution n'a plus de sens physique.
SYMMETRY_TOLERANCE = 0.01

FIELD_STYLES = {
    "Smoke": ("fraction de fumee", "Magma", 0.35),
    "Temperature": ("temperature normalisee", "Inferno", 1.05),
    "NumDen": ("densite numerique de particules", "Viridis", 0.35),
}


def _load_field(case: str, variable: str) -> np.ndarray:
    path = FIELDS_DIR / f"{case}{variable}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable dans {FIELDS_DIR}")
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def axial_asymmetry(field: np.ndarray) -> float:
    """Écart moyen entre le champ et son miroir, rapporté à son amplitude.

    Zéro signifie parfaitement symétrique autour de y = 0, donc axisymétrique.
    """
    scale = np.nanmax(np.abs(field))
    if not scale:
        return 0.0
    return float(np.nanmean(np.abs(field - field[::-1])) / scale)


def _check_axisymmetry(field: np.ndarray, case: str) -> float:
    """Interdit la révolution d'un champ qui n'est pas de révolution."""
    asymmetry = axial_asymmetry(field)
    if asymmetry > SYMMETRY_TOLERANCE:
        raise ValueError(
            f"{case} : asymetrie {asymmetry:.4f} au-dessus de "
            f"{SYMMETRY_TOLERANCE}. Ce champ n'est pas axisymetrique — la "
            "gravite y designe une direction — et le revolutionner "
            "produirait une forme sans realite physique."
        )
    return asymmetry


def _isocontour(
    field: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray, level: float
) -> list[np.ndarray]:
    """Courbes de niveau dans le plan. matplotlib sert de moteur géométrique."""
    figure = plt.figure()
    try:
        contour = plt.contour(axis_x, axis_y, field, levels=[level])
        paths = []
        for path in contour.get_paths():
            vertices = np.asarray(path.vertices)
            if len(vertices) <= 8:
                continue
            if len(vertices) > CONTOUR_POINTS:
                keep = np.linspace(0, len(vertices) - 1, CONTOUR_POINTS).astype(int)
                vertices = vertices[keep]
            paths.append(vertices)
    finally:
        plt.close(figure)
    return paths


def _revolve(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fait tourner une courbe (x, y) autour de l'axe y = 0."""
    x = curve[:, 0]
    radius = np.abs(curve[:, 1])
    theta = np.radians(np.linspace(0, SWEEP_DEGREES, ANGLES))
    return (
        np.tile(x[:, None], (1, ANGLES)),
        radius[:, None] * np.cos(theta)[None, :],
        radius[:, None] * np.sin(theta)[None, :],
    )


def _plane(field: np.ndarray, axis_x, axis_y, colorscale: str, showscale: bool):
    """Le plan de calcul, affiché tel qu'il est."""
    sliced = field[::PLANE_STEP, ::PLANE_STEP]
    plane_x, plane_y = np.meshgrid(axis_x[::PLANE_STEP], axis_y[::PLANE_STEP])
    return go.Surface(
        x=plane_x,
        y=plane_y,
        z=np.zeros_like(plane_x),
        surfacecolor=sliced,
        colorscale=colorscale,
        showscale=showscale,
        colorbar=dict(title=dict(text="valeur", side="right"), thickness=12, x=1.0),
        hovertemplate="x %{x:.2f}<br>y %{y:.2f}<br>valeur %{surfacecolor:.3f}<extra></extra>",
    )


def compare(variable: str = "Smoke", burner_size_cm: int = 8) -> go.Figure:
    """Microgravité en volume, gravité terrestre en plan — et le pourquoi."""
    label, colorscale, level = FIELD_STYLES[variable]
    axis_x, axis_y = load_mesh()

    micro = _load_field(f"{burner_size_cm}cmNoGravity", variable)
    earth = _load_field(f"{burner_size_cm}cmGravity", variable)
    micro_asymmetry = _check_axisymmetry(micro, f"{burner_size_cm}cmNoGravity")
    earth_asymmetry = axial_asymmetry(earth)

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"Microgravite — volume de revolution<br>"
            f"<sub>asymetrie {micro_asymmetry:.4f} : axisymetrique</sub>",
            f"Gravite terrestre — plan de calcul<br>"
            f"<sub>asymetrie {earth_asymmetry:.4f} : la revolution n'a pas de sens</sub>",
        ),
        horizontal_spacing=0.02,
    )

    for curve in _isocontour(micro, axis_x, axis_y, level):
        X, Y, Z = _revolve(curve)
        figure.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=np.tile(curve[:, 0][:, None], (1, ANGLES)),
                colorscale=colorscale,
                showscale=False,
                opacity=0.9,
                lighting=dict(ambient=0.55, diffuse=0.85, specular=0.2),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(_plane(micro, axis_x, axis_y, colorscale, False), row=1, col=1)
    figure.add_trace(_plane(earth, axis_x, axis_y, colorscale, True), row=1, col=2)

    axes = dict(
        xaxis_title="x (u. norm.)",
        yaxis_title="y (u. norm.)",
        zaxis_title="z (u. norm.)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    figure.update_layout(
        title=dict(
            text=(
                f"PSI-115 — {label}, bruleur {burner_size_cm} cm, isosurface a {level:g}"
                "<br><sub>En microgravite rien ne designe une direction : le panache est"
                " un corps de revolution. Sur Terre la gravite en designe une, le panache"
                " devie,<br>et la symetrie axiale disparait. Les deux panneaux ne sont pas"
                " dessines pareil parce que les deux situations ne le sont pas."
                "<br>Simulation numerique, pas une mesure. Unites normalisees.</sub>"
            ),
            x=0.01,
            font=dict(size=15),
        ),
        scene=axes,
        scene2=axes,
        margin=dict(l=0, r=0, t=130, b=0),
        height=720,
        template="plotly_dark",
    )
    return figure


def main() -> None:
    output = Path("data/figures")
    output.mkdir(parents=True, exist_ok=True)

    print("asymetrie axiale mesuree (0 = corps de revolution) :\n")
    for size in (8, 10):
        for suffix, name in [("NoGravity", "microgravite"), ("Gravity", "1g")]:
            try:
                field = _load_field(f"{size}cm{suffix}", "Smoke")
            except FileNotFoundError:
                continue
            value = axial_asymmetry(field)
            verdict = (
                "revolution licite"
                if value <= SYMMETRY_TOLERANCE
                else "revolution refusee par le code"
            )
            print(f"  {size:2d} cm  {name:14} {value:.4f}   {verdict}")

    print()
    for variable in ["Smoke", "Temperature"]:
        figure = compare(variable=variable, burner_size_cm=8)
        destination = output / f"psi115_plume3d_{variable.lower()}.html"
        figure.write_html(destination, include_plotlyjs="cdn")
        print(f"  {destination}  ({destination.stat().st_size / 1024:.0f} Ko)")

    print(
        "\nOuvrir dans un navigateur. Les deux scenes se tournent a la souris"
        "\nindependamment ; comparer le volume de gauche au plan de droite."
    )


if __name__ == "__main__":
    main()
