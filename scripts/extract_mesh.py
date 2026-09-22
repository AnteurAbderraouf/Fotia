"""Extrait les axes du maillage PSI-115 depuis le fichier .dat de NASA.

    python scripts/extract_mesh.py

Répare le cas limite laissé ouvert au §5 du handoff. Les fichiers
`*_sGrid_grid3d.dat` étaient notés « FAILED » dans le manifeste parce que
pandas y voyait des lignes de longueurs incompatibles. Le format est en fait
très simple :

    ligne 1   trois entiers : nx ny nz  (960 240 1)
    ligne 2   nx * ny * nz * 3 valeurs, toutes sur UNE seule ligne
              c'est-à-dire les coordonnées x, y, z de chaque nœud

Soit 691 200 nombres sur une ligne. Rien d'irrégulier, simplement un format
que `read_csv` ne peut pas deviner. `numpy.fromstring` le lit sans difficulté.

CE QUE LE MAILLAGE APPREND :

  * z vaut zéro partout et y est symétrique autour de zéro : c'est une
    tranche 2D AXISYMÉTRIQUE, dont la ligne y = 0 est l'axe de révolution.
    C'est ce qui autorise à la révolutionner en volume 3D — on reconstruit
    alors le volume que la simulation représentait déjà, on n'invente rien.

  * le maillage est ÉTIRÉ : le pas varie d'un facteur 3.8 en x et 1.5 en y.
    Toute statistique calculée en supposant des cellules équivalentes est
    donc fausse, et c'était le cas de l'étendue de panache calculée
    auparavant.

  * la grille est SÉPARABLE : x ne dépend que de la colonne, y que de la
    ligne. Deux axes de 960 et 240 valeurs suffisent, au lieu des 460 800
    coordonnées. D'où ce script : les 23 Ko produits entrent dans le dépôt,
    les 8,8 Mo bruts n'y entreraient pas.

  * les deux fichiers, 8cm et 10cm, sont IDENTIQUES. Le domaine est le même,
    seule la physique change d'un cas à l'autre.

UNITÉS. Les coordonnées sont normalisées, pas métriques : x va de 0 à 40 et
y de -3.409 à 3.409, ce qui ne peut pas être des mètres pour un brûleur de
8 cm. La convention de normalisation n'est documentée nulle part dans les
fichiers NASA. Les formes et les rapports sont donc justes, l'échelle absolue
reste inconnue — à dire explicitement partout où ces chiffres s'affichent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.common.paths import psi_dir  # noqa: E402

RAW_MESH = "8cm_sGrid_grid3d.dat"


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lit un fichier de maillage et renvoie ses deux axes."""
    with open(path) as handle:
        nx, ny, nz = (int(value) for value in handle.readline().split())
        flat = np.fromstring(handle.read(), sep=" ")

    expected = 3 * nx * ny * nz
    if flat.size != expected:
        raise ValueError(
            f"{path.name} : {flat.size} valeurs lues, {expected} attendues "
            f"pour une grille {nx}x{ny}x{nz}"
        )

    x, y, z = flat.reshape(3, ny, nx)

    if not np.allclose(z, 0):
        raise ValueError(f"{path.name} : z non nul, la tranche n'est pas plane")
    if not np.allclose(x, x[0][None, :]) or not np.allclose(y, y[:, 0][:, None]):
        raise ValueError(
            f"{path.name} : grille non separable, les deux axes ne suffisent pas"
        )

    return x[0], y[:, 0]


def main() -> None:
    candidates = sorted(psi_dir("PSI-115").rglob("*_sGrid_grid3d.dat"))
    if not candidates:
        raise SystemExit(
            "aucun fichier de maillage trouve. Il vit dans raw/, qui est exclu "
            "du depot : le reconstruire avec scripts/unflatten.py depuis le zip "
            "NASA d'origine."
        )

    meshes = {path.name: read_mesh(path) for path in candidates}
    names = list(meshes)
    reference = meshes[names[0]]
    for name in names[1:]:
        if not (
            np.allclose(meshes[name][0], reference[0])
            and np.allclose(meshes[name][1], reference[1])
        ):
            raise ValueError(f"{name} differe de {names[0]} : un seul maillage attendu")

    axis_x, axis_y = reference
    destination = psi_dir("PSI-115") / "mesh_axes.csv"
    pd.DataFrame(
        {
            "axis": ["x"] * len(axis_x) + ["y"] * len(axis_y),
            "index": list(range(len(axis_x))) + list(range(len(axis_y))),
            "coordinate": np.concatenate([axis_x, axis_y]),
        }
    ).to_csv(destination, index=False)

    size_kb = destination.stat().st_size / 1024
    print(f"{len(candidates)} fichiers de maillage lus, tous identiques")
    print(f"  ecrit : {destination.relative_to(psi_dir('PSI-115').parents[1])}  ({size_kb:.0f} Ko)")
    print(f"\n  axe x : {len(axis_x)} points, {axis_x.min():.3f} -> {axis_x.max():.3f}")
    print(f"  axe y : {len(axis_y)} points, {axis_y.min():.3f} -> {axis_y.max():.3f}")
    steps_x, steps_y = np.diff(axis_x), np.diff(axis_y)
    print(
        f"\n  maillage etire : facteur {steps_x.max() / steps_x.min():.1f} en x, "
        f"{steps_y.max() / steps_y.min():.1f} en y"
    )
    print("  -> une statistique calculee en cellules est fausse, il faut les coordonnees")
    print("\n  y symetrique autour de zero : axe de revolution en y = 0")
    print("  unites normalisees, pas metriques — echelle absolue inconnue")


if __name__ == "__main__":
    main()
