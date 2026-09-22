"""PSI-115 — champs CFD de fumée, gravité contre microgravité.

Ce n'est pas un jeu de données d'entraînement et ça ne le deviendra pas : cinq
cas simulés, chacun décrit par des grilles de 240 x 960 cellules. Cinq lignes
et deux cent trente mille colonnes, autrement dit tout l'inverse de ce qu'un
modèle peut apprendre.

Sa valeur est ailleurs, et elle est grande. C'est la seule source du catalogue
qui montre la même configuration AVEC et SANS gravité, tout le reste égal.
Deux usages :

  1. Des cartes de champ côte à côte dans le dashboard. Visuellement, c'est
     l'élément le plus parlant du projet : le panache s'élève sur Terre, il
     s'étale en microgravité.
  2. La réduction que fait ce loader — chaque grille ramenée à quelques
     scalaires, pour une table de comparaison lisible.

Source (lecture seule) :
    combustion_science/ground_investigation/PSI-115/fields/  (42 grilles)
    Produites par scripts/convert_dat.py puis scripts/clean_dat_csv.py.

LES CINQ CAS :
    8cm  gravite terrestre
    8cm  microgravite
    8cm  microgravite sans thermophorese  (variante « NoTF »)
    10cm gravite terrestre
    10cm microgravite

CE QUE LES GRILLES NE PORTENT PAS.
  * Pas d'axes physiques. Les deux fichiers de maillage de coordonnées
    (`*_sGrid_grid3d.dat`) n'ont pas pu être lus — lignes de longueur
    inégale, c'est le cas limite connu et non corrigé du §5 du handoff. Les
    étendues calculées ici sont donc exprimées EN CELLULES, pas en
    millimètres. Une carte reste juste, une distance chiffrée ne l'est pas.
  * Les températures sont NORMALISÉES, pas en kelvins : elles vont de 1 à
    1.577, soit un rapport à la température ambiante. Les lire comme des
    degrés serait un contresens.

LES VALEURS NON FINIES SONT UNE INFORMATION, PAS UN TROU.
La grille de taux de nucléation ne contient que 89 924 valeurs finies sur
230 400. Le reste était `-inf` avant nettoyage : le logarithme d'un taux nul,
c'est-à-dire l'absence de nucléation hors du panache. Le handoff insiste et il
a raison — ne jamais remplacer ces cellules par zéro. « Indéfini » et « nul »
ne sont pas la même chose, et la proportion de cellules définies est en
elle-même une mesure de l'étendue du panache. Elle est donc conservée comme
statistique à part entière (`finite_fraction`).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from flame.common.clean import add_provenance
from flame.common.paths import processed_path, psi_dir

FIELDS_DIR = psi_dir("PSI-115") / "fields"

# Fichiers qui ne sont pas des champs 2D.
NOT_A_GRID = {"manifest.csv", "nodeDiameters.csv"}

VARIABLE_LABELS = {
    "temperature": "temperature (normalisee)",
    "smoke": "fraction de fumee",
    "numden": "densite numerique de particules",
    "numdenr": "densite numerique (repere tourne)",
    "nucrate": "taux de nucleation (log)",
    "u": "vitesse axiale",
    "v": "vitesse transverse",
    "ur": "vitesse axiale (repere tourne)",
    "vr": "vitesse transverse (repere tourne)",
    "vort": "vorticite",
    "speciesmf": "fraction massique d'espece",
    "vapormf": "fraction massique de vapeur",
    "dam": "nombre de Damkohler",
}

FILENAME = re.compile(
    r"^(?P<size>\d+)cm(?P<gravity>NoGravity|Gravity)(?P<notf>NoTF_)?(?P<variable>.+)$",
    re.IGNORECASE,
)


def _describe(path) -> dict | None:
    """Décompose un nom de fichier en cas simulé et variable."""
    match = FILENAME.match(path.stem)
    if not match:
        return None
    variable = match.group("variable").lower().lstrip("_")
    return {
        "burner_size_cm": int(match.group("size")),
        "gravity": "1g" if match.group("gravity").lower() == "gravity" else "microgravity",
        "thermophoresis": match.group("notf") is None,
        "variable": variable,
        "variable_label": VARIABLE_LABELS.get(variable, variable),
        "grid_file": path.name,
    }


def _summarise(values: np.ndarray) -> dict:
    """Ramène une grille à des scalaires, sans inventer d'axes physiques."""
    finite = np.isfinite(values)
    result = {
        "cells": int(values.size),
        "finite_fraction": float(finite.mean()),
    }
    if not finite.any():
        return result

    data = values[finite]
    result |= {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }

    # Etendue transverse du champ, mesuree comme l'ecart-type des indices de
    # ligne ponderes par la valeur. Sans seuil a choisir, donc sans convention
    # arbitraire — mais exprimee EN CELLULES, faute d'axes physiques.
    weights = np.where(finite, values, 0.0)
    weights = np.clip(weights - np.nanmin(data), 0, None)
    total = weights.sum()
    if total > 0:
        rows = np.arange(values.shape[0])[:, None]
        centre = float((weights * rows).sum() / total)
        spread = float(np.sqrt((weights * (rows - centre) ** 2).sum() / total))
        result |= {"transverse_centre_cells": centre, "transverse_spread_cells": spread}
    return result


def load() -> pd.DataFrame:
    """Une ligne par grille : le cas simulé, la variable, et ses scalaires."""
    rows = []
    for path in sorted(FIELDS_DIR.glob("*.csv")):
        if path.name in NOT_A_GRID:
            continue
        described = _describe(path)
        if described is None:
            continue

        frame = pd.read_csv(path, header=None)
        # Les fichiers PSD ne sont pas des champs mais des distributions
        # plates, et portent un en-tete texte « value ».
        if str(frame.iat[0, 0]).strip().lower() == "value":
            continue
        values = frame.to_numpy(dtype=float)
        if values.ndim != 2 or min(values.shape) < 2:
            continue

        rows.append(described | {"rows": values.shape[0], "cols": values.shape[1]} | _summarise(values))

    df = pd.DataFrame(rows)
    df["case"] = (
        df["burner_size_cm"].astype(str)
        + "cm / "
        + df["gravity"]
        + np.where(df["thermophoresis"], "", " / sans thermophorese")
    )
    # source=None : la provenance ligne a ligne porte deja 1g ou microgravite,
    # qui est ici la variable comparee, pas une constante.
    df["source"] = "simulation"
    return add_provenance(df, investigation="PSI-115", gravity=None, source=None)


def main() -> None:
    df = load()
    path = processed_path("psi115_field_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"{path.name} : {len(df)} grilles reduites\n")

    print("cas simules :")
    print(df.groupby("case")["variable"].count().rename("grilles").to_string())

    print("\ncomparaison gravite / microgravite, bruleur 8 cm :")
    view = df[(df["burner_size_cm"] == 8) & df["thermophoresis"]]
    pivot = view.pivot_table(
        index="variable_label", columns="gravity", values="max"
    ).round(3)
    print(pivot.to_string())

    print("\netendue transverse du panache (en cellules, pas en mm) :")
    spread = df[df["variable"].isin(["smoke", "temperature", "numden"])].pivot_table(
        index=["variable_label", "burner_size_cm"],
        columns="gravity",
        values="transverse_spread_cells",
    ).round(1)
    print(spread.to_string())

    print("\ncellules definies, la ou l'absence est une information :")
    sparse = df[df["finite_fraction"] < 0.999][
        ["grid_file", "finite_fraction"]
    ].sort_values("finite_fraction")
    for _, row in sparse.iterrows():
        print(f"  {row['grid_file']:34} {row['finite_fraction']:.1%} de cellules definies")
    print("  -> hors panache, le log d'un taux nul est indefini, PAS zero.")


if __name__ == "__main__":
    main()
