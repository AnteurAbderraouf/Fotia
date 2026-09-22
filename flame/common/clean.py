"""Outils de nettoyage partagés par tous les loaders.

Les fichiers NASA viennent d'équipes, d'époques et de tableurs différents, mais
les mêmes défauts reviennent partout : des nombres stockés en texte, des unités
collées à la valeur, des marqueurs de valeur manquante maison, des espaces
parasites en fin de chaîne. Ces fonctions traitent ces cas une fois pour toutes
plutôt qu'à neuf endroits légèrement différents.
"""

from __future__ import annotations

import re

import pandas as pd

# Marqueurs de valeur absente rencontrés dans les fichiers NASA. Le tiret
# cadratin (U+2013) est écrit en échappement pour qu'il survive à n'importe
# quelle manipulation du fichier source.
MISSING_TOKENS = {
    "\u2013",  # – tiret demi-cadratin (PSI-69)
    "\u2014",  # — tiret cadratin
    "-",
    "--",
    "",
    "n/a",
    "na",
    "nan",
    "none",
    "?",
    # Erreurs de formule Excel figees dans les donnees exportees. Elles ne
    # signalent pas une valeur absente mais un calcul impossible : division
    # par zero sur une cellule vide, reference cassee. Le resultat est le
    # meme pour nous, il n'y a pas de mesure. Rencontrees dans PSI-159.
    "#div/0!",
    "#value!",
    "#ref!",
    "#n/a",
    "#name?",
    "#null!",
    "#num!",
}


def as_text(series: pd.Series) -> pd.Series:
    """Série en texte, espaces de bord retirés, marqueurs d'absence en NaN."""
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\s+", " ", regex=True)
    return text.mask(text.str.lower().isin(MISSING_TOKENS))


def to_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convertit une colonne texte en numérique.

    Renvoie aussi un masque des valeurs préfixées de « ~ », que NASA utilise
    pour signaler une mesure approximative. On ne jette pas cette information :
    la valeur est gardée, et le fait qu'elle soit approximative aussi.
    """
    text = as_text(series)
    approx = text.str.startswith("~").fillna(False)
    return pd.to_numeric(text.str.lstrip("~"), errors="coerce"), approx


def strip_unit(series: pd.Series) -> pd.Series:
    """Extrait le nombre d'une valeur où l'unité est collée au texte.

    Gère « 12 w », « 12000ms », « 3 atm », « .5 atm » — la casse, l'espace
    manquant et le zéro initial omis ne changent rien au résultat.
    """
    text = as_text(series)
    number = text.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(number, errors="coerce")


def check_fractions_sum_to_one(
    df: pd.DataFrame, columns: list[str], tolerance: float = 0.05
) -> None:
    """Vérifie que des fractions molaires couvrent bien tout le mélange.

    Sert à deux choses. D'abord confirmer qu'on a identifié tous les gaz —
    c'est ce test qui a prouvé que la colonne « CO » de PSI-69 était du CO2 et
    non du monoxyde. Ensuite rappeler que des colonnes qui somment à 1 sont
    linéairement dépendantes : les donner toutes à une régression rend ses
    coefficients instables, il faut en retirer une.

    Lève une erreur si la somme s'écarte de 1 au-delà de la tolérance, parce
    qu'un tel écart signifie qu'on a mal compris la composition du mélange.
    """
    total = df[columns].sum(axis=1)
    off = total[(total - 1.0).abs() > tolerance]
    if len(off):
        raise ValueError(
            f"{len(off)} lignes dont les fractions {columns} ne somment pas à 1 "
            f"(observé {off.min():.3f} à {off.max():.3f}). "
            "La composition du mélange est mal comprise."
        )


def add_provenance(
    df: pd.DataFrame,
    investigation: str,
    gravity: str | None,
    source: str | None = "experiment",
) -> pd.DataFrame:
    """Marque l'origine de chaque ligne.

    Obligatoire sur toute table de ce projet. `gravity` en particulier empêche
    qu'une mise en commun ultérieure mélange silencieusement des essais au sol
    (PSI-142, brûleur à contre-courant) avec des essais ISS — ce sont deux
    régimes physiques différents, pas deux échantillons du même.
    """
    out = df.assign(investigation=investigation)

    # `source` et `gravity` peuvent deja exister et varier d'une ligne a
    # l'autre. PSI-117 melange mesures et simulations dans un meme fichier ;
    # PSI-115 compare 1g et microgravite, la gravite y est la variable etudiee
    # et non une constante. Passer None preserve la colonne au lieu de
    # l'aplatir sur une valeur unique, ce qui effacerait justement la
    # distinction qui fait la valeur de ces tables.
    for name, value in (("source", source), ("gravity", gravity)):
        if value is not None:
            out[name] = value
        elif name not in out.columns:
            raise ValueError(
                f"{name}=None suppose une colonne « {name} » deja presente "
                "dans la table"
            )
    return out
