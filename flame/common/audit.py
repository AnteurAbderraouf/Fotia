"""Audit standard d'une table nettoyée.

Chaque loader produit une table censée respecter les mêmes règles. Plutôt que
de les revérifier à la main à chaque fois — ce qu'on finit toujours par bâcler
— cette fonction les contrôle systématiquement et affiche ce qu'elle trouve.

Les règles contrôlées :
  1. Les colonnes annoncées comme features et comme mesures existent vraiment.
  2. Aucune mesure post-expérience ne s'est glissée dans les features.
     C'est la fuite de données la plus courante et aucun test statistique ne
     la détecte : il faut comparer deux listes déclarées.
  3. L'étiquette est binaire, sans valeur manquante, et son plancher
     majoritaire est affiché — c'est le score qu'un modèle doit battre pour
     avoir appris quoi que ce soit.
  4. Les colonnes numériques le sont réellement, sans texte résiduel.
  5. La provenance est présente et sans ambiguïté.
  6. Les trous restants dans les features sont listés, puisqu'ils imposeront
     un choix (imputer ou écarter) au moment de la modélisation.
"""

from __future__ import annotations

import pandas as pd

PROVENANCE = ["investigation", "source", "gravity"]


def audit(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    post_burn: list[str] | None = None,
    name: str = "table",
) -> list[str]:
    """Affiche l'audit et renvoie la liste des problèmes détectés."""
    post_burn = post_burn or []
    problems: list[str] = []

    print(f"\n{'=' * 66}\n{name}   {df.shape[0]} lignes x {df.shape[1]} colonnes\n{'=' * 66}")

    missing_cols = [c for c in features + [label] if c not in df.columns]
    if missing_cols:
        problems.append(f"colonnes annoncees absentes : {missing_cols}")

    overlap = set(features) & set(post_burn)
    if overlap:
        problems.append(f"FUITE : mesures post-experience dans les features : {overlap}")

    if label in df.columns:
        values = sorted(df[label].dropna().unique().tolist())
        n_missing = int(df[label].isna().sum())
        print(f"\netiquette « {label} » : valeurs {values}, {n_missing} manquantes")
        if n_missing:
            problems.append(f"{n_missing} etiquettes manquantes")
        if set(values) - {0, 1}:
            problems.append(f"etiquette non binaire : {values}")
        counts = df[label].value_counts().sort_index()
        total = int(counts.sum())
        if total:
            for value, count in counts.items():
                print(f"    {value} : {count:4d}  ({count / total:5.1%})")
            print(f"    plancher majoritaire a battre : {counts.max() / total:.0%}")

    print("\nfeatures :")
    for col in features:
        if col not in df.columns:
            continue
        series = df[col]
        kind = "num" if pd.api.types.is_numeric_dtype(series) else "cat"
        gaps = int(series.isna().sum())
        if kind == "num":
            detail = f"{series.min():g} -> {series.max():g}"
        else:
            detail = f"{series.nunique()} valeurs : {sorted(series.dropna().unique())[:4]}"
        flag = f"  <- {gaps} trous" if gaps else ""
        print(f"    {col:22} {kind}  {detail}{flag}")

    leftover_text = [
        c
        for c in features
        if c in df.columns
        and not pd.api.types.is_numeric_dtype(df[c])
        and df[c].dropna().map(lambda v: isinstance(v, str)).all()
        and df[c].nunique() > 20
    ]
    if leftover_text:
        problems.append(
            f"features texte a nombreuses modalites (texte non normalise ?) : {leftover_text}"
        )

    present = [c for c in PROVENANCE if c in df.columns]
    if len(present) < len(PROVENANCE):
        problems.append(f"provenance incomplete : manque {set(PROVENANCE) - set(present)}")
    else:
        values = {c: df[c].dropna().unique().tolist() for c in PROVENANCE}
        print(f"\nprovenance : {values}")
        for col, vals in values.items():
            if len(vals) != 1:
                problems.append(f"provenance « {col} » non unique : {vals}")

    if problems:
        print("\nPROBLEMES :")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\naucun probleme detecte")

    return problems
