"""À quel diamètre la flamme lâche-t-elle ? — régression sur PSI-69.

CE QUE CE MODULE AJOUTE À LA CLASSIFICATION.

Le module Suppression répond « la flamme va-t-elle s'éteindre ». Celui-ci
répond à la question suivante : **si elle s'éteint, à quelle taille de
gouttelette ?** Les deux se complètent et s'affichent ensemble —

    probabilite d'extinction   72 %
    si extinction, diametre    1.8 mm

Ce n'est pas une fuite de données. `dext_mm` est mesuré à la fin de l'essai,
mais on le PRÉDIT à partir des seules conditions connues avant l'allumage. On
prévoit un résultat, on ne le recopie pas.

POURQUOI SEULEMENT LES EXTINCTIONS RÉELLES.

172 essais portent un `dext_mm`, mais ils recouvrent deux événements
physiques différents :

    Extinction   159 essais, dext de 0.69 a 4.60 mm, median 1.96
                 la flamme s'est eteinte a cette taille
    Completion    14 essais, dext de 0.00 a 3.04 mm, median 0.75
                 dont 4 a exactement zero

Sur une combustion complète, la gouttelette a été CONSOMMÉE : la flamme ne
s'est pas éteinte, elle n'avait plus de carburant. Appeler « diamètre
d'extinction » la taille finale d'une goutte épuisée mélange deux phénomènes.
Le modèle n'apprend donc que sur les 158 extinctions réelles ayant toutes
leurs conditions, et la question qu'il répond reste celle que son nom annonce.

LE MODÈLE RETENU, ENCORE UNE FOIS PAR LA MESURE.

    plancher, toujours la moyenne   R2 -0.06   erreur moyenne 0.79 mm
    ridge (lineaire)                R2 +0.752  erreur moyenne 0.35 mm
    foret 300 arbres                R2 +0.749  erreur moyenne 0.36 mm

La forêt n'apporte rien ici, contrairement à PSI-159 où elle gagnait douze
points. À égalité de performance, le modèle linéaire l'emporte puisqu'il
s'explique. Troisième jeu de données, troisième réponse : le choix du modèle
se mesure, il ne se décrète pas.

UNE PART DU RÉSULTAT EST TRIVIALE, ET IL FAUT LE DIRE. Le diamètre initial
corrèle à +0.73 avec le diamètre d'extinction, et le rapport `dext/d0` vaut
0.70 en moyenne. Autrement dit, « la flamme lâche à 70 % du diamètre de
départ » est déjà une approximation grossière mais honnête. Le modèle sert à
corriger cette règle en fonction du mélange — c'est cette correction, et non
le R2 brut, qui mesure ce qu'il apporte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.loaders.psi69 import FEATURES, load

RANDOM_STATE = 0
FOLDS = 5
TARGET = "dext_mm"


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Les extinctions réelles, avec toutes leurs conditions."""
    df = load()
    usable = df[
        (df["test_end"] == "Extinction")
        & df[TARGET].notna()
        & df[FEATURES].notna().all(axis=1)
    ]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[TARGET]


def model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge())])


def typical_error() -> float:
    """Erreur moyenne en validation croisée, à afficher avec la prédiction.

    Une prédiction sans son incertitude se lit comme une certitude.
    """
    X, y = prepare()
    scores = cross_val_score(
        model(),
        X,
        y,
        cv=KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error",
    )
    return float(-scores.mean())


def main() -> None:
    X, y = prepare()
    cv = KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 74)
    print("DIAMETRE D'EXTINCTION — PSI-69 / FLEX-1")
    print("=" * 74)
    print(f"\n{len(X)} extinctions reelles")
    print(f"  dext : {y.min():.2f} a {y.max():.2f} mm, ecart-type {y.std():.2f}")

    print("\n" + "-" * 74)
    print("1. LE PLANCHER, ET CE QU'IL VAUT")
    print("-" * 74)
    baseline = cross_val_score(
        DummyRegressor(strategy="mean"), X, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    print(
        f"\n  Repondre toujours la moyenne ({y.mean():.2f} mm) se trompe en"
        f"\n  moyenne de {-baseline.mean():.2f} mm. C'est le seuil a battre."
    )

    print("\n" + "-" * 74)
    print("2. LE MODELE")
    print("-" * 74)
    r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(
        f"\n  ridge (lineaire)   R2 {r2.mean():+.3f}   erreur moyenne "
        f"{mae.mean():.2f} mm"
        f"\n                     plis : {' '.join(f'{v:.2f}' for v in r2)}"
    )

    print("\n" + "-" * 74)
    print("3. CE QUE LE MODELE AJOUTE A UNE REGLE TRIVIALE")
    print("-" * 74)
    df = load()
    usable = df[
        (df["test_end"] == "Extinction")
        & df[TARGET].notna()
        & df[FEATURES].notna().all(axis=1)
    ]
    ratio = usable[TARGET] / usable["d0_mm"]
    naive_error = float((usable[TARGET] - ratio.mean() * usable["d0_mm"]).abs().mean())
    print(
        f"\n  correlation d0 <-> dext : {usable['d0_mm'].corr(usable[TARGET]):+.2f}"
        f"\n  rapport dext/d0         : {ratio.mean():.2f} en moyenne, "
        f"ecart-type {ratio.std():.2f}"
        f"\n\n  Regle triviale « dext = {ratio.mean():.2f} x d0 » : erreur "
        f"{naive_error:.2f} mm"
        f"\n  Modele complet                          : erreur {mae.mean():.2f} mm"
        f"\n\n  Le modele gagne {naive_error - mae.mean():.2f} mm sur la regle de trois."
        f"\n  C'est la correction apportee par le melange — et c'est la vraie"
        f"\n  mesure de son apport, pas le R2 brut, qui beneficie deja de d0."
    )

    print("\n" + "-" * 74)
    print("4. LES COEFFICIENTS")
    print("-" * 74)
    fitted = model().fit(X, y)
    coefficients = pd.Series(
        fitted.named_steps["model"].coef_, index=X.columns
    ).sort_values(key=abs)
    print("\n  Positif = la flamme lache a un diametre PLUS GRAND,")
    print("  c'est-a-dire plus tot dans la combustion.\n")
    for name, value in coefficients.items():
        bar = "#" * int(min(abs(value) * 26, 26))
        print(f"    {name:18} {value:+7.3f}  {bar}")

    print("\n" + "=" * 74)
    print(
        "BILAN\n"
        f"\n  Prediction utilisable : erreur moyenne {mae.mean():.2f} mm sur une"
        f"\n  plage de {y.min():.2f} a {y.max():.2f} mm."
        "\n\n  A afficher TOUJOURS avec son incertitude, et toujours a cote de la"
        "\n  probabilite d'extinction : un diametre d'extinction n'a de sens que"
        "\n  si la flamme s'eteint."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
