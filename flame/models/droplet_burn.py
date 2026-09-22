"""L'histoire de la combustion d'une gouttelette — la loi en d².

CORRECTION D'UNE AFFIRMATION ERRONÉE (2026-09-23).

Il a été écrit deux fois dans ce projet qu'on ne pouvait pas animer le
rétrécissement de la gouttelette, faute de série temporelle : PSI-69 ne relève
que deux instants, la taille au départ et la taille à l'extinction. C'était
faux, et l'erreur venait d'avoir négligé une colonne.

`burning_rate_mm2_s` est la constante K de la LOI EN d², le résultat fondateur
de la combustion de gouttelettes :

    d²(t) = d₀² − K·t

Autrement dit, le jeu de données ne contient pas deux instants isolés : il
contient les deux bornes ET la loi qui les relie. La trajectoire n'est donc
pas une interpolation inventée, c'est de la physique mesurée.

LA LOI EST VÉRIFIÉE SUR CES DONNÉES, PAS SUPPOSÉE.

Si elle tient, la durée de combustion doit valoir (d₀² − dext²) / K. Confronté
aux durées réellement mesurées sur les 158 extinctions complètes :

    correlation predite <-> mesuree     +0.995
    erreur mediane                      0.50 s, soit 6 % de la duree
    essais a moins de 20 % pres         146/158  (92 %)
    rapport median predit/mesure        0.95

Le rapport de 0.95 n'est pas du bruit, il est lui aussi physique : la loi en d²
décrit le régime quasi-stationnaire, et une gouttelette passe d'abord par une
phase de chauffage avant d'y entrer. La combustion réelle dure donc un peu plus
longtemps que la loi pure ne le prévoit. C'est ce qu'annoncent les manuels.

PRÉDIRE K POUR UNE CONDITION QUELCONQUE.

    plancher, toujours la moyenne   R2 -0.02   erreur 0.102 mm²/s
    ridge lineaire                  R2 +0.643  erreur 0.051 mm²/s
    foret 300 arbres                R2 +0.607  erreur 0.054 mm²/s

Le linéaire l'emporte encore, à 11 % d'erreur sur la médiane. C'est suffisant
pour une trajectoire dont le rythme est indicatif ; ça ne le serait pas pour
une mesure. La durée affichée porte donc son incertitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.loaders.psi69 import FEATURES, load

RANDOM_STATE = 0
FOLDS = 5
TARGET = "burning_rate_mm2_s"


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Tous les essais dont la constante de combustion est mesurée.

    Contrairement au diamètre d'extinction, K se mesure aussi sur les
    combustions complètes : c'est la vitesse à laquelle la goutte se consume,
    qu'elle finisse par s'éteindre ou par s'épuiser.
    """
    df = load()
    usable = df[df[TARGET].notna() & df[FEATURES].notna().all(axis=1)]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[TARGET]


def model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge())])


def typical_error() -> float:
    X, y = prepare()
    scores = cross_val_score(
        model(),
        X,
        y,
        cv=KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error",
    )
    return float(-scores.mean())


def burn_duration(initial_mm: float, final_mm: float, rate_mm2_s: float) -> float:
    """Durée de combustion selon la loi en d², en secondes."""
    if rate_mm2_s <= 0:
        return 0.0
    return max((initial_mm**2 - final_mm**2) / rate_mm2_s, 0.0)


def trajectory(
    initial_mm: float, final_mm: float, rate_mm2_s: float, steps: int = 120
) -> pd.DataFrame:
    """Diamètre en fonction du temps, du départ à l'extinction.

    C'est la loi en d² appliquée, rien de plus : d(t) = sqrt(d₀² − K·t).
    Aucun lissage, aucune interpolation entre deux points mesurés.
    """
    duration = burn_duration(initial_mm, final_mm, rate_mm2_s)
    time = np.linspace(0.0, duration, steps)
    squared = np.clip(initial_mm**2 - rate_mm2_s * time, final_mm**2, None)
    return pd.DataFrame({"time_s": time, "diameter_mm": np.sqrt(squared)})


def validate_law() -> pd.DataFrame:
    """Confronte la loi aux durées réellement mesurées."""
    df = load()
    usable = df[
        (df["test_end"] == "Extinction")
        & df[["d0_mm", "dext_mm", TARGET, "burn_time_s"]].notna().all(axis=1)
    ].copy()
    usable["duration_predicted"] = (
        usable["d0_mm"] ** 2 - usable["dext_mm"] ** 2
    ) / usable[TARGET]
    usable["ratio"] = usable["duration_predicted"] / usable["burn_time_s"]
    return usable


def main() -> None:
    print("=" * 74)
    print("LOI EN d² — PSI-69 / FLEX-1")
    print("=" * 74)

    print("\n" + "-" * 74)
    print("1. LA LOI TIENT-ELLE SUR CES DONNEES ?")
    print("-" * 74)
    checked = validate_law()
    predicted, observed = checked["duration_predicted"], checked["burn_time_s"]
    error = (predicted - observed).abs()
    within = ((checked["ratio"] - 1).abs() < 0.20).mean()
    print(
        f"\n  {len(checked)} extinctions avec d0, dext, K et duree mesures."
        f"\n  Si d²(t) = d0² - K t, la duree vaut (d0² - dext²) / K.\n"
        f"\n    correlation predite <-> mesuree   {predicted.corr(observed):+.3f}"
        f"\n    erreur mediane                    {error.median():.2f} s "
        f"({100 * (error / observed).median():.0f} % de la duree)"
        f"\n    essais a moins de 20 % pres       "
        f"{int(within * len(checked))}/{len(checked)}  ({within:.0%})"
        f"\n    rapport median predit/mesure      {checked['ratio'].median():.2f}"
    )
    print(
        "\n  Le rapport de 0.95 n'est pas du bruit. La loi decrit le regime"
        "\n  quasi-stationnaire ; une gouttelette passe d'abord par une phase de"
        "\n  chauffage avant d'y entrer, donc la combustion reelle dure un peu"
        "\n  plus longtemps que la loi pure. C'est attendu."
    )

    print("\n" + "-" * 74)
    print("2. PREDIRE K POUR UNE CONDITION QUELCONQUE")
    print("-" * 74)
    X, y = prepare()
    cv = KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(
        f"\n  {len(X)} essais avec K mesure, de {y.min():.3f} a {y.max():.3f} mm²/s"
        f"\n  ridge lineaire : R2 {r2.mean():+.3f}, erreur {mae.mean():.3f} mm²/s "
        f"({100 * mae.mean() / y.median():.0f} % de la mediane)"
    )

    print("\n" + "-" * 74)
    print("3. UNE TRAJECTOIRE, POUR VERIFIER A L'OEIL")
    print("-" * 74)
    example = checked.iloc[len(checked) // 2]
    path = trajectory(
        example["d0_mm"], example["dext_mm"], example[TARGET], steps=9
    )
    print(
        f"\n  essai reel : {example['fuel']}, O2 {example['o2_frac']:.2f}, "
        f"CO2 {example['co2_frac']:.2f}"
        f"\n  d0 {example['d0_mm']:.2f} mm, dext {example['dext_mm']:.2f} mm, "
        f"K {example[TARGET]:.3f} mm²/s"
        f"\n  duree mesuree {example['burn_time_s']:.1f} s, "
        f"loi {example['duration_predicted']:.1f} s\n"
    )
    print("     t (s)   diametre (mm)")
    for _, step in path.iterrows():
        bar = "#" * int(step["diameter_mm"] / example["d0_mm"] * 34)
        print(f"    {step['time_s']:6.2f}   {step['diameter_mm']:8.2f}  {bar}")

    print("\n" + "=" * 74)
    print(
        "BILAN\n"
        "\n  La trajectoire n'est pas une interpolation entre deux points mesures."
        "\n  C'est une loi physique, dont la constante est relevee essai par essai"
        "\n  et dont l'accord avec les durees mesurees vaut 0.995 de correlation."
        "\n  L'animer est donc legitime — ce qui ne l'aurait pas ete, c'est de"
        "\n  faire varier la taille au jugé entre les deux bornes."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
