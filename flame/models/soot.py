"""Module « Suie » — PSI-107 / SPICE, point de fumée.

Le point de fumée est la longueur qu'atteint une flamme de diffusion juste
avant de commencer à émettre de la suie. Plus il est court, plus le carburant
encrasse. C'est une mesure directe de production de fumée, donc de ce qui
obscurcit une cabine et déclenche, ou non, un détecteur.

UNE FUITE DE DONNÉES TROUVÉE ET CORRIGÉE (2026-09-23).

Ce module donnait d'abord un R² de 0.984, ce qui aurait dû alerter tout de
suite. Les débits de carburant figuraient parmi les entrées, et les en-têtes
NASA disaient pourtant clairement ce qu'ils étaient :

    colonne 25   « S.P. Total Fuel Flow (SCCM) »
    colonne 28   « S.P. HC Mass Flow (mg/s) »

« S.P. » signifie Smoke Point. Ce ne sont pas des consignes : ce sont les
débits AU MOMENT où la suie apparaît. L'équipage montait le débit jusqu'à
voir la flamme fumer, puis relevait le débit ET la longueur de flamme. Les
deux décrivent le même instant, et la corrélation entre le débit et la cible
valait +0.975.

Le modèle prédisait donc le point de fumée à partir du point de fumée.

    avec les debits    R2 0.984, erreur 1.8 mm     <- sans valeur
    sans les debits    R2 0.540, erreur 9.0 mm     <- ce qu'on sait vraiment

C'est le second chiffre qui décrit ce que le modèle sait faire. Le premier
décrivait la capacité de l'arithmétique à recopier une colonne.

CE QUI RESTE, ET CE QUE ÇA VAUT. Quatre conditions réellement réglées avant
l'essai : le carburant, sa dilution, le diamètre de la buse et la vitesse du
co-courant d'air. Le plancher se trompe de 16 mm sur des longueurs allant de
13 à 105 mm ; le modèle de 9 mm. Un gain réel, modeste, et honnête.

SOIXANTE-DIX ESSAIS. Avec quatre entrées, cela reste dans la règle des dix à
cinquante lignes par variable, mais tout juste. Les intervalles sont larges et
l'affichage doit le dire.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.loaders.psi107 import FEATURES, OUTCOME, POST_BURN, load

RANDOM_STATE = 0
FOLDS = 5


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    usable = df[df[OUTCOME].notna() & df[FEATURES].notna().all(axis=1)]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[OUTCOME]


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


def main() -> None:
    X, y = prepare()
    cv = KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 74)
    print("MODULE SUIE — PSI-107 / SPICE")
    print("=" * 74)
    print(f"\n{len(X)} essais SPICE, {X.shape[1]} colonnes apres encodage")
    print(
        f"  point de fumee : {y.min():.1f} a {y.max():.1f} mm, "
        f"mediane {y.median():.1f}, ecart-type {y.std():.1f}"
    )
    print("  court = le carburant fume beaucoup")

    print("\n" + "-" * 74)
    print("1. LE PLANCHER")
    print("-" * 74)
    baseline = -cross_val_score(
        DummyRegressor(strategy="mean"), X, y, cv=cv,
        scoring="neg_mean_absolute_error",
    ).mean()
    print(
        f"\n  Repondre toujours la moyenne ({y.mean():.1f} mm) se trompe de "
        f"{baseline:.1f} mm."
    )

    print("\n" + "-" * 74)
    print("2. LE MODELE")
    print("-" * 74)
    r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(
        f"\n  ridge   R2 {r2.mean():+.3f}   erreur {mae.mean():.1f} mm"
        f"\n          plis : {' '.join(f'{v:+.2f}' for v in r2)}"
        f"\n\n  Gain sur le plancher : {baseline - mae.mean():.1f} mm."
    )

    print("\n" + "-" * 74)
    print("3. LA FUITE QUI AVAIT ETE CORRIGEE")
    print("-" * 74)
    df = load()
    usable = df[df[OUTCOME].notna() & df[FEATURES].notna().all(axis=1)]
    leaky_columns = FEATURES + ["mdot_hc_mg_s", "fuel_jet_velocity_cm_s"]
    complete = df[df[OUTCOME].notna() & df[leaky_columns].notna().all(axis=1)]
    leaky = pd.get_dummies(complete[leaky_columns], drop_first=True).astype(float)
    leaky_r2 = cross_val_score(model(), leaky, complete[OUTCOME], cv=cv, scoring="r2")
    print(
        f"\n  Si l'on remet les debits parmi les entrees : R2 {leaky_r2.mean():+.3f}"
        f"\n  Correlation debit <-> point de fumee : "
        f"{complete['mdot_hc_mg_s'].corr(complete[OUTCOME]):+.3f}"
        "\n\n  Ces debits sont mesures A L'INSTANT du point de fumee — les"
        "\n  en-tetes NASA les nomment « S.P. Fuel Flow ». Les donner en entree"
        "\n  revient a predire le point de fumee a partir de lui-meme."
        "\n\n  Un R2 de 0.98 sur soixante-dix lignes aurait du alerter seul."
    )

    print("\n" + "-" * 74)
    print("4. LES COEFFICIENTS")
    print("-" * 74)
    fitted = model().fit(X, y)
    coefficients = pd.Series(
        fitted.named_steps["model"].coef_, index=X.columns
    ).sort_values(key=abs)
    print("\n  Positif = point de fumee PLUS LONG, donc carburant moins fumigene.\n")
    for name, value in coefficients.items():
        bar = "#" * int(min(abs(value) / 2, 26))
        print(f"    {name:26} {value:+8.2f}  {bar}")

    print("\n" + "-" * 74)
    print("5. CE QUE LES DONNEES DISENT SANS MODELE")
    print("-" * 74)
    by_fuel = (
        usable.groupby("fuel")[OUTCOME].agg(["count", "mean"]).round(1).sort_values("mean")
    )
    print("\n  point de fumee moyen, par carburant :\n")
    print(by_fuel.to_string())
    print(
        "\n  Le propylene fume nettement plus que le propane et l'ethylene."
        "\n  C'est ce qu'on attend d'un alcene face a un alcane, et ca confirme"
        "\n  au passage que les colonnes n'ont pas ete melangees au nettoyage."
    )

    print("\n" + "=" * 74)
    print(
        f"BILAN\n"
        f"\n  plancher   erreur {baseline:.1f} mm"
        f"\n  ridge      erreur {mae.mean():.1f} mm, R2 {r2.mean():+.3f}"
        f"\n\n  Modeste, et c'est le chiffre juste. Les {len(X)} essais et les"
        f"\n  {X.shape[1]} entrees ne permettent pas mieux — mais la version a"
        f"\n  R2 0.98 ne permettait rien du tout."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
