"""Module « Détection » — PSI-101 / SAME-R, ce que voient les détecteurs ISS.

Des matériaux réellement embarqués sont chauffés jusqu'à émettre de la fumée,
et l'on mesure à la fois ce que la fumée EST physiquement et ce que les
détecteurs de bord en PERÇOIVENT. L'écart entre les deux est la question de
sécurité.

UNE RÉGRESSION, PAS UNE ALARME. Aucune colonne du classeur n'enregistre un
déclenchement de détecteur. Ce qui existe est la réponse en volts, continue.
Fabriquer un seuil pour en faire un oui/non reviendrait à inventer
l'étiquette.

DEUX CANAUX, DEUX NATURES DIFFÉRENTES — et c'est le cœur du module.

    diffusion       134 valeurs, de 0.001 a 6.29 V, continue.
                    Se modelise.

    obscurcissement 134 valeurs, dont 109 EXACTEMENT NULLES (81 %).
                    Ce n'est pas une grandeur continue, c'est un canal qui
                    ne repond presque jamais.

Traiter l'obscurcissement comme une régression serait un contresens : on
ajusterait une droite sur une colonne de zéros. La question honnête est
binaire — le canal a-t-il répondu ? — et la réponse est non dans quatre cas
sur cinq. Pour le Pyrell, jamais : vingt essais sur vingt à zéro.

LE RÉSULTAT DE SÉCURITÉ, LISIBLE SANS MODÈLE.

    materiau    diffusion V   particules/cc
    Teflon             1.4          48 855
    Kapton             2.4          37 590
    Silicone           6.2          43 984
    Lampwick           6.2          42 954
    Pyrell             6.0          45 804

Le Teflon produit le PLUS de particules et donne le MOINS de signal. Le Kapton
suit. Une fumée composée de fines particules échappe largement à un détecteur
à diffusion, qui répond à la taille bien plus qu'au nombre. Deux des cinq
matériaux testés sont donc mal vus par le capteur censé les détecter.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.loaders.psi101 import FEATURES, load

RANDOM_STATE = 0
FOLDS = 5
SCATTER = "iss_scatter_volts"
OBSCURATION = "iss_obscuration_volts"


def prepare(target: str = SCATTER) -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    usable = df[df[target].notna() & df[FEATURES].notna().all(axis=1)]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[target]


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
    X, y = prepare(SCATTER)
    cv = KFold(FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 74)
    print("MODULE DETECTION — PSI-101 / SAME-R")
    print("=" * 74)
    print(f"\n{len(X)} essais sur materiaux reels, {X.shape[1]} colonnes")
    print(f"  signal de diffusion ISS : {y.min():.3f} a {y.max():.2f} V")

    print("\n" + "-" * 74)
    print("1. LE CANAL DE DIFFUSION — ce qui se modelise")
    print("-" * 74)
    baseline = -cross_val_score(
        DummyRegressor(strategy="mean"), X, y, cv=cv,
        scoring="neg_mean_absolute_error",
    ).mean()
    r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    forest = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    forest_r2 = cross_val_score(forest, X, y, cv=cv, scoring="r2")
    print(
        f"\n    plancher (toujours la moyenne)   erreur {baseline:.2f} V"
        f"\n    ridge                            R2 {r2.mean():+.3f}, "
        f"erreur {mae.mean():.2f} V"
        f"\n    foret 300 arbres                 R2 {forest_r2.mean():+.3f}"
        f"\n\n  La foret n'apporte rien : le lineaire suffit et il s'explique."
        f"\n  Quatrieme jeu de donnees, quatrieme reponse sur le choix du modele."
    )

    print("\n" + "-" * 74)
    print("2. LE CANAL D'OBSCURCISSEMENT — pourquoi on ne le modelise PAS")
    print("-" * 74)
    df = load()
    obscuration = df[OBSCURATION].dropna()
    zeros = int((obscuration == 0).sum())
    print(
        f"\n  {len(obscuration)} valeurs, dont {zeros} exactement nulles "
        f"({100 * zeros / len(obscuration):.0f} %)."
        "\n\n  Ce n'est pas une grandeur continue, c'est un canal qui ne repond"
        "\n  presque jamais. Ajuster une regression dessus reviendrait a tracer"
        "\n  une droite dans une colonne de zeros. La question honnete est"
        "\n  binaire : le canal a-t-il repondu ?\n"
    )
    responded = (
        df.assign(repondu=df[OBSCURATION] > 0)
        .groupby("material")["repondu"]
        .agg(["size", "sum"])
        .rename(columns={"size": "essais", "sum": "reponses"})
    )
    responded["taux"] = (responded["reponses"] / responded["essais"]).map("{:.0%}".format)
    print(responded.to_string())
    print(
        "\n  Le canal n'a JAMAIS repondu au Pyrell : vingt essais sur vingt a zero."
    )

    print("\n" + "-" * 74)
    print("3. LE RESULTAT DE SECURITE, VISIBLE SANS MODELE")
    print("-" * 74)
    summary = (
        df.groupby("material")[[SCATTER, "ptrak_particles_cc", "dusttrak_b_mg_m3"]]
        .mean()
        .round(1)
        .sort_values(SCATTER)
    )
    summary.columns = ["diffusion V", "particules/cc", "masse mg/m3"]
    print("\n" + summary.to_string())
    print(
        "\n  Le Teflon produit le PLUS de particules et donne le MOINS de signal."
        "\n  Le Kapton suit. Un detecteur a diffusion repond a la TAILLE des"
        "\n  particules bien plus qu'a leur nombre : une fumee fine lui echappe."
        "\n\n  Deux des cinq materiaux testes sont donc mal vus par le capteur"
        "\n  cense les detecter. C'est un constat de securite, et il ne demande"
        "\n  aucun modele."
    )

    print("\n" + "-" * 74)
    print("4. LES COEFFICIENTS")
    print("-" * 74)
    fitted = model().fit(X, y)
    coefficients = pd.Series(
        fitted.named_steps["model"].coef_, index=X.columns
    ).sort_values(key=abs)
    print("\n  Positif = le detecteur voit DAVANTAGE.\n")
    for name, value in coefficients.items():
        bar = "#" * int(min(abs(value) * 9, 26))
        print(f"    {name:26} {value:+7.3f}  {bar}")

    print("\n" + "=" * 74)
    print(
        f"BILAN\n"
        f"\n  diffusion        R2 {r2.mean():+.3f}, erreur {mae.mean():.2f} V "
        f"sur une plage de 0 a 6.3"
        f"\n  obscurcissement  non modelise : {100 * zeros / len(obscuration):.0f} % "
        f"de zeros, la question est binaire"
        f"\n\n  Le module repond « combien de volts », jamais « l'alarme sonne »."
        f"\n  Aucune colonne du classeur n'enregistre un declenchement."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
