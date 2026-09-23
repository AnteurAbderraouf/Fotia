"""Module « Référence au sol » — PSI-142 / Princeton, à 1 g.

À quelle vitesse d'étirement une flamme de diffusion s'éteint-elle ? Cinq
alcanes, avec ou sans ozone ajouté, sur un brûleur à contre-courant.

CE MODULE EST LE SEUL À 1 g, ET CE N'EST PAS UN DÉTAIL.

Tout le reste du catalogue vient de l'ISS. Ces essais-là sont faits sur Terre,
sur un dispositif qui n'a rien d'une expérience orbitale. La colonne `gravity`
vaut « 1g » sur chaque ligne, et aucune mise en commun avec les modules ISS
n'est licite sans le dire. Ce module existe pour servir de point de
comparaison, pas pour être fondu dans les autres.

L'OZONE, ET UN PIÈGE DE COMPARAISON.

L'ozone sensibilise le mélange et permet à une flamme froide d'exister là où
elle ne tiendrait pas. Mais les deux campagnes NE SE RECOUVRENT PAS : sans
ozone la richesse va de 0.044 à 0.220, avec ozone de 0.234 à 0.555, et la
bande commune est vide.

Comparer les vitesses d'étirement moyennes des deux campagnes n'a donc aucun
sens. Une moyenne plus basse avec ozone ne dit rien de l'ozone, elle dit qu'on
a mesuré ailleurs. Ce que le plan d'expérience montre, c'est que l'ozone
DÉPLACE le domaine où une flamme froide existe, et cela se lit dans les bornes
du tableau, pas dans ses moyennes.

Conséquence pour le modèle : `ozone` est en partie un indicateur de la plage
de richesse. L'importance dominante de `fuel_mass_fraction` (0.79 contre 0.05
pour l'ozone) le confirme.

Le plan de ce projet avait par ailleurs interverti les deux conditions,
corrigé depuis après lecture de l'index rédigé par Princeton.

CINQUIÈME JEU, CINQUIÈME RÉPONSE SUR LE CHOIX DU MODÈLE.

    plancher, toujours la moyenne   R2 -0.089   erreur 20.5 1/s
    ridge (lineaire)                R2 -0.102   erreur 20.3 1/s
    foret 300 arbres                R2 +0.547   erreur 11.6 1/s

La régression linéaire est **moins bonne que de ne rien faire**. C'est le cas
le plus net rencontré jusqu'ici, et il s'explique : la limite d'extinction
dépend du carburant ET de la présence d'ozone ET de la richesse, de façon non
additive. Un alcane lourd sans ozone et un alcane léger avec ozone ne se
déduisent pas d'une somme de contributions. Une droite ne peut pas représenter
ça ; un arbre sépare d'abord sur le carburant, puis sur l'ozone.

Le récapitulatif sur les cinq jeux de données :

    PSI-69   regression 0.753  >  arbre illimite 0.678
    PSI-159  foret 0.840       >  regression 0.722
    PSI-101  regression 0.676  ~  foret 0.653
    PSI-107  regression 0.540  ~  foret 0.514
    PSI-142  foret 0.547       >  regression -0.102

Aucune règle générale ne survit à ces cinq lignes. Le modèle se choisit par la
mesure, jeu par jeu.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.loaders.psi142 import load

RANDOM_STATE = 0
FOLDS = 5
FEATURES = ["fuel", "fuel_mass_fraction", "extinction_type", "ozone"]
TARGET = "extinction_strain_rate_1_s"


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    usable = df[df[TARGET].notna() & df[FEATURES].notna().all(axis=1)]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[TARGET]


def model() -> RandomForestRegressor:
    """La forêt, retenue parce que le linéaire échoue ici, pas par principe."""
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    )


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
    print("MODULE REFERENCE AU SOL — PSI-142 / Princeton")
    print("=" * 74)
    print(f"\n{len(X)} limites d'extinction, {X.shape[1]} colonnes")
    print(f"  vitesse d'etirement : {y.min():.1f} a {y.max():.1f} 1/s")
    print("  GRAVITE TERRESTRE, 1 g. A ne jamais melanger avec les modules ISS.")

    print("\n" + "-" * 74)
    print("1. LE LINEAIRE EST PIRE QUE DE NE RIEN FAIRE")
    print("-" * 74)
    candidates = [
        ("plancher, toujours la moyenne", DummyRegressor(strategy="mean")),
        ("ridge (lineaire)", Pipeline([("s", StandardScaler()), ("m", Ridge())])),
        ("foret 300 arbres", model()),
    ]
    print()
    for name, estimator in candidates:
        r2 = cross_val_score(estimator, X, y, cv=cv, scoring="r2").mean()
        mae = -cross_val_score(
            estimator, X, y, cv=cv, scoring="neg_mean_absolute_error"
        ).mean()
        print(f"    {name:32} R2 {r2:+.3f}   erreur {mae:.1f} 1/s")
    print(
        "\n  La limite d'extinction depend du carburant ET de l'ozone ET de la"
        "\n  richesse, de facon non additive : un alcane lourd sans ozone et un"
        "\n  alcane leger avec ozone ne se deduisent pas d'une somme de"
        "\n  contributions. Une droite ne peut pas representer ca."
    )

    print("\n" + "-" * 74)
    print("2. CE QUE FAIT L'OZONE")
    print("-" * 74)
    source = load()
    pivot = (
        source.pivot_table(
            index="fuel", columns=["extinction_type", "ozone"], values=TARGET,
            aggfunc="mean",
        )
        .round(1)
    )
    print("\n  vitesse d'etirement moyenne a l'extinction, 1/s :\n")
    print(pivot.to_string())
    cool = source[source["extinction_type"] == "cool"]
    ranges = cool.groupby("ozone")["fuel_mass_fraction"].agg(["count", "min", "max"])
    print("\n  richesses explorees, par campagne :\n")
    print(ranges.round(3).to_string())
    print(
        "\n  LES DEUX CAMPAGNES NE SE RECOUVRENT PAS. Sans ozone la richesse va"
        "\n  de 0.044 a 0.220 ; avec ozone, de 0.234 a 0.555. La bande commune"
        "\n  est VIDE."
        "\n\n  Comparer les vitesses d'etirement moyennes des deux campagnes n'a"
        "\n  donc aucun sens : elles ont ete mesurees a des richesses"
        "\n  entierement differentes. Une moyenne plus basse avec ozone ne dit"
        "\n  rien de l'ozone, elle dit qu'on a mesure ailleurs."
        "\n\n  Ce que le plan d'experience montre, c'est que l'ozone DEPLACE le"
        "\n  domaine ou une flamme froide existe : Princeton a pu travailler a"
        "\n  des richesses deux a trois fois plus elevees avec ozone qu'il ne"
        "\n  pouvait sans. C'est cela, la sensibilisation, et elle se lit dans"
        "\n  les BORNES du tableau, pas dans ses moyennes."
        "\n\n  Consequence pour le modele : la variable `ozone` est en partie un"
        "\n  indicateur de la plage de richesse, et son importance ci-dessous"
        "\n  est a lire avec cette reserve."
    )

    print("\n" + "-" * 74)
    print("3. CE QUI COMPTE POUR LA FORET")
    print("-" * 74)
    fitted = model().fit(X, y)
    importances = pd.Series(fitted.feature_importances_, index=X.columns).sort_values()
    print("\n  importance relative des entrees :\n")
    for name, weight in importances.items():
        bar = "#" * int(weight * 60)
        print(f"    {name:34} {weight:.3f}  {bar}")
    print(
        "\n  Une importance n'est pas un coefficient : elle dit qu'une variable"
        "\n  sert aux decoupages, pas dans quel SENS elle agit. C'est ce qu'on"
        "\n  perd en quittant le lineaire, et c'est pourquoi le tableau du"
        "\n  point 2 reste necessaire."
    )

    print("\n" + "=" * 74)
    print(
        f"BILAN\n"
        f"\n  foret    erreur {typical_error():.1f} 1/s sur une plage de "
        f"{y.min():.0f} a {y.max():.0f}"
        f"\n  a 1 g, jamais a melanger avec les modules ISS sans le dire."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
