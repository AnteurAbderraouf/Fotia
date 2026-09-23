"""Module « Flammes froides » — PSI-39 / CFI.

Après l'extinction de la flamme visible, une combustion invisible
persiste-t-elle ? L'enjeu de sécurité est direct : un feu déclaré éteint qui
continue à consommer du carburant et à produire des imbrûlés, sans être vu ni
détecté.

L'ÉTIQUETTE A DÛ ÊTRE REFAITE, ET C'EST LE POINT PRINCIPAL DU MODULE.

Le plan initial annonçait « 120 des 227 essais ont un diamètre d'extinction de
flamme froide, 107 non, étiquette à peu près équilibrée ». Deux erreurs s'y
cachaient.

    81 essais sur 227 ne sont pas depouilles. « Completed Analysis? = No »,
    et seuls 3 d'entre eux portent la moindre mesure. Les compter comme
    « pas de flamme froide » aurait entraine le modele a predire si un essai
    a ete ANALYSE, pas si une flamme froide a eu lieu.

    Parmi les 146 depouilles, le diametre d'extinction seul en manque 7 :
    ces essais ont un taux de combustion froide (kcool) sans diametre releve.
    Une flamme froide qui a un taux de combustion a bel et bien existe.

Étiquette honnête : kcool OU Dext,cool présent, sur les essais dépouillés
seulement. Soit 123 contre 21, c'est-à-dire 85/15. Pas « équilibrée ».

LE FARNESANE EST INTÉGRALEMENT PERDU. Ses 44 essais sont tous non dépouillés.
Ce biocarburant, un kérosène renouvelable, ne contribue donc aucune ligne. Le
module porte sur trois carburants, pas quatre.

CE QUE LE MODÈLE DONNE. Une AUC de 0.913, la meilleure du catalogue : le
modèle ordonne très bien les essais du moins au plus susceptible de porter une
flamme froide. La justesse équilibrée, elle, reste à 0.771 — trier est plus
facile que trancher quand une classe ne pèse que 15 %.

UNE SÉPARATION PARFAITE À SURVEILLER. À 5 atm, 15 essais sur 15 présentent une
flamme froide. Un sous-ensemble sans contre-exemple fait diverger une
régression logistique non régularisée ; ici la régularisation par défaut de
scikit-learn suffit, mais la propriété reste vraie du jeu de données.
"""

from __future__ import annotations

import pandas as pd

from flame.loaders.psi39 import FEATURES, LABEL, POST_BURN, load
from flame.models.common import (
    evaluate,
    learning_curve_table,
    logistic,
    overfit_table,
    physics_check,
    report,
)

RARE_LABEL = 0  # aucune flamme froide observee

PHYSICS_EXPECTATION = {
    "pressure_atm": ("+", "la pression favorise la chimie de basse temperature"),
    "o2_frac": ("+", "plus d'oxygene soutient aussi la flamme froide"),
    "he_frac": ("?", "l'helium refroidit, effet sur la flamme froide non evident"),
    "d0_mm": ("+", "une grosse goutte laisse plus de temps a la chimie froide"),
    "ignition_power_w": ("?", ""),
    "ignition_time_ms": ("?", ""),
    "fiber_Yes": ("?", "support de la goutte, pas d'attente physique"),
    "fuel_dodecane75-isododecane25": ("?", ""),
    "fuel_n-dodecane": ("?", ""),
}


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    usable = df[df[FEATURES].notna().all(axis=1) & df[LABEL].notna()]
    X = pd.get_dummies(usable[FEATURES], drop_first=True).astype(float)
    return X, usable[LABEL].astype(int)


def main() -> None:
    X, y = prepare()
    rare = int((y == RARE_LABEL).sum())

    print("=" * 74)
    print("MODULE FLAMMES FROIDES — PSI-39 / CFI")
    print("=" * 74)
    print(f"\n{len(X)} essais complets, {X.shape[1]} colonnes apres encodage")
    print(f"  flamme froide observee : {int((y == 1).sum())}")
    print(f"  aucune (classe rare)   : {rare}")

    print("\n" + "-" * 74)
    print("1. CE QUE L'ETIQUETTE A COUTE")
    print("-" * 74)
    full = load(analysed_only=False)
    print(
        f"\n  campagne complete            {len(full)} essais"
        f"\n  non depouilles, ecartes      {int((~full['analysis_completed']).sum())}"
        f"\n  depouilles, etiquetables     {int(full['analysis_completed'].sum())}"
        f"\n  complets pour ce modele      {len(X)}"
        "\n\n  Les compter comme « pas de flamme froide » aurait entraine le"
        "\n  modele a predire si un essai a ete ANALYSE. Le score aurait ete"
        "\n  bon et le modele sans objet."
    )
    fuels = full.groupby("fuel")["analysis_completed"].agg(["size", "sum"])
    fuels.columns = ["essais", "depouilles"]
    print("\n  par carburant :\n")
    print(fuels.to_string())
    print(
        "\n  Le farnesane n'a AUCUN essai depouille. Ce biocarburant ne"
        "\n  contribue donc rien : le module porte sur trois carburants."
    )

    print("\n" + "-" * 74)
    print("2. LE MODELE")
    print("-" * 74)
    floor = float(max(y.mean(), 1 - y.mean()))
    print(f"\n  plancher : repondre toujours « flamme froide » -> {floor:.0%}")
    scores = evaluate(logistic(balanced=True), X, y, RARE_LABEL)
    report("regression logistique ponderee", scores)
    print(
        "\n  L'AUC de {:.3f} est la meilleure du catalogue : le modele ORDONNE"
        "\n  tres bien les essais. La justesse equilibree reste plus basse —"
        "\n  trier est plus facile que trancher quand une classe pese 15 %."
        .format(scores["test_roc_auc"].mean())
    )

    print("\n" + "-" * 74)
    print("3. LES COEFFICIENTS CONTRE LA PHYSIQUE")
    print("-" * 74)
    fitted = logistic(balanced=True).fit(X, y)
    coefficients = pd.Series(fitted.named_steps["model"].coef_[0], index=X.columns)
    print("\n  Positif = flamme froide PLUS probable.\n")
    agree, disagree = physics_check(coefficients, PHYSICS_EXPECTATION)
    print(f"\n  {agree} conformes, {disagree} contraires.")

    print("\n" + "-" * 74)
    print("4. LA SEPARATION PARFAITE A 5 ATM")
    print("-" * 74)
    source = load()
    table = pd.crosstab(source["pressure_atm"], source[LABEL])
    print("\n" + table.to_string())
    print(
        "\n  A 5 atm, aucun contre-exemple. Un sous-ensemble parfaitement separe"
        "\n  fait diverger une regression logistique non regularisee ; la"
        "\n  regularisation par defaut de scikit-learn suffit ici, mais la"
        "\n  propriete reste vraie du jeu de donnees et limite ce qu'on peut"
        "\n  conclure a haute pression."
    )

    print("\n" + "-" * 74)
    print("5. COURBE D'APPRENTISSAGE ET SUR-APPRENTISSAGE")
    print("-" * 74)
    sizes, train_scores, test_scores = learning_curve_table(logistic(True), X, y)
    print("\n    essais    apprentissage    validation")
    for size, train, test in zip(sizes, train_scores.mean(1), test_scores.mean(1)):
        print(f"    {int(size):6d}    {train:13.3f}    {test:10.3f}")
    overfit_table(X, y, RARE_LABEL)

    print("\n" + "=" * 74)
    print(
        f"BILAN\n"
        f"\n  plancher                  {floor:.0%}"
        f"\n  regression ponderee       justesse equilibree "
        f"{scores['test_balanced_accuracy'].mean():.3f}, "
        f"AUC {scores['test_roc_auc'].mean():.3f}"
        f"\n\n  Avec {rare} exemples de la classe rare, chaque pli n'en teste que"
        f"\n  ~{rare // 5}. A lire comme une fourchette, jamais comme un chiffre."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
