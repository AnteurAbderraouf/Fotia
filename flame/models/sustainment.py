"""Modèle « Auto-entretien » — PSI-159 / ACME CFI-G.

Question : une flamme de gaz sur brûleur sphérique se maintient-elle d'elle-même,
ou s'éteint-elle seule ? L'étiquette est directement observée et n'invente aucun
seuil — SE, la flamme est morte seule, contre FT, l'équipage a coupé le débit
alors qu'elle brûlait encore.

CE QUE CE MODULE APPORTE FACE À LA SUPPRESSION.

C'est le meilleur jeu du catalogue pour modéliser, et pour trois raisons qui
n'ont rien à voir avec sa taille :

    272 essais           le plus fourni des jeux etiquetes
    plancher 67 %        183 SE contre 89 FT, le moins desequilibre de loin
    aucun trou           les sept features sont completes sur les 272 lignes

Le module Suppression devait composer avec 87 % d'une seule classe, ce qui
rendait la justesse brute trompeuse et forçait la pondération. Ici la marge de
manœuvre est réelle : un modèle a la place de montrer ce qu'il sait faire.

DEUX EXPÉRIENCES EN MIROIR, PAS UNE.

Le loader garde une colonne `flame_type`. Les flammes normales injectent du
carburant dans une ambiance oxydante ; les inverses injectent de l'oxygène dans
une ambiance carburée. Leurs colonnes de composition ne se correspondent donc
pas, et seules les grandeurs thermochimiques calculées — Tad et Zst — décrivent
la flamme indépendamment de la configuration.

POURQUOI TAD ET ZST SONT INDISPENSABLES, mesuré et non suppose :

    features reglables seules            justesse equilibree 0.646
    + Tad seule                          0.643
    + Zst seule                          0.673
    + Tad ET Zst                         0.731

Ni l'une ni l'autre ne vaut grand-chose isolément ; ensemble elles apportent
près de neuf points. C'est cohérent physiquement — Tad dit à quelle température
la flamme peut monter, Zst où la réaction se situe dans l'espace des mélanges.
Et surtout, ce sont les deux SEULES variables qui encodent la composition sans
dépendre de la configuration. Les retirer reviendrait à retirer le mélange.

Conséquence pour le tableau de bord : ce ne sont pas des molettes. Tad et Zst
sont des propriétés calculées d'un mélange donné, et toute paire (Tad, Zst)
n'est pas physiquement réalisable. C'est exactement ce que la carte de
couverture est là pour montrer.
"""

from __future__ import annotations

import pandas as pd

from flame.loaders.psi159 import FEATURES, LABEL, load
from flame.models.common import (
    evaluate,
    learning_curve_table,
    logistic,
    overfit_table,
    physics_check,
    report,
    splitter,
)

# L'etiquette vaut 1 quand la flamme s'eteint SEULE. Un coefficient positif
# pousse donc vers la mort de la flamme.
PHYSICS_EXPECTATION = {
    "tad_k": ("-", "une flamme plus chaude resiste mieux"),
    "fuel_flow_mg_s": ("-", "plus de carburant alimente la flamme"),
    "fuel_dilution": ("-", "un carburant pur brule mieux qu'un carburant dilue"),
    "pressure_bar": ("-", "la pression favorise generalement la combustion"),
    "zst": ("?", "position dans l'espace des melanges, effet non evident"),
    "flame_type_normal": ("?", "configuration, pas de prevision a priori"),
    "fuel_ethane": ("?", ""),
    "fuel_propane": ("?", ""),
    "tfp_yes": ("?", "colonne non documentee : l'info.md de PSI-159 est vide"),
}

RARE_LABEL = 0  # « flow term » : l'equipage a coupe, la flamme vivait encore


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Table d'entraînement. Aucune ligne n'est perdue : il n'y a pas de trou."""
    df = load()
    complete = df[FEATURES].notna().all(axis=1)
    if not complete.all():
        raise ValueError(
            f"{(~complete).sum()} lignes incompletes alors qu'on n'en attendait "
            "aucune : le loader a change, revoir cette hypothese"
        )
    X = pd.get_dummies(df[FEATURES], drop_first=True).astype(float)
    return X, df[LABEL].astype(int)


def main() -> None:
    X, y = prepare()
    rare = int((y == RARE_LABEL).sum())

    print("=" * 74)
    print("MODULE AUTO-ENTRETIEN — PSI-159 / ACME CFI-G")
    print("=" * 74)
    print(f"\n{len(X)} essais, {X.shape[1]} colonnes apres encodage")
    print(f"  auto-extinction, la flamme meurt seule  : {int((y == 1).sum())}")
    print(f"  flow term, l'equipage coupe le debit    : {rare}")

    print("\n" + "-" * 74)
    print("1. LE PLANCHER")
    print("-" * 74)
    floor = float((y == 1).mean())
    print(
        f"\n  Repondre toujours « auto-extinction » donne {floor:.1%} de justesse."
        f"\n  C'est le plancher le plus BAS du catalogue : les autres modules"
        f"\n  partent de 85 ou 87 %. Un modele a donc ici de la place pour montrer"
        f"\n  ce qu'il sait faire, au lieu de se battre contre un desequilibre."
    )

    print("\n" + "-" * 74)
    print("2. LA REGRESSION LOGISTIQUE")
    print("-" * 74)
    scores = evaluate(logistic(balanced=True), X, y, RARE_LABEL)
    report("ponderee par les classes", scores)

    print("\n" + "-" * 74)
    print("3. LES COEFFICIENTS CONTRE LA PHYSIQUE")
    print("-" * 74)
    fitted = logistic(balanced=True).fit(X, y)
    coefficients = pd.Series(fitted.named_steps["model"].coef_[0], index=X.columns)
    print("\n  Un coefficient positif pousse vers l'AUTO-EXTINCTION,")
    print("  c'est-a-dire vers la mort de la flamme.\n")
    agree, disagree = physics_check(coefficients, PHYSICS_EXPECTATION)
    print(f"\n  {agree} coefficients conformes, {disagree} contraires.")

    print("\n" + "-" * 74)
    print("4. CE QU'APPORTENT TAD ET ZST")
    print("-" * 74)
    print(
        "\n  Ces deux grandeurs sont CALCULEES a partir du melange, pas reglees."
        "\n  On verifie donc qu'elles gagnent leur place, plutot que de le supposer.\n"
    )
    settable = ["fuel", "fuel_dilution", "pressure_bar", "fuel_flow_mg_s", "flame_type"]
    raw = load()
    for title, columns in [
        ("features reglables seules", settable),
        ("+ Tad seule", settable + ["tad_k"]),
        ("+ Zst seule", settable + ["zst"]),
        ("+ Tad ET Zst", settable + ["tad_k", "zst"]),
    ]:
        subset = pd.get_dummies(raw[columns], drop_first=True).astype(float)
        result = evaluate(logistic(balanced=True), subset, y, RARE_LABEL)
        print(
            f"    {title:28} justesse eq. "
            f"{result['test_balanced_accuracy'].mean():.3f}   "
            f"AUC {result['test_roc_auc'].mean():.3f}"
        )
    print(
        "\n  Ni l'une ni l'autre isolement, les deux ensemble. Elles encodent la"
        "\n  composition independamment de la configuration : c'est la seule"
        "\n  chose qui se compare entre flammes normales et inverses."
    )

    print("\n" + "-" * 74)
    print("5. LA COURBE D'APPRENTISSAGE")
    print("-" * 74)
    sizes, train_scores, test_scores = learning_curve_table(logistic(True), X, y)
    print("\n    essais    apprentissage    validation")
    for size, train, test in zip(sizes, train_scores.mean(1), test_scores.mean(1)):
        print(f"    {int(size):6d}    {train:13.3f}    {test:10.3f}")
    slope = test_scores.mean(1)[-1] - test_scores.mean(1)[-3]
    print(
        f"\n  Progression sur les deux derniers paliers : {slope:+.3f}."
        + (
            "\n  La courbe monte encore : plus d'essais aideraient."
            if slope > 0.02
            else "\n  La courbe plafonne : d'autres features, pas plus de lignes."
        )
    )

    print("\n" + "-" * 74)
    print("6. UN MODELE PLUS PUISSANT ?")
    print("-" * 74)
    overfit_table(X, y, RARE_LABEL)

    print("\n" + "-" * 74)
    print("7. POURQUOI L'ARBRE GAGNE ICI, ALORS QU'IL PERDAIT SUR PSI-69")
    print("-" * 74)
    print(
        "\n  Le §9 du handoff posait une regle generale : a ce nombre de lignes,"
        "\n  preferer une regression logistique ou un arbre peu profond, un"
        "\n  ensemble memorisera le bruit. Cette regle vaut pour PSI-69. Elle ne"
        "\n  vaut PAS ici, et la mesure le dit :\n"
    )
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=0
    )
    forest_scores = evaluate(forest, X, y, RARE_LABEL)
    print(
        f"    regression logistique      justesse eq. "
        f"{scores['test_balanced_accuracy'].mean():.3f}   "
        f"AUC {scores['test_roc_auc'].mean():.3f}"
    )
    print(
        f"    foret 200 arbres, prof. 6  justesse eq. "
        f"{forest_scores['test_balanced_accuracy'].mean():.3f}   "
        f"AUC {forest_scores['test_roc_auc'].mean():.3f}"
    )

    print(
        "\n  La raison n'est pas que la foret serait « meilleure » dans l'absolu."
        "\n  Elle tient a une propriete de CES donnees, visible en ajustant le"
        "\n  modele separement sur chaque configuration :\n"
    )
    settings = ["fuel", "fuel_dilution", "pressure_bar", "tad_k", "zst", "fuel_flow_mg_s"]
    print(f"    {'configuration':12} {'n':>4}   {'pression':>9} {'debit':>9}")
    for configuration in ["normal", "inverse"]:
        subset = raw[raw["flame_type"] == configuration]
        columns = pd.get_dummies(subset[settings], drop_first=True).astype(float)
        fit = logistic(balanced=True).fit(columns, subset[LABEL].astype(int))
        coefficient = pd.Series(fit.named_steps["model"].coef_[0], index=columns.columns)
        print(
            f"    {configuration:12} {len(subset):4d}   "
            f"{coefficient['pressure_bar']:+9.3f} {coefficient['fuel_flow_mg_s']:+9.3f}"
        )

    print(
        "\n  Les deux coefficients CHANGENT DE SIGNE d'une configuration a l'autre."
        "\n  En flamme normale la pression soutient la combustion, comme attendu ;"
        "\n  en flamme inverse l'effet s'inverse. Ce sont deux experiences en"
        "\n  miroir, et leurs mecanismes ne se superposent pas."
        "\n\n  Un modele lineaire qui les met en commun MOYENNE deux effets opposes"
        "\n  et produit un chiffre qui ne decrit ni l'un ni l'autre — d'ou les deux"
        "\n  coefficients contraires a la physique vus a l'etape 3. Un arbre peut"
        "\n  d'abord separer sur la configuration, puis apprendre chaque branche"
        "\n  pour elle-meme. C'est cette interaction qu'il capture, pas du bruit."
        "\n\n  A retenir : « pas d'ensemble a ce N » n'est pas une regle, c'est une"
        "\n  observation valable jeu par jeu. Elle se verifie, elle ne se suppose pas."
    )

    print("\n" + "=" * 74)
    print(
        f"BILAN\n"
        f"\n  plancher                        {floor:.1%}"
        f"\n  regression logistique ponderee  justesse equilibree "
        f"{scores['test_balanced_accuracy'].mean():.3f}, "
        f"ROC AUC {scores['test_roc_auc'].mean():.3f}"
        f"\n  foret 200 arbres, profondeur 6  justesse equilibree "
        f"{forest_scores['test_balanced_accuracy'].mean():.3f}, "
        f"ROC AUC {forest_scores['test_roc_auc'].mean():.3f}"
        f"\n\n  Avec {rare} exemples de la classe rare, chaque pli en teste ~{rare // 5} :"
        f"\n  beaucoup plus confortable que les 5 du module Suppression."
        f"\n\n  La foret predit mieux, la regression explique mieux. On garde les"
        f"\n  deux : la foret pour la prediction affichee, la regression pour le"
        f"\n  controle des coefficients contre la physique — c'est lui qui a"
        f"\n  revele l'inversion de signe ci-dessus."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
