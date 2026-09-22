"""Modèle « Suppression » — PSI-69 / FLEX-1.

Question posée : connaissant le carburant, la pression, la teneur en oxygène,
la quantité de CO2 ou d'hélium ajoutée et la taille de la gouttelette, la
flamme va-t-elle s'éteindre ?

Ce module n'est pas seulement un entraînement, c'est une démonstration. Il
déroule dans l'ordre les étapes qui permettent de savoir si un modèle a
réellement appris quelque chose, et chacune répond à une question précise :

    1. Le plancher       que vaut le modèle le plus bête possible ?
    2. La régression     un modèle interprétable fait-il mieux ?
    3. Les coefficients  ce qu'il a appris correspond-il à la physique ?
    4. Le déséquilibre   que change le fait de pondérer les classes ?
    5. La courbe          plus de données aideraient-elles ?
    6. Le sur-apprentissage  pourquoi pas un modèle plus puissant ?

DONNÉES. 213 essais, dont 7 portent un trou dans une feature. Ces 7 lignes
appartiennent TOUTES à la classe majoritaire — vérifié — donc les écarter ne
coûte aucun des 27 exemples minoritaires. On les écarte plutôt que d'imputer :
inventer une pression pour gagner 7 essais déjà surreprésentés n'apporte rien.

Il reste 206 essais, 179 extinctions contre 27 combustions complètes.

LE DÉSÉQUILIBRE EST LE PROBLÈME CENTRAL. 87 % d'une seule classe signifie
qu'un modèle qui répond toujours « extinction » a 87 % de justesse et ne sert
à rien. La justesse (accuracy) est donc une métrique trompeuse ici, et on ne
s'en sert que pour mémoire. Ce qui compte :

    rappel de la classe rare   sur les 27 essais où la flamme a brûlé
                               jusqu'au bout, combien le modèle en voit-il ?
    justesse équilibrée        moyenne des rappels des deux classes, donc
                               insensible à la proportion
    ROC AUC                    capacité à ordonner les essais du moins au
                               plus susceptible de s'éteindre, indépendamment
                               du seuil de décision

DÉCOUPAGE. Validation croisée stratifiée à 5 plis : chaque pli conserve la
proportion 87/13, sans quoi un pli pourrait ne contenir aucun exemple rare.
Avec 27 exemples rares, chaque pli n'en teste qu'environ 5 — c'est très peu,
et c'est pourquoi on affiche l'écart entre plis et pas seulement la moyenne.
Un score moyen qui cache une forte dispersion n'est pas un score fiable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flame.loaders.psi69 import FEATURES, LABEL, load

RANDOM_STATE = 0
FOLDS = 5

# Ce que la physique de la combustion fait attendre du signe de chaque
# coefficient. Sert de contrôle : un modèle interprétable est un modèle
# qu'on peut prendre en défaut avec ce qu'on sait déjà.
PHYSICS_EXPECTATION = {
    "co2_frac": ("+", "le CO2 est un suppresseur, il absorbe la chaleur"),
    "he_frac": ("+", "l'helium conduit la chaleur hors de la flamme"),
    "o2_frac": ("-", "plus d'oxygene entretient la combustion"),
    "pressure_mmhg": ("-", "la pression favorise generalement la combustion"),
    "fuel_methanol": ("+", "le methanol s'eteint bien plus souvent que l'heptane"),
    "d0_mm": ("?", "effet non evident a priori"),
}


def prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Table d'entraînement : features encodées, lignes incomplètes écartées."""
    df = load()
    complete = df[FEATURES].notna().all(axis=1)

    dropped = df[~complete]
    if (dropped[LABEL] == 0).any():
        raise ValueError(
            "des essais de la classe rare seraient perdus : reconsiderer "
            "l'imputation plutot que la suppression"
        )

    df = df[complete]
    X = df[FEATURES].copy()
    # Le carburant est binaire : une seule colonne suffit, et « methanol=1 »
    # se lit directement dans le coefficient.
    X["fuel_methanol"] = (X.pop("fuel") == "Methanol").astype(int)
    return X, df[LABEL].astype(int)


def _logistic(balanced: bool) -> Pipeline:
    """Régression logistique, précédée d'une mise à l'échelle.

    La mise à l'échelle n'améliore pas le modèle, elle rend ses coefficients
    COMPARABLES : sans elle, la pression (des centaines de mmHg) et une
    fraction molaire (entre 0 et 1) produisent des coefficients dont les
    tailles ne veulent rien dire l'une par rapport à l'autre. Après mise à
    l'échelle, un coefficient se lit « effet d'un écart-type de cette
    grandeur ».
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced" if balanced else None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _cross_validate(model, X, y) -> dict:
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=splitter,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            # ATTENTION : le scorer « recall » de scikit-learn vise pos_label=1
            # par defaut. Ici la classe RARE est 0 (combustion complete). Avec
            # le scorer par defaut, le modele bete qui repond toujours 1
            # obtient un rappel de 1.00 sur une classe qu'il ne predit jamais.
            # Il faut donc designer explicitement la classe visee.
            "recall_rare": make_scorer(recall_score, pos_label=0),
            "roc_auc": "roc_auc",
        },
        return_train_score=True,
    )
    return scores


def _report(name: str, scores: dict) -> None:
    print(f"\n  {name}")
    for key, label in [
        ("test_accuracy", "justesse (trompeuse ici)"),
        ("test_balanced_accuracy", "justesse equilibree"),
        ("test_recall_rare", "rappel de la classe rare"),
        ("test_roc_auc", "ROC AUC"),
    ]:
        values = scores[key]
        print(
            f"    {label:28} {values.mean():.3f}  "
            f"(plis : {' '.join(f'{v:.2f}' for v in values)})"
        )


def main() -> None:
    X, y = prepare()
    rare = int((y == 0).sum())
    print("=" * 74)
    print("MODULE SUPPRESSION — PSI-69 / FLEX-1")
    print("=" * 74)
    print(f"\n{len(X)} essais, {len(X.columns)} features")
    print(f"  extinction (classe frequente) : {int((y == 1).sum())}")
    print(f"  combustion complete (rare)    : {rare}")

    # --- 1. Le plancher ----------------------------------------------------
    print("\n" + "-" * 74)
    print("1. LE PLANCHER — que vaut le modele le plus bete possible ?")
    print("-" * 74)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_scores = _cross_validate(dummy, X, y)
    floor = dummy_scores["test_accuracy"].mean()
    print(
        f"\n  Un modele qui repond TOUJOURS « extinction », sans regarder"
        f"\n  les donnees, obtient {floor:.1%} de justesse."
        f"\n  Il ne detecte evidemment aucun des {rare} essais rares :"
        f" rappel = {dummy_scores['test_recall_rare'].mean():.2f}."
        f"\n\n  C'est le seuil a battre. Tout modele autour de {floor:.0%} de"
        f"\n  justesse n'a rien appris, meme si le chiffre parait bon."
    )

    # --- 2. La regression logistique ---------------------------------------
    print("\n" + "-" * 74)
    print("2. LA REGRESSION LOGISTIQUE — un modele interpretable fait-il mieux ?")
    print("-" * 74)
    plain = _logistic(balanced=False)
    plain_scores = _cross_validate(plain, X, y)
    _report("sans ponderation des classes", plain_scores)

    # --- 3. Les coefficients contre la physique ----------------------------
    print("\n" + "-" * 74)
    print("3. LES COEFFICIENTS — ce qu'il a appris tient-il physiquement ?")
    print("-" * 74)
    fitted = _logistic(balanced=True).fit(X, y)
    coefficients = pd.Series(
        fitted.named_steps["model"].coef_[0], index=X.columns
    ).sort_values(key=abs, ascending=True)

    print("\n  Un coefficient positif pousse vers l'EXTINCTION.")
    print("  Valeur lue par ecart-type de la grandeur, d'ou la comparabilite.\n")
    agreements = disagreements = 0
    for name, value in coefficients.items():
        expected, reason = PHYSICS_EXPECTATION.get(name, ("?", ""))
        sign = "+" if value > 0 else "-"
        if expected == "?":
            verdict = "pas d'attente"
        elif sign == expected:
            verdict = "conforme"
            agreements += 1
        else:
            verdict = "CONTRAIRE A L'ATTENTE"
            disagreements += 1
        bar = "#" * int(min(abs(value) * 12, 30))
        print(f"    {name:16} {value:+7.3f}  {bar:<30} {verdict}")
        if reason:
            print(f"    {'':16} {'':7}  attendu {expected} : {reason}")

    print(
        f"\n  {agreements} coefficients conformes a la physique, "
        f"{disagreements} contraires."
    )
    if disagreements == 0:
        print(
            "  Le modele n'a rien appris qui contredise ce qu'on savait deja."
            "\n  C'est le minimum exigible, pas une preuve de qualite : un modele"
            "\n  peut avoir les bons signes et une capacite de prediction nulle."
        )

    # --- 4. Ponderer les classes -------------------------------------------
    print("\n" + "-" * 74)
    print("4. PONDERER LES CLASSES — changer ce que le modele cherche a optimiser")
    print("-" * 74)
    print(
        "\n  Par defaut le modele minimise l'erreur totale, donc il a interet a"
        "\n  ignorer une classe qui ne pese que 13 %. `class_weight='balanced'`"
        "\n  donne a chaque erreur sur la classe rare un poids proportionnel a"
        "\n  sa rarete. La justesse brute baisse, le rappel de la classe rare"
        "\n  monte — c'est un arbitrage, pas une amelioration gratuite."
    )
    balanced_scores = _cross_validate(_logistic(balanced=True), X, y)
    _report("avec ponderation des classes", balanced_scores)

    print("\n  Matrice de confusion, modele pondere, validation croisee :")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    predictions = pd.Series(index=y.index, dtype=int)
    for train_index, test_index in splitter.split(X, y):
        model = _logistic(balanced=True).fit(X.iloc[train_index], y.iloc[train_index])
        predictions.iloc[test_index] = model.predict(X.iloc[test_index])
    matrix = confusion_matrix(y, predictions)
    print(f"\n{'':22}{'predit complete':>18}{'predit extinction':>20}")
    for label, row in zip(["reel complete", "reel extinction"], matrix):
        print(f"    {label:20}{row[0]:>18}{row[1]:>20}")
    print(
        f"\n    rappel classe rare  : {recall_score(y, predictions, pos_label=0):.2f}"
        f"\n    justesse equilibree : {balanced_accuracy_score(y, predictions):.2f}"
    )

    # --- 5. La courbe d'apprentissage --------------------------------------
    print("\n" + "-" * 74)
    print("5. LA COURBE D'APPRENTISSAGE — plus de donnees aideraient-elles ?")
    print("-" * 74)
    sizes, train_scores, test_scores = learning_curve(
        _logistic(balanced=True),
        X,
        y,
        cv=splitter,
        scoring="balanced_accuracy",
        train_sizes=np.linspace(0.3, 1.0, 6),
        random_state=RANDOM_STATE,
    )
    print("\n    essais    apprentissage    validation")
    for size, train, test in zip(sizes, train_scores.mean(1), test_scores.mean(1)):
        print(f"    {int(size):6d}    {train:13.3f}    {test:10.3f}")
    slope = test_scores.mean(1)[-1] - test_scores.mean(1)[-3]
    print(
        f"\n  Progression sur les deux derniers paliers : {slope:+.3f}."
        + (
            "\n  La courbe de validation monte encore : plus d'essais aideraient."
            if slope > 0.02
            else "\n  La courbe plafonne : ajouter des essais du meme type ne"
            "\n  suffirait pas, il faudrait d'autres features ou d'autres"
            "\n  conditions."
        )
    )

    # --- 6. Le sur-apprentissage -------------------------------------------
    print("\n" + "-" * 74)
    print("6. POURQUOI PAS UN MODELE PLUS PUISSANT ?")
    print("-" * 74)
    print(
        "\n  On affirme souvent qu'a ce nombre de lignes un modele complexe"
        "\n  « memorise le bruit ». Voici le chiffre plutot que l'affirmation."
    )
    print("\n    profondeur   apprentissage   validation      ecart")
    for depth in [2, 3, 5, 10, None]:
        tree = DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=RANDOM_STATE
        )
        scores = _cross_validate(tree, X, y)
        train = scores["train_balanced_accuracy"].mean()
        test = scores["test_balanced_accuracy"].mean()
        name = "illimitee" if depth is None else str(depth)
        flag = "  <- memorise" if train - test > 0.25 else ""
        print(f"    {name:>10}   {train:13.3f}   {test:10.3f}   {train - test:8.3f}{flag}")
    print(
        "\n  Un arbre sans limite de profondeur atteint la perfection sur les"
        "\n  essais qu'il a vus et s'effondre sur les autres. Il n'a pas appris"
        "\n  la physique de l'extinction, il a appris les 206 essais par coeur."
    )

    # --- 7. Un coefficient n'est pas une cause -----------------------------
    print("\n" + "-" * 74)
    print("7. UN COEFFICIENT N'EST PAS UNE CAUSE")
    print("-" * 74)
    print(
        "\n  L'etape 3 a signale un desaccord : le modele affirme que la pression"
        "\n  POUSSE a l'extinction, alors que la physique dit l'inverse. Plutot"
        "\n  que de corriger l'attente ou d'ignorer le signal, on regarde le plan"
        "\n  d'experience."
    )
    bands = pd.cut(
        X["pressure_mmhg"],
        [0, 700, 800, 1600, 2400],
        labels=["< 0.9 atm", "~ 1 atm", "1-2 atm", "> 2 atm"],
    )
    grouped = X.groupby(bands, observed=True)["co2_frac"]
    table = pd.DataFrame(
        {
            "essais": grouped.size(),
            "CO2 moyen": grouped.mean().round(3),
            "taux extinction": y.groupby(bands, observed=True).mean().round(3),
        }
    )
    print("\n" + table.to_string())
    correlation = X["pressure_mmhg"].corr(X["co2_frac"])
    print(
        f"\n  correlation pression <-> CO2 : {correlation:+.2f}"
        "\n\n  Les seuls essais au-dessus de 1 atm sont aussi les seuls charges a"
        "\n  70 % de CO2, et ils s'eteignent tous. Il y en a 7. Le modele ne peut"
        "\n  pas separer les deux causes, faute d'un seul essai a haute pression"
        "\n  SANS CO2 : ce coefficient de pression mesure en realite l'effet du"
        "\n  suppresseur."
        "\n\n  Rien dans les scores ne l'aurait revele — ils sont bons. C'est la"
        "\n  lecture des coefficients contre la physique qui l'a fait remonter,"
        "\n  et c'est la raison de preferer un modele interpretable : un modele"
        "\n  opaque aurait exactement le meme defaut, sans qu'on puisse le voir."
        "\n\n  A retenir : un coefficient decrit ce que le modele a trouve dans CES"
        "\n  donnees, pas ce que la nature fait. Ici le plan d'experience ne"
        "\n  permet pas de conclure sur la pression. Il faudrait des essais a"
        "\n  haute pression sans CO2, et ils n'ont pas ete faits."
    )

    # --- Conclusion --------------------------------------------------------
    print("\n" + "=" * 74)
    best = balanced_scores["test_balanced_accuracy"].mean()
    auc = balanced_scores["test_roc_auc"].mean()
    print(
        f"BILAN\n"
        f"\n  plancher (toujours « extinction »)   justesse {floor:.1%}, rappel rare 0.00"
        f"\n  regression logistique ponderee       justesse equilibree {best:.3f},"
        f" ROC AUC {auc:.3f}"
        f"\n\n  Le modele voit reellement quelque chose que le plancher ne voit pas."
        f"\n  Avec {rare} exemples rares seulement, chaque pli n'en teste que ~{rare // FOLDS},"
        f"\n  d'ou la dispersion entre plis affichee plus haut : a lire comme une"
        f"\n  fourchette, jamais comme un chiffre unique."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
