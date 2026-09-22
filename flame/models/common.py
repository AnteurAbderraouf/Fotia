"""Machinerie d'évaluation partagée par les modules de modélisation.

Les questions qu'on pose à un modèle sont les mêmes d'un module à l'autre —
bat-il le plancher, ce qu'il a appris tient-il physiquement, plus de données
aideraient-elles, un modèle plus puissant ferait-il mieux. Seules les réponses
changent. Ce fichier tient les outils ; chaque module apporte ses données et
ses attentes physiques.

UN PIÈGE INSCRIT ICI UNE FOIS POUR TOUTES. Le scorer « recall » de
scikit-learn vise `pos_label=1` par défaut. Quand la classe rare porte
l'étiquette 0, le modèle le plus bête — qui répond toujours 1 — obtient un
rappel de 1.00 sur une classe qu'il ne prédit jamais. C'est silencieux et
c'est faux. `scorers()` demande donc explicitement la classe visée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 0
FOLDS = 5


def splitter(folds: int = FOLDS) -> StratifiedKFold:
    """Découpage stratifié : chaque pli garde la proportion des classes.

    Sans stratification, un pli pourrait ne contenir aucun exemple de la
    classe rare, et son score n'aurait plus de sens.
    """
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def scorers(rare_label: int) -> dict:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "recall_rare": make_scorer(recall_score, pos_label=rare_label),
        "roc_auc": "roc_auc",
    }


def logistic(balanced: bool = True) -> Pipeline:
    """Régression logistique précédée d'une mise à l'échelle.

    La mise à l'échelle ne change pas les prédictions, elle rend les
    coefficients COMPARABLES : sans elle, une pression en bars et une fraction
    molaire produisent des coefficients dont les tailles ne se comparent pas.
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


def evaluate(model, X, y, rare_label: int) -> dict:
    return cross_validate(
        model,
        X,
        y,
        cv=splitter(),
        scoring=scorers(rare_label),
        return_train_score=True,
    )


def report(title: str, scores: dict) -> None:
    print(f"\n  {title}")
    for key, label in [
        ("test_accuracy", "justesse (trompeuse si deseq.)"),
        ("test_balanced_accuracy", "justesse equilibree"),
        ("test_recall_rare", "rappel de la classe rare"),
        ("test_roc_auc", "ROC AUC"),
    ]:
        values = scores[key]
        print(
            f"    {label:32} {values.mean():.3f}  "
            f"(plis : {' '.join(f'{v:.2f}' for v in values)})"
        )


def physics_check(coefficients: pd.Series, expectations: dict) -> tuple[int, int]:
    """Confronte chaque coefficient à ce que la physique fait attendre.

    Un modèle interprétable est un modèle qu'on peut prendre en défaut avec ce
    qu'on sait déjà. C'est ce contrôle, et non les scores, qui a révélé le
    confondement pression/CO2 dans le module Suppression.
    """
    agree = disagree = 0
    for name, value in coefficients.sort_values(key=abs).items():
        expected, reason = expectations.get(name, ("?", ""))
        sign = "+" if value > 0 else "-"
        if expected == "?":
            verdict = "pas d'attente"
        elif sign == expected:
            verdict = "conforme"
            agree += 1
        else:
            verdict = "CONTRAIRE A L'ATTENTE"
            disagree += 1
        bar = "#" * int(min(abs(value) * 12, 26))
        print(f"    {name:24} {value:+7.3f}  {bar:<26} {verdict}")
        if reason:
            print(f"    {'':24} {'':7}  attendu {expected} : {reason}")
    return agree, disagree


def learning_curve_table(model, X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return learning_curve(
        model,
        X,
        y,
        cv=splitter(),
        scoring="balanced_accuracy",
        train_sizes=np.linspace(0.3, 1.0, 6),
        random_state=RANDOM_STATE,
    )


def overfit_table(X, y, rare_label: int, depths=(2, 3, 5, 10, None)) -> None:
    """Le sur-apprentissage en chiffres plutôt qu'en affirmation."""
    print("\n    profondeur   apprentissage   validation      ecart")
    for depth in depths:
        tree = DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=RANDOM_STATE
        )
        scores = evaluate(tree, X, y, rare_label)
        train = scores["train_balanced_accuracy"].mean()
        test = scores["test_balanced_accuracy"].mean()
        name = "illimitee" if depth is None else str(depth)
        flag = "  <- memorise" if train - test > 0.25 else ""
        print(
            f"    {name:>10}   {train:13.3f}   {test:10.3f}   "
            f"{train - test:8.3f}{flag}"
        )
