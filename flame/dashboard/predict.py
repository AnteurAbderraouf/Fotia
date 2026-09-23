"""Construction des modèles servis par le tableau de bord.

UN MODÈLE PAR MODULE, CHOISI PAR LA MESURE ET NON PAR PRINCIPE.

Le §9 du handoff posait une règle générale — à ce nombre de lignes, préférer
une régression logistique, un ensemble mémoriserait le bruit. Mesurée jeu par
jeu, cette règle tient pour l'un et pas pour l'autre :

    PSI-69   206 essais, 27 rares   regression 0.753   arbre illimite 0.678
    PSI-159  272 essais, 89 rares   regression 0.722   foret prof. 6  0.840

Sur PSI-159 la forêt gagne de douze points, et la raison est physique : les
flammes normales et inverses ont des coefficients de signe OPPOSÉ sur la
pression et le débit. Un modèle linéaire qui met les deux configurations en
commun moyenne deux effets contraires ; un arbre sépare d'abord sur la
configuration puis apprend chaque branche. Ce n'est pas du bruit qu'il
capture, c'est une interaction réelle.

D'où le choix ici : chaque module déclare le modèle que la mesure a retenu.

DEUX MODÈLES POUR PSI-159, PAS UN. La forêt prédit mieux, la régression
explique mieux. On garde les deux — la forêt sert la prédiction affichée, la
régression sert le contrôle des coefficients contre la physique, et c'est ce
contrôle qui a révélé l'inversion de signe.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flame.dashboard.proximity import ProximityIndex
from flame.dashboard.registry import Module
from flame.models.common import logistic


@dataclass
class Bundle:
    """Tout ce dont la page a besoin pour un module."""

    module: Module
    model: object
    columns: list[str]
    frame: pd.DataFrame
    index: ProximityIndex
    interpretable: object | None = None
    # Regresseur secondaire : il repond « a quelle taille », quand le modele
    # principal repond « est-ce que ». Les deux s'affichent ensemble, car un
    # diametre d'extinction n'a de sens que si la flamme s'eteint.
    regressor: object | None = None
    regressor_columns: list[str] | None = None
    regressor_error: float = 0.0
    # Constante de la loi en d2. Elle donne le RYTHME du retrecissement, la ou
    # le regresseur precedent donne son point d'arrivee. Les deux ensemble
    # decrivent toute l'histoire de la combustion.
    rate_model: object | None = None
    rate_columns: list[str] | None = None
    rate_error: float = 0.0


def _estimator(module: Module):
    """Le modèle retenu pour ce module, tel que la mesure l'a désigné.

    Quatre jeux de donnees, quatre reponses : la foret gagne douze points sur
    PSI-159, n'apporte rien sur PSI-101 ni sur PSI-107, et perd sur PSI-69.
    Le choix se mesure jeu par jeu, il ne se decrete pas.
    """
    if module.outcome_kind == "regression":
        # PSI-142 est le seul jeu ou le lineaire fait PIRE que de repondre la
        # moyenne (R2 -0.102 contre -0.089) : la limite d'extinction y depend
        # du carburant, de l'ozone et de la richesse de facon non additive.
        if module.key == "ground":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=300, max_depth=8, random_state=0
            )
        from sklearn.linear_model import Ridge

        return Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    if module.key == "sustainment":
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=0
        )
    return logistic(balanced=True)


def build(module: Module) -> Bundle:
    """Entraîne le module et prépare son index de proximité."""
    frame = module.loader()
    outcome = module.label or module.target
    frame = frame[
        frame[module.features].notna().all(axis=1) & frame[outcome].notna()
    ].reset_index(drop=True)

    X = pd.get_dummies(frame[module.features], drop_first=True).astype(float)
    y = frame[outcome] if module.target else frame[outcome].astype(int)

    model = _estimator(module).fit(X, y)
    interpretable = None
    if module.key == "sustainment":
        # Gardee a cote de la foret : c'est elle qui rend les coefficients
        # lisibles, donc verifiables contre la physique.
        interpretable = logistic(balanced=True).fit(X, y)

    regressor = regressor_columns = None
    rate_model = rate_columns = None
    regressor_error = rate_error = 0.0
    if module.key == "suppression":
        from flame.models.extinction_diameter import model as diameter_model
        from flame.models.extinction_diameter import prepare, typical_error

        diameter_X, diameter_y = prepare()
        regressor = diameter_model().fit(diameter_X, diameter_y)
        regressor_columns = list(diameter_X.columns)
        regressor_error = typical_error()

        from flame.models.droplet_burn import model as rate_estimator
        from flame.models.droplet_burn import prepare as rate_prepare
        from flame.models.droplet_burn import typical_error as rate_typical_error

        rate_X, rate_y = rate_prepare()
        rate_model = rate_estimator().fit(rate_X, rate_y)
        rate_columns = list(rate_X.columns)
        rate_error = rate_typical_error()

    return Bundle(
        module=module,
        model=model,
        columns=list(X.columns),
        frame=frame,
        index=ProximityIndex(frame, module.features),
        interpretable=interpretable,
        regressor=regressor,
        regressor_columns=regressor_columns,
        regressor_error=regressor_error,
        rate_model=rate_model,
        rate_columns=rate_columns,
        rate_error=rate_error,
    )


def to_features(module: Module, query: dict) -> dict:
    """Traduit l'état des commandes en features du modèle.

    Les commandes parlent l'unite de l'utilisateur — la pression en atm —
    quand le jeu de donnees peut porter autre chose, ici des mmHg. La
    conversion est declaree dans le registre plutot qu'enfouie dans la page.
    """
    features = dict(query)
    for feature, (control, factor) in module.unit_conversions.items():
        if control in features:
            features[feature] = features.pop(control) * factor
    return features


def predicted_diameter(bundle: Bundle, features: dict) -> float | None:
    """Diamètre d'extinction prédit, ou None si le module n'en a pas."""
    if bundle.regressor is None:
        return None
    row = pd.get_dummies(pd.DataFrame([features])[bundle.module.features])
    row = row.reindex(columns=bundle.regressor_columns, fill_value=0).astype(float)
    return float(bundle.regressor.predict(row)[0])


def predicted_rate(bundle: Bundle, features: dict) -> float | None:
    """Constante K de la loi en d², qui fixe le rythme du retrecissement."""
    if bundle.rate_model is None:
        return None
    row = pd.get_dummies(pd.DataFrame([features])[bundle.module.features])
    row = row.reindex(columns=bundle.rate_columns, fill_value=0).astype(float)
    return max(float(bundle.rate_model.predict(row)[0]), 1e-3)


def _row(bundle: Bundle, features: dict) -> pd.DataFrame:
    row = pd.get_dummies(pd.DataFrame([features])[bundle.module.features])
    return row.reindex(columns=bundle.columns, fill_value=0).astype(float)


def probability(bundle: Bundle, features: dict) -> float:
    """Probabilité de la classe 1, alignée sur les colonnes d'entraînement."""
    return float(bundle.model.predict_proba(_row(bundle, features))[0, 1])


def value(bundle: Bundle, features: dict) -> float:
    """Valeur prédite, pour un module a sortie continue."""
    return float(bundle.model.predict(_row(bundle, features))[0])


def cross_validated_error(bundle: Bundle) -> float:
    """Erreur moyenne du module, a afficher a cote de toute prediction.

    Une valeur continue sans son incertitude se lit comme une certitude.
    """
    from sklearn.model_selection import KFold, cross_val_score

    module = bundle.module
    outcome = module.target
    frame = bundle.frame
    X = pd.get_dummies(frame[module.features], drop_first=True).astype(float)
    scores = cross_val_score(
        _estimator(module),
        X,
        frame[outcome],
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="neg_mean_absolute_error",
    )
    return float(-scores.mean())
