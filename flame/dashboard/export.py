"""Export des modèles vers le navigateur — pour supprimer les coupures.

LE PROBLÈME.

Streamlit relance tout le script à chaque mouvement de curseur. Le composant
3D est alors recréé de zéro : l'iframe se vide, la scène se reconstruit,
l'animation repart au début. Visuellement, chaque réglage produit un blanc
suivi d'un saut.

LA SOLUTION, PERMISE PAR UNE PROPRIÉTÉ DES MODÈLES.

Les trois modèles du module Suppression sont LINÉAIRES — une régression
logistique et deux régressions ridge. Un modèle linéaire, une fois sa mise à
l'échelle appliquée, n'est qu'un produit scalaire :

    valeur = somme( (x - moyenne) / ecart_type * coefficient ) + constante

et pour le classifieur, une sigmoïde par-dessus. Cela s'écrit en cinq lignes
de JavaScript. Vérifié : la formule reproduit scikit-learn au quatrième
chiffre après la virgule.

On envoie donc les coefficients au navigateur plutôt que les prédictions. Les
curseurs vivent dans le composant, plus rien ne repasse par le serveur, et la
gouttelette peut glisser d'un état à l'autre au lieu de sauter.

LE VOYANT DE DISTANCE SUIT LE MÊME CHEMIN. Il repose sur la distance au plus
proche essai réel dans l'espace réduit : 206 essais sur 5 variables, soit
1030 nombres. Autant les embarquer aussi, pour que l'avertissement reste
instantané.

CE QUE CELA NE CHANGE PAS. Les seuils, les modèles et les données sont
exactement les mêmes des deux côtés. On déplace un calcul, on ne le simplifie
pas : la page reste la source de vérité de ce qui s'affiche, et Python reste
celle de ce qui est appris.
"""

from __future__ import annotations

import json

import numpy as np

from flame.dashboard.predict import Bundle


def _linear(pipeline, columns: list[str]) -> dict:
    """Un modèle linéaire réduit à ce qu'il faut pour le recalculer ailleurs."""
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    return {
        "columns": columns,
        "mean": [round(float(v), 8) for v in scaler.mean_],
        "scale": [round(float(v), 8) for v in scaler.scale_],
        "coef": [round(float(v), 8) for v in np.ravel(model.coef_)],
        "intercept": round(float(np.ravel(model.intercept_)[0]), 8),
    }


def payload(bundle: Bundle) -> dict:
    """Tout ce dont le composant a besoin pour travailler seul."""
    module = bundle.module
    index = bundle.index
    frame = bundle.frame

    fuels = sorted(frame["fuel"].unique())
    ranges = {
        name: {
            "min": round(float(frame[name].min()), 4),
            "max": round(float(frame[name].max()), 4),
            "values": sorted(round(float(v), 4) for v in frame[name].unique()),
        }
        for name in ["o2_frac", "co2_frac", "he_frac", "pressure_atm", "d0_mm"]
    }

    # Essais reels, pour afficher les voisins sans repasser par le serveur.
    shown = ["fuel", "pressure_atm", "o2_frac", "co2_frac", "he_frac", "d0_mm",
             "dext_mm", "test_end"]
    def _clean(value):
        """JSON n'encode ni NaN ni le NA de pandas : ils deviennent null."""
        import pandas as pd

        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        if isinstance(value, (int, float, np.floating, np.integer)):
            return round(float(value), 3)
        return str(value)

    tests = [
        {key: _clean(row[key]) for key in shown if key in frame.columns}
        for _, row in frame.iterrows()
    ]

    return {
        "fuels": fuels,
        "ranges": ranges,
        "models": {
            "extinction": _linear(bundle.model, bundle.columns),
            "diameter": _linear(bundle.regressor, bundle.regressor_columns),
            "rate": _linear(bundle.rate_model, bundle.rate_columns),
        },
        "errors": {
            "diameter": round(bundle.regressor_error, 4),
            "rate": round(bundle.rate_error, 4),
        },
        "proximity": {
            # Espace reduit de l'index : mêmes moyennes et ecarts-types.
            "features": index.numeric_features,
            "mean": [round(float(v), 8) for v in index.scaler.mean_],
            "scale": [round(float(v), 8) for v in index.scaler.scale_],
            "points": [[round(float(v), 4) for v in row] for row in index.points],
            "typical": round(index.typical_distance, 5),
            "largest": round(index.largest_internal_gap, 5),
        },
        "tests": tests,
        "conversions": {
            feature: {"control": control, "factor": factor}
            for feature, (control, factor) in module.unit_conversions.items()
        },
    }


def as_json(bundle: Bundle) -> str:
    return json.dumps(payload(bundle), separators=(",", ":"))
