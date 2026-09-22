"""Distance aux données — le voyant central du tableau de bord.

POURQUOI C'EST LA PIÈCE LA PLUS IMPORTANTE.

Une mesure faite sur PSI-69 : avec une grille de curseurs modeste — dix crans
sur l'oxygène, dix sur le CO2, huit sur l'hélium, huit sur la pression — on
obtient 6 400 combinaisons, dont **40 seulement** ont été visitées par au
moins un essai. Autrement dit 99,4 % des positions de curseurs ne
correspondent à aucune expérience.

Un tableau de bord qui répondrait une probabilité à chacune de ces positions,
du même air assuré partout, mentirait par omission dans presque tous les cas.

D'où ce module. À chaque position, il répond à trois questions :

    1. Les valeurs demandées sont-elles seulement dans la plage testée ?
    2. À quelle distance se trouve l'essai réel le plus proche ?
    3. Cette distance est-elle ordinaire au regard du jeu de données, ou
       sommes-nous dans un trou ?

COMMENT LE SEUIL EST FIXÉ — ET POURQUOI IL N'EST PAS ARBITRAIRE.

On ne choisit pas une distance « raisonnable » à la main. On mesure d'abord, à
l'intérieur même du jeu d'entraînement, la distance de chaque essai à son plus
proche voisin. Cette distribution dit ce qu'est un voisinage NORMAL pour ces
données-là. Une requête est ensuite jugée par rapport à cette référence :

    dans le domaine   son plus proche voisin est aussi proche que les essais
                      le sont typiquement entre eux
    extrapolation     plus loin que d'ordinaire, mais dans l'ordre de grandeur
                      du plus grand vide interne du jeu de données
    hors du domaine   au-delà : aucune campagne n'est passée par là

Le seuil est donc calibré par les données elles-mêmes, et se recalibre tout
seul pour un autre module.

Les distances sont calculées sur des variables RÉDUITES, faute de quoi la
pression, en centaines de mmHg, écraserait les fractions molaires comprises
entre 0 et 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Bornes, en multiples de la distance typique entre voisins du jeu lui-meme.
TYPICAL_PERCENTILE = 75
OUTSIDE_PERCENTILE = 100

VERDICTS = {
    "inside": "dans le domaine teste",
    "extrapolation": "extrapolation",
    "outside": "hors du domaine teste",
}


@dataclass
class Proximity:
    """Ce que le tableau de bord doit afficher à côté d'une prédiction."""

    verdict: str
    distance: float
    typical_distance: float
    neighbours: pd.DataFrame
    out_of_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return VERDICTS[self.verdict]

    @property
    def trustworthy(self) -> bool:
        return self.verdict == "inside"

    def explain(self) -> str:
        """Une phrase, destinée à être lue par quelqu'un qui n'a pas le code."""
        if self.out_of_range:
            names = ", ".join(self.out_of_range)
            return (
                f"Aucun essai NASA n'a explore ces valeurs de {names}. "
                "La prediction est une extrapolation pure."
            )
        if self.verdict == "inside":
            return (
                f"{len(self.neighbours)} essais reels se trouvent aussi pres de "
                "cette condition que les essais le sont typiquement entre eux."
            )
        if self.verdict == "extrapolation":
            return (
                "Cette condition tombe dans un vide du plan d'experience : "
                f"l'essai le plus proche est {self.distance / self.typical_distance:.1f} "
                "fois plus loin que d'ordinaire."
            )
        return (
            "Aucune campagne n'est passee par la. La prediction repose "
            "entierement sur la forme du modele, pas sur des observations."
        )


class ProximityIndex:
    """Mesure la distance d'une condition demandée aux essais réellement faits."""

    def __init__(self, frame: pd.DataFrame, features: list[str]):
        self.features = features
        self.frame = frame.reset_index(drop=True)

        numeric = self.frame[features].select_dtypes(include="number")
        self.numeric_features = list(numeric.columns)
        self.categorical_features = [
            name for name in features if name not in self.numeric_features
        ]

        self.scaler = StandardScaler().fit(numeric)
        self.points = self.scaler.transform(numeric)
        self.ranges = {
            name: (float(numeric[name].min()), float(numeric[name].max()))
            for name in self.numeric_features
        }

        # Distance de chaque essai a son plus proche voisin : la reference de
        # ce qu'est un voisinage normal pour CE jeu de donnees.
        gaps = self._pairwise_nearest()
        self.typical_distance = float(np.percentile(gaps, TYPICAL_PERCENTILE))
        self.largest_internal_gap = float(np.percentile(gaps, OUTSIDE_PERCENTILE))

    def _pairwise_nearest(self) -> np.ndarray:
        differences = self.points[:, None, :] - self.points[None, :, :]
        distances = np.sqrt((differences**2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        return distances.min(axis=1)

    def assess(self, query: dict, neighbours: int = 5) -> Proximity:
        """Situe une condition demandée par rapport aux essais réels."""
        out_of_range = {
            name: bounds
            for name, bounds in self.ranges.items()
            if name in query and not bounds[0] <= float(query[name]) <= bounds[1]
        }

        # On passe un DataFrame et non un tableau nu : le scaler a ete ajuste
        # avec des noms de colonnes, et sklearn avertit sinon a chaque appel.
        vector = pd.DataFrame(
            [[float(query[name]) for name in self.numeric_features]],
            columns=self.numeric_features,
        )
        scaled = self.scaler.transform(vector)
        distances = np.sqrt(((self.points - scaled) ** 2).sum(axis=1))

        # Les variables categorielles ne se mesurent pas, elles se filtrent :
        # un essai sur un autre carburant n'est pas un voisin, quelle que soit
        # la proximite de ses autres conditions.
        eligible = np.ones(len(self.frame), dtype=bool)
        for name in self.categorical_features:
            if name in query:
                eligible &= (self.frame[name] == query[name]).to_numpy()
        if not eligible.any():
            eligible = np.ones(len(self.frame), dtype=bool)

        masked = np.where(eligible, distances, np.inf)
        order = np.argsort(masked)[:neighbours]
        nearest = float(masked[order[0]])

        if out_of_range or nearest > self.largest_internal_gap:
            verdict = "outside"
        elif nearest <= self.typical_distance:
            verdict = "inside"
        else:
            verdict = "extrapolation"

        table = self.frame.iloc[order].copy()
        table.insert(0, "distance", np.round(masked[order], 3))

        return Proximity(
            verdict=verdict,
            distance=nearest,
            typical_distance=self.typical_distance,
            neighbours=table,
            out_of_range=out_of_range,
        )
