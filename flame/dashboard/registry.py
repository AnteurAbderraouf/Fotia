"""Registre des modules — quel modèle est compétent pour quelle question.

L'AIGUILLEUR, ET POURQUOI IL EXISTE.

Le §8 du handoff a tranché : pas de modèle unique sur toutes les données. Les
jeux ne partagent ni entrées ni sorties, et fabriquer une colonne « risque »
commune serait malhonnête.

Une interface unifiée n'est pourtant pas un modèle unifié. Le tableau de bord
présente une seule flamme et un seul jeu de commandes ; derrière, c'est le
module compétent pour le régime choisi qui répond, avec ses propres features
et sa propre sortie. Ce fichier tient la liste de ces modules et de ce que
chacun sait faire.

Une règle qui ne se négocie pas : un module ne répond QUE dans son régime.
Demander à PSI-69, qui ne connaît que des gouttelettes de méthanol et
d'heptane, ce que fait une plaque de PMMA, n'aurait aucun sens — et rien dans
les scores ne le signalerait.

ÉTAT. Seul le module Suppression est branché sur un modèle entraîné. Les
autres sont déclarés avec ce qu'ils couvrent, pour que l'aiguilleur soit
complet dès maintenant et que l'ajout d'un module se réduise à remplir son
entrée.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Module:
    """Un module : un régime physique, une question, un jeu de données."""

    key: str
    name: str
    question: str
    regime: str
    investigation: str
    loader: Callable[[], pd.DataFrame] | None = None
    features: list[str] = field(default_factory=list)
    label: str | None = None
    outcome_kind: str = "classification"
    ready: bool = False
    note: str = ""


def _psi69():
    from flame.loaders.psi69 import load

    return load()


def _psi39():
    from flame.loaders.psi39 import load

    return load()


def _psi159():
    from flame.loaders.psi159 import load

    return load()


MODULES: dict[str, Module] = {
    "suppression": Module(
        key="suppression",
        name="Suppression",
        question="Combien de CO2 ou d'helium faut-il pour eteindre cette flamme ?",
        regime="gouttelette de carburant liquide, microgravite",
        investigation="PSI-69",
        loader=_psi69,
        features=["fuel", "pressure_mmhg", "o2_frac", "co2_frac", "he_frac", "d0_mm"],
        label="extinction",
        outcome_kind="classification",
        ready=True,
        note="Seul jeu du catalogue portant sur la suppression d'incendie.",
    ),
    "cool_flames": Module(
        key="cool_flames",
        name="Flammes froides",
        question="Une combustion invisible persiste-t-elle apres l'extinction visible ?",
        regime="gouttelette d'alcane lourd, microgravite",
        investigation="PSI-39",
        loader=_psi39,
        features=[
            "fuel",
            "pressure_atm",
            "o2_frac",
            "he_frac",
            "fiber",
            "d0_mm",
            "ignition_power_w",
            "ignition_time_ms",
        ],
        label="cool_flame",
        outcome_kind="classification",
        ready=False,
        note="Etiquette tres desequilibree, 85/15. Le farnesane n'a aucune ligne exploitable.",
    ),
    "sustainment": Module(
        key="sustainment",
        name="Auto-entretien",
        question="Une flamme de gaz se maintient-elle seule ou s'eteint-elle ?",
        regime="brûleur a gaz spherique, microgravite",
        investigation="PSI-159",
        loader=_psi159,
        features=[
            "fuel",
            "fuel_dilution",
            "pressure_bar",
            "tad_k",
            "zst",
            "fuel_flow_mg_s",
            "flame_type",
        ],
        label="self_extinguished",
        outcome_kind="classification",
        ready=False,
        note="Etiquette la mieux equilibree du catalogue, plancher 67 %.",
    ),
    "detection": Module(
        key="detection",
        name="Detection",
        question="Que voient reellement les detecteurs de fumee de l'ISS ?",
        regime="materiau solide chauffe, microgravite",
        investigation="PSI-101",
        outcome_kind="regression",
        ready=False,
        note="Regression sur la reponse en volts. Aucune colonne n'enregistre une alarme.",
    ),
    "soot": Module(
        key="soot",
        name="Suie",
        question="Quelle quantite de fumee ce carburant produit-il ?",
        regime="flamme de diffusion sur injecteur, microgravite",
        investigation="PSI-107",
        outcome_kind="regression",
        ready=False,
        note="Point de fumee. Trois plateformes distinctes, a ne pas melanger.",
    ),
    "materials": Module(
        key="materials",
        name="Materiaux",
        question="Comment ce materiau de vaisseau se comporte-t-il ?",
        regime="materiau solide, microgravite",
        investigation="PSI-25",
        outcome_kind="descriptive",
        ready=False,
        note="AUCUNE etiquette entrainable : 20 issues explicites sur 129. Catalogue seul.",
    ),
    "ground": Module(
        key="ground",
        name="Reference au sol",
        question="Quelles sont les limites d'extinction a 1 g ?",
        regime="brûleur a contre-courant, gravite terrestre",
        investigation="PSI-142",
        outcome_kind="regression",
        ready=False,
        note="1 g. Ne jamais mettre en commun avec l'ISS sans le dire.",
    ),
}


def ready_modules() -> dict[str, Module]:
    return {key: module for key, module in MODULES.items() if module.ready}


def coverage_summary() -> pd.DataFrame:
    """Tableau de ce que le tableau de bord sait faire, et ne sait pas encore."""
    return pd.DataFrame(
        [
            {
                "module": module.name,
                "investigation": module.investigation,
                "regime": module.regime,
                "sortie": module.outcome_kind,
                "branche": "oui" if module.ready else "pas encore",
                "remarque": module.note,
            }
            for module in MODULES.values()
        ]
    )
