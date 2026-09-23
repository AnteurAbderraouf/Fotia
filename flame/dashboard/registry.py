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
class Control:
    """Une commande du tableau de bord, declaree par le module.

    `derived` marque une grandeur qui n'est pas un reglage mais une propriete
    CALCULEE du melange — Tad et Zst dans PSI-159. On peut la faire varier
    pour explorer, mais toute combinaison n'est pas physiquement realisable,
    et c'est la carte de couverture qui le dit.
    """

    feature: str
    label: str
    unit: str = ""
    derived: bool = False
    help: str = ""


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
    controls: list[Control] = field(default_factory=list)
    rare_label: int = 0
    unit_conversions: dict = field(default_factory=dict)
    # Pour une sortie continue : ce qu'on predit, son unite, et le sens d'une
    # valeur elevee. Une regression sans unite ni sens ne se lit pas.
    target: str | None = None
    target_label: str = ""
    target_unit: str = ""
    target_meaning: str = ""


def _psi69():
    from flame.loaders.psi69 import load

    return load()


def _psi39():
    from flame.loaders.psi39 import load

    return load()


def _psi159():
    from flame.loaders.psi159 import load

    return load()


def _psi107():
    from flame.loaders.psi107 import load

    return load()


def _psi101():
    from flame.loaders.psi101 import load

    return load()


def _psi142():
    from flame.loaders.psi142 import load

    return load()


def _psi25():
    from flame.loaders.psi25 import load

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
        rare_label=0,
        controls=[
            Control("fuel", "Carburant"),
            Control("o2_frac", "Oxygene", "fraction molaire"),
            Control("co2_frac", "CO2 ajoute", "fraction molaire"),
            Control("he_frac", "Helium ajoute", "fraction molaire"),
            Control("pressure_atm", "Pression", "atm"),
            Control("d0_mm", "Diametre initial de la goutte", "mm"),
        ],
        unit_conversions={"pressure_mmhg": ("pressure_atm", 760.0)},
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
        ready=True,
        rare_label=0,
        note=(
            "Etiquette tres desequilibree, 85/15, mais la meilleure AUC du "
            "catalogue (0.913). Le farnesane n'a aucun essai depouille : trois "
            "carburants, pas quatre."
        ),
        controls=[
            Control("fuel", "Carburant"),
            Control("pressure_atm", "Pression", "atm"),
            Control("o2_frac", "Oxygene", "fraction molaire"),
            Control("he_frac", "Helium", "fraction molaire"),
            Control("fiber", "Fibre de support"),
            Control("d0_mm", "Diametre initial", "mm"),
            Control("ignition_power_w", "Puissance d'allumage", "W"),
            Control("ignition_time_ms", "Duree d'allumage", "ms"),
        ],
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
        ready=True,
        note=(
            "Etiquette la mieux equilibree du catalogue, plancher 67 %. "
            "Seul module ou une foret bat la regression : les deux configurations "
            "ont des effets de signe oppose."
        ),
        rare_label=0,
        controls=[
            Control("flame_type", "Configuration",
                    help="Normale : carburant injecte dans une ambiance oxydante. "
                         "Inverse : oxygene injecte dans une ambiance carburee."),
            Control("fuel", "Carburant"),
            Control("fuel_dilution", "Purete du carburant", "1 = pur",
                    help="0.3 signifie dilue a 30 % dans de l'azote."),
            Control("pressure_bar", "Pression", "bar"),
            Control("fuel_flow_mg_s", "Debit de carburant", "mg/s"),
            Control("tad_k", "Temperature de flamme adiabatique", "K", derived=True,
                    help="CALCULEE a partir du melange, pas reglee independamment."),
            Control("zst", "Fraction de melange stoechiometrique", "", derived=True,
                    help="CALCULEE a partir du melange, pas reglee independamment."),
        ],
    ),
    "detection": Module(
        key="detection",
        name="Detection",
        question="Que voient reellement les detecteurs de fumee de l'ISS ?",
        regime="materiau solide chauffe, microgravite",
        investigation="PSI-101",
        loader=_psi101,
        features=[
            "material",
            "inlet_velocity_cm_s",
            "duration_s",
            "primary_aging_s",
            "primary_mixing_s",
            "net_aging_s",
        ],
        target="iss_scatter_volts",
        target_label="Signal de diffusion ISS",
        target_unit="V",
        target_meaning="eleve = le detecteur voit la fumee",
        outcome_kind="regression",
        ready=True,
        note=(
            "Aucune colonne n'enregistre une alarme : le module repond « combien "
            "de volts », jamais « ca sonne ». Le canal d'obscurcissement n'est pas "
            "modelise, 81 % de ses lectures valent zero."
        ),
        controls=[
            Control("material", "Materiau"),
            Control("inlet_velocity_cm_s", "Vitesse d'entree", "cm/s"),
            Control("duration_s", "Duree de chauffe", "s"),
            Control("primary_aging_s", "Vieillissement primaire", "s"),
            Control("primary_mixing_s", "Melange primaire", "s"),
            Control("net_aging_s", "Vieillissement net", "s"),
        ],
    ),
    "soot": Module(
        key="soot",
        name="Suie",
        question="Quelle quantite de fumee ce carburant produit-il ?",
        regime="flamme de diffusion sur injecteur, microgravite",
        investigation="PSI-107",
        loader=_psi107,
        features=["fuel", "fuel_fraction", "nozzle_mm", "coflow_velocity_cm_s"],
        target="smoke_point_mm",
        target_label="Point de fumee",
        target_unit="mm",
        target_meaning="court = le carburant fume beaucoup",
        outcome_kind="regression",
        ready=True,
        note=(
            "Une fuite de donnees a ete trouvee ici : les debits NASA sont "
            "mesures A L'INSTANT du point de fumee, pas regles avant. Les "
            "retirer fait tomber le R2 de 0.98 a 0.54."
        ),
        controls=[
            Control("fuel", "Carburant"),
            Control("fuel_fraction", "Fraction de carburant", "1 = pur"),
            Control("nozzle_mm", "Diametre de buse", "mm"),
            Control("coflow_velocity_cm_s", "Vitesse du co-courant d'air", "cm/s"),
        ],
    ),
    "materials": Module(
        key="materials",
        name="Materiaux",
        question="Comment ce materiau de vaisseau se comporte-t-il ?",
        regime="materiau solide, microgravite",
        investigation="PSI-25",
        loader=_psi25,
        outcome_kind="descriptive",
        ready=True,
        note=(
            "AUCUNE etiquette entrainable, et c'est un constat pas un renoncement : "
            "20 issues explicites sur 129, et l'oxygene consomme donne des valeurs "
            "physiquement impossibles. Ce module decrit, il ne predit pas."
        ),
    ),
    "ground": Module(
        key="ground",
        name="Reference au sol",
        question="Quelles sont les limites d'extinction a 1 g ?",
        regime="brûleur a contre-courant, GRAVITE TERRESTRE",
        investigation="PSI-142",
        loader=_psi142,
        features=["fuel", "fuel_mass_fraction", "extinction_type", "ozone"],
        target="extinction_strain_rate_1_s",
        target_label="Vitesse d'etirement a l'extinction",
        target_unit="1/s",
        target_meaning="eleve = la flamme resiste a un etirement plus fort",
        outcome_kind="regression",
        ready=True,
        controls=[
            Control("fuel", "Alcane"),
            Control("fuel_mass_fraction", "Richesse", "fraction massique"),
            Control("extinction_type", "Type de flamme",
                    help="cool : flamme froide. hot : flamme chaude."),
            Control("ozone", "Ozone ajoute",
                    help="ATTENTION : les campagnes avec et sans ozone ne "
                         "couvrent pas les memes richesses, leurs bandes ne se "
                         "recouvrent pas du tout."),
        ],
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
