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


# TOUT CE QUI EST AFFICHE EST EN ANGLAIS : les noms de modules, les etiquettes
# de commande, les unites, les aides et les remarques. Le code et ses
# commentaires restent en francais. Un panneau de commandes a moitie traduit
# serait pire que pas traduit du tout.
MODULES: dict[str, Module] = {
    "suppression": Module(
        key="suppression",
        name="Suppression",
        question="How much CO2 or helium does it take to put this flame out?",
        regime="liquid fuel droplet, microgravity",
        investigation="PSI-69",
        loader=_psi69,
        features=["fuel", "pressure_mmhg", "o2_frac", "co2_frac", "he_frac", "d0_mm"],
        label="extinction",
        outcome_kind="classification",
        ready=True,
        note="The only dataset in the catalogue about putting a fire out.",
        rare_label=0,
        controls=[
            Control("fuel", "Fuel"),
            Control("o2_frac", "Oxygen", "mole fraction"),
            Control("co2_frac", "Added CO2", "mole fraction"),
            Control("he_frac", "Added helium", "mole fraction"),
            Control("pressure_atm", "Pressure", "atm"),
            Control("d0_mm", "Initial droplet diameter", "mm"),
        ],
        unit_conversions={"pressure_mmhg": ("pressure_atm", 760.0)},
    ),
    "cool_flames": Module(
        key="cool_flames",
        name="Cool flames",
        question="Does invisible burning carry on after the visible flame dies?",
        regime="heavy alkane droplet, microgravity",
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
            "Badly skewed label, 85/15, yet the best AUC in the catalogue "
            "(0.913). Farnesane has no test written up: three fuels, not four."
        ),
        controls=[
            Control("fuel", "Fuel"),
            Control("pressure_atm", "Pressure", "atm"),
            Control("o2_frac", "Oxygen", "mole fraction"),
            Control("he_frac", "Helium", "mole fraction"),
            Control("fiber", "Support fibre"),
            Control("d0_mm", "Initial diameter", "mm"),
            Control("ignition_power_w", "Ignition power", "W"),
            Control("ignition_time_ms", "Ignition duration", "ms"),
        ],
    ),
    "sustainment": Module(
        key="sustainment",
        name="Self-sustainment",
        question="Does a gas flame keep itself alive, or die on its own?",
        regime="spherical gas burner, microgravity",
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
            "The best balanced label in the catalogue, 67 % floor. The only "
            "module where a forest beats regression: the two configurations "
            "pull in opposite directions."
        ),
        rare_label=0,
        controls=[
            Control("flame_type", "Configuration",
                    help="Normal: fuel injected into an oxidising surround. "
                         "Inverse: oxygen injected into a fuel-rich surround."),
            Control("fuel", "Fuel"),
            Control("fuel_dilution", "Fuel purity", "1 = pure",
                    help="0.3 means diluted to 30 % in nitrogen."),
            Control("pressure_bar", "Pressure", "bar"),
            Control("fuel_flow_mg_s", "Fuel flow", "mg/s"),
            Control("tad_k", "Adiabatic flame temperature", "K", derived=True,
                    help="COMPUTED from the mixture, not set independently."),
            Control("zst", "Stoichiometric mixture fraction", "", derived=True,
                    help="COMPUTED from the mixture, not set independently."),
        ],
    ),
    "detection": Module(
        key="detection",
        name="Detection",
        question="What do the ISS smoke detectors actually see?",
        regime="heated solid material, microgravity",
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
        target_label="ISS scattering signal",
        target_unit="V",
        target_meaning="high = the detector sees the smoke",
        outcome_kind="regression",
        ready=True,
        note=(
            "No column records an alarm, so this module answers how many volts "
            "and never whether it rings. The obscuration channel is not "
            "modelled: 81 % of its readings are zero."
        ),
        controls=[
            Control("material", "Material"),
            Control("inlet_velocity_cm_s", "Inlet velocity", "cm/s"),
            Control("duration_s", "Heating duration", "s"),
            Control("primary_aging_s", "Primary ageing", "s"),
            Control("primary_mixing_s", "Primary mixing", "s"),
            Control("net_aging_s", "Net ageing", "s"),
        ],
    ),
    "soot": Module(
        key="soot",
        name="Soot",
        question="How much smoke does this fuel make?",
        regime="jet diffusion flame, microgravity",
        investigation="PSI-107",
        loader=_psi107,
        features=["fuel", "fuel_fraction", "nozzle_mm", "coflow_velocity_cm_s"],
        target="smoke_point_mm",
        target_label="Smoke point",
        target_unit="mm",
        target_meaning="short = the fuel smokes heavily",
        outcome_kind="regression",
        ready=True,
        note=(
            "A data leak was found here: NASA's flow columns are measured AT "
            "the smoke point, not set beforehand. Removing them drops the R2 "
            "from 0.98 to 0.54."
        ),
        controls=[
            Control("fuel", "Fuel"),
            Control("fuel_fraction", "Fuel fraction", "1 = pure"),
            Control("nozzle_mm", "Nozzle diameter", "mm"),
            Control("coflow_velocity_cm_s", "Coflow air velocity", "cm/s"),
        ],
    ),
    "materials": Module(
        key="materials",
        name="Materials",
        question="How does this spacecraft material behave?",
        regime="solid material, microgravity",
        investigation="PSI-25",
        loader=_psi25,
        outcome_kind="descriptive",
        ready=True,
        note=(
            "NO trainable label, and that is a finding rather than a retreat: "
            "20 stated outcomes out of 129, and the oxygen consumed reads as "
            "physically impossible in places. This module describes, it does "
            "not predict."
        ),
    ),
    "ground": Module(
        key="ground",
        name="Ground reference",
        question="What are the extinction limits at 1 g?",
        regime="counterflow burner, EARTH GRAVITY",
        investigation="PSI-142",
        loader=_psi142,
        features=["fuel", "fuel_mass_fraction", "extinction_type", "ozone"],
        target="extinction_strain_rate_1_s",
        target_label="Extinction strain rate",
        target_unit="1/s",
        target_meaning="high = the flame withstands stronger stretching",
        outcome_kind="regression",
        ready=True,
        controls=[
            Control("fuel", "Alkane"),
            Control("fuel_mass_fraction", "Fuel richness", "mass fraction"),
            Control("extinction_type", "Flame type",
                    help="cool: a cool flame. hot: a hot flame."),
            Control("ozone", "Added ozone",
                    help="CAREFUL: the campaigns with and without ozone do not "
                         "cover the same fuel richness. Their bands do not "
                         "overlap at all."),
        ],
        note="1 g. Never to be pooled with the ISS without saying so.",
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
                "output": module.outcome_kind,
                "wired in": "yes" if module.ready else "not yet",
                "note": module.note,
            }
            for module in MODULES.values()
        ]
    )
