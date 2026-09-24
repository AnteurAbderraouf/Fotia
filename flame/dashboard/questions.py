"""Les cinq questions — l'entrée par ce qu'on veut savoir, pas par le régime.

LE REPROCHE AUQUEL CE FICHIER REPOND.

Le tableau de bord etait organise comme on l'a construit, pas comme on y
arrive. Un visiteur ne pense pas « je veux PSI-159 », il pense « est-ce que ce
materiau est dangereux ». Il devait pourtant choisir un regime physique avant
de pouvoir poser sa question.

Ce fichier regroupe les sept modules en cinq questions humaines. Le numero PSI
ne disparait pas : il passe en source, a un clic.

RIEN N'EST MIS EN COMMUN AU PASSAGE. Regrouper deux modules sous une meme
question ne les fusionne pas. « Will the detector see it ? » couvre la
detection ET la suie parce que les deux repondent a la meme preoccupation,
mais chacun garde son jeu de donnees, ses entrees et sa sortie. Le §8 du
handoff tient toujours : une interface unifiee n'est pas un modele unifie.

L'ORDRE EST CELUI D'UN INCENDIE. Quels materiaux brulent, comment l'eteindre,
est-ce vraiment eteint, le detecteur le verra-t-il, la flamme se maintient-elle
seule. C'est la sequence que suit quelqu'un qui pense securite incendie, et
elle porte donc une information : les numerotations decoratives n'en portent
aucune.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    """Une question, et les modules qui savent y répondre."""

    key: str
    label: str
    blurb: str
    modules: list[str] = field(default_factory=list)
    # « predicts » : une probabilite. « estimates » : une valeur continue.
    # « describes » : aucune etiquette entrainable, et c'est un constat mesure.
    kind: str = "predicts"


QUESTIONS: dict[str, Question] = {
    "materials": Question(
        key="materials",
        label="Which materials burn in space?",
        blurb=(
            "Real ISS burns of PMMA, cotton-fibreglass and Nomex, plus the "
            "orbit-versus-Earth comparison on nine Saffire samples."
        ),
        modules=["materials"],
        kind="describes",
    ),
    "suppression": Question(
        key="suppression",
        label="How do I put a fire out?",
        blurb=(
            "How much CO2 or helium it takes to extinguish a droplet flame, "
            "and the diameter the flame dies at."
        ),
        modules=["suppression"],
        kind="predicts",
    ),
    "cool_flames": Question(
        key="cool_flames",
        label="Is it really out?",
        blurb=(
            "Whether an invisible cool flame keeps burning after the visible "
            "one dies. It usually does."
        ),
        modules=["cool_flames"],
        kind="predicts",
    ),
    "detection": Question(
        key="detection",
        label="Will the detector see it?",
        blurb=(
            "What the ISS smoke sensors actually register, and how much soot "
            "each fuel produces."
        ),
        modules=["detection", "soot"],
        kind="estimates",
    ),
    "sustainment": Question(
        key="sustainment",
        label="Will this flame sustain itself?",
        blurb=(
            "Whether a gas flame keeps itself alive or self-extinguishes, in "
            "both burner configurations."
        ),
        modules=["sustainment"],
        kind="predicts",
    ),
    # PSI-142 merite sa propre entree. Le fondre dans la suppression
    # suggererait qu'on peut mettre 1 g et ISS en commun, ce que le §9 du
    # handoff interdit. Le reserver au niveau 3 le cacherait. Il reste donc
    # visible, et son regime est annonce dans le titre meme.
    "ground": Question(
        key="ground",
        label="What do ground tests say? (1 g)",
        blurb=(
            "Extinction limits measured on a counterflow burner on Earth, with "
            "and without added ozone. A reference point, never to be pooled "
            "with the orbital results."
        ),
        modules=["ground"],
        kind="estimates",
    ),
}

KIND_LABELS = {
    "predicts": "Predicts odds",
    "estimates": "Estimates a value",
    "describes": "Catalogue only",
}

# Vocabulaire FIXE du feu tricolore. Il apparait a l'identique partout dans
# l'interface : trois etats, trois formulations, jamais reformulees.
CONFIDENCE = {
    "inside": (
        "Measured here",
        "good",
        "Real tests sit at this condition. The nearest is as close as tests "
        "typically are to each other.",
    ),
    "extrapolation": (
        "Between tests",
        "warning",
        "This falls in a gap in the test matrix. The nearest run is further "
        "away than usual.",
    ),
    "outside": (
        "Never tested",
        "critical",
        "No campaign went here. The number comes from the model's shape, not "
        "from observation.",
    ),
}


def for_module(module_key: str) -> Question | None:
    """La question à laquelle un module répond."""
    for question in QUESTIONS.values():
        if module_key in question.modules:
            return question
    return None
