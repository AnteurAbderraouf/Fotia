"""Preregles d'environnement — chaque chiffre vient d'une ligne de data/context.

UN PREREGLAGE NE CONTIENT AUCUNE VALEUR. Il ne porte que des numeros de
ligne : celle de contexte_atmospheres.csv (pression, O2) et celle de
contexte_gravite.csv (gravite). Les chiffres sont lus au chargement. Corriger
une valeur se fait donc dans le CSV, a un seul endroit, et la page suit.

CE QUI MANQUE RESTE VIDE. Si une ligne n'a pas de pression, le preregle n'en
invente pas : `missing` la nomme, et l'interface le dit. Deux cas assumes :

  * Apollo en vol : aucune ligne de gravite ne decrit une capsule en croisiere.
    La chute libre est une definition, pas une mesure ; la gravite est donc
    mise a zero SANS source chiffree, et la page l'ecrit.
  * Atmosphere d'exploration : la source ne dit pas ou elle s'appliquera, donc
    le preregle ne touche pas a la gravite deja reglee.

LANGUE. Les CSV sont rediges en francais. Ce qui part a l'ecran est en
anglais : les noms et les mises en garde sont ecrits ici, en anglais ; seule
la citation, deja en anglais dans la source, est reprise mot pour mot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from flame.loaders.context import load_atmospheres, load_gravity

# Pres de zero, une gravite s'affiche « microgravity » plutot qu'un nombre :
# 1e-6 g et 0 g sont le meme regime pour une flamme.
MICROGRAVITY_BELOW = 0.01


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    atmosphere_id: int
    gravity_id: int | None
    # « free_fall » : gravite nulle, sans ligne de source. « keep » : la
    # source ne dit rien de la gravite, on ne la touche pas.
    gravity_rule: str = "row"
    caveat: str = ""


PRESETS: list[Preset] = [
    Preset("iss", "ISS", atmosphere_id=1, gravity_id=4),
    Preset(
        "apollo1_pad", "Apollo 1, launch pad test", atmosphere_id=4, gravity_id=1,
        caveat=(
            "Above the 5 psia flown in space. That higher pressure of pure "
            "oxygen is the key difference behind the 1967 fire."
        ),
    ),
    Preset(
        "apollo_flight", "Apollo in flight", atmosphere_id=3, gravity_id=None,
        gravity_rule="free_fall",
    ),
    Preset(
        "shuttle_eva_prep", "Shuttle, before a spacewalk", atmosphere_id=12,
        gravity_id=None, gravity_rule="free_fall",
        caveat=(
            "The Shuttle's staged prebreathe atmosphere: the cabin was lowered "
            "to it before spacewalks, to shorten the time crews breathed pure "
            "oxygen."
        ),
    ),
    Preset(
        "iss_campout", "ISS airlock, before a spacewalk",
        atmosphere_id=13, gravity_id=None, gravity_rule="free_fall",
        caveat=(
            "Only the Quest airlock is lowered to this atmosphere overnight; "
            "the rest of the station stays at sea-level air. Same atmosphere "
            "as the Shuttle before a spacewalk."
        ),
    ),
    Preset(
        "orion_design", "Orion, lunar design",
        atmosphere_id=14, gravity_id=None, gravity_rule="free_fall",
        caveat=(
            "The richest atmosphere Orion is designed for: every material on "
            "board was certified at it. A design limit, not the usual setting."
        ),
    ),
    Preset(
        "orion_full", "Orion at full pressure",
        atmosphere_id=15, gravity_id=None, gravity_rule="free_fall",
        caveat=(
            "A calculated ceiling, not an operating point: above this oxygen "
            "level at sea-level pressure, the material certification would no "
            "longer cover the cabin."
        ),
    ),
    Preset(
        "soyuz_max", "Soyuz, richest air",
        atmosphere_id=16, gravity_id=None, gravity_rule="free_fall",
        caveat=(
            "The richest corner of the range NASA lists for Soyuz: lowest "
            "pressure with highest oxygen pressure, 28.5% oxygen by our "
            "calculation. Nothing says both limits were reached together; the "
            "usual setting is close to sea-level air."
        ),
    ),
    Preset(
        "moon", "Moon surface habitat", atmosphere_id=11, gravity_id=2,
        caveat=(
            "A NASA study concept, explicitly not a final plan. NASA stated in "
            "2022 that Artemis habitat atmospheres were not finalized."
        ),
    ),
    Preset(
        "mars", "Mars surface habitat", atmosphere_id=10, gravity_id=3,
        caveat="A NASA study concept, explicitly not a final plan.",
    ),
    Preset(
        "exploration", "Proposed exploration atmosphere", atmosphere_id=7,
        gravity_id=None, gravity_rule="keep",
        caveat=(
            "A recommendation for future missions, not an atmosphere in "
            "service. The source does not say where it applies, so gravity "
            "is left as you set it."
        ),
    ),
]

BY_KEY = {preset.key: preset for preset in PRESETS}

# Les sources sont libellees en francais dans les CSV (« diapo », « copie »).
# On traduit les quelques mots qui y figurent, sans toucher aux titres.
_SOURCE_WORDS = {
    "diapos": "slides",
    "diapo": "slide",
    "copie UNT": "UNT archive copy",
    "resume de": "summary of",
    "secondaire": "secondary",
    "conclusion": "finding",
    " et ": " and ",
}


def english_source(text: str) -> str:
    for french, english in _SOURCE_WORDS.items():
        text = text.replace(french, english)
    return text


@dataclass
class Citation:
    """Une valeur affichee et ce qui la justifie."""

    what: str
    source: str
    url: str
    quote: str
    verified: bool


@dataclass
class Resolved:
    preset: Preset
    pressure_atm: float | None
    o2_frac: float | None
    gravity_g: float | None  # None = ne pas toucher a la gravite
    missing: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    gravity_note: str = ""


def _row(table: pd.DataFrame, row_id: int, name: str) -> pd.Series:
    match = table[table["id"] == row_id]
    if match.empty:
        raise KeyError(f"{name} : ligne id={row_id} absente de data/context")
    return match.iloc[0]


def _value(cell) -> float | None:
    return None if pd.isna(cell) else float(cell)


def resolve(key: str, atmospheres: pd.DataFrame | None = None,
            gravity: pd.DataFrame | None = None) -> Resolved:
    """Lit les valeurs d'un prereglage dans data/context."""
    preset = BY_KEY[key]
    atmospheres = load_atmospheres() if atmospheres is None else atmospheres
    gravity = load_gravity() if gravity is None else gravity

    atm = _row(atmospheres, preset.atmosphere_id, "contexte_atmospheres")
    # Une plage de pression n'est pas une pression : on n'en choisit pas une.
    pressure = None if atm["pressure_is_range"] else _value(atm["pressure_atm_min"])
    o2 = _value(atm["o2_frac"])
    missing = [name for name, v in (("pressure", pressure), ("oxygen", o2)) if v is None]

    citations = [Citation(
        what="Pressure and oxygen",
        source=english_source(str(atm["source"])),
        url=str(atm["url"]),
        quote=str(atm["citation"]),
        verified=bool(atm["verified"]),
    )]

    note = ""
    if preset.gravity_rule == "row":
        row = _row(gravity, preset.gravity_id, "contexte_gravite")
        g = _value(row["fraction_of_earth"])
        if g is None:
            missing.append("gravity")
        citations.append(Citation(
            what="Gravity",
            source=english_source(str(row["source"])),
            url=str(row["url"]),
            quote=str(row["citation"]),
            verified=bool(row["verified"]),
        ))
        if row["is_requirement"]:
            note = "ISS value is a design requirement, not a measurement."
    elif preset.gravity_rule == "free_fall":
        g = 0.0
        note = "Free fall in flight. No measured value in our sources."
    else:
        g = None
        note = "Not specified by the source: gravity left unchanged."

    return Resolved(preset, pressure, o2, g, missing, citations, note)


def gravity_label(g: float) -> str:
    if g < MICROGRAVITY_BELOW:
        return "microgravity"
    if abs(g - 1.0) < 1e-9:
        return "Earth, 1 g"
    return f"{g:.3f} g"


def main() -> None:
    atmospheres, gravity = load_atmospheres(), load_gravity()
    for preset in PRESETS:
        r = resolve(preset.key, atmospheres, gravity)
        p = "MISSING" if r.pressure_atm is None else f"{r.pressure_atm:.3f} atm"
        o2 = "MISSING" if r.o2_frac is None else f"{r.o2_frac:.2f}"
        g = "inchangee" if r.gravity_g is None else gravity_label(r.gravity_g)
        print(f"{preset.name:34} {p:12} O2 {o2:6} {g:14} manquant={r.missing}")


if __name__ == "__main__":
    main()
