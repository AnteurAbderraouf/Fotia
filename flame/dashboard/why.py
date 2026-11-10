"""Panneau « Why it matters » — ce que chaque reglage veut dire pour un vaisseau.

DONNEES SEULEMENT. Ce module ne dessine rien : pour un reglage donne, il
prepare les fiches de contexte (incidents, resultats publies, gravite), la
carte des atmospheres, le controle NASA-STD-3001 et la comparaison avec
l'air de l'ISS. La console (flame/dashboard/console.py) les affiche.

RIEN DE FRANCAIS NE PART A L'ECRAN. Les CSV de data/context sont rediges en
francais ; leurs champs descriptifs ne sont donc JAMAIS affiches tels quels.
On affiche :
  * la citation, deja en anglais dans la source, mot pour mot ;
  * la source, passee par `english_source` ;
  * un resume anglais ecrit ICI, par numero de ligne, fidele aux colonnes du
    CSV (on n'y ajoute aucun fait). Une ligne sans resume anglais s'affiche
    avec sa seule citation, et `untranslated()` la signale a check_app.

LES GARDE-FOUS VIENNENT DU LOADER, PAS D'ICI.
  * `verified` faux partout : chaque fiche dit qu'elle n'est pas verifiee.
  * `established` faux (Bondarenko) : la fiche porte « Secondary source, not
    an established fact » et le resume commence par « Reported: ».
  * La litterature n'a pas de colonne verifie : traitee comme non verifiee.
  * Aucune comparaison chiffree avec la litterature ici (c'est l'etape 5).
    Les seuls calculs du panneau portent sur le moteur lui-meme (rapport a
    l'air de l'ISS) et sur la norme NASA-STD-3001 (ligne 9 des atmospheres).

LES LIENS DE LA LITTERATURE. literature_crosscheck.csv ne donne que le nom du
PDF. Les liens ci-dessous ont ete retrouves le 2026-11-09 : le DOI de Dietrich
2014 est imprime en tete du PDF ; le numero NTRS de Takahashi 2015 aussi ;
les trois autres viennent de l'API de recherche NTRS, titre, auteurs et date
identiques au PDF. Une ligne dont le PDF n'est pas ici s'affiche sans lien.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from flame.dashboard.presets import english_source
from flame.loaders.context import (
    PSIA_PER_ATM,
    load_atmospheres,
    load_gravity,
    load_incidents,
    load_literature,
)
from flame.models.hybrid import MICROGRAVITY_BELOW, Engine, Prediction

# Seuil d'oxygene a partir duquel Apollo 1 et Bondarenko deviennent
# pertinents : les deux atmospheres etaient a 50 % d'O2 ou plus.
RICH_OXYGEN = 0.50

# Lignes de contexte_atmospheres.csv utilisees par nom.
ISS_ROW = 1
STANDARD_ROW = 9

LITERATURE_LINKS = {
    "Cool-flame Extinction During N-Alkane Droplet Combustion in Microgravity.pdf":
        "https://ntrs.nasa.gov/citations/20140006750",
    "Droplet Combustion Experiments Aboard the International Space Station.pdf":
        "https://doi.org/10.1007/s12217-014-9372-2",
    "2013_Nayagam_methanoldroplet-O2-Inert-8thUSNCM.pdf":
        "https://ntrs.nasa.gov/citations/20130014061",
    "Microgravity Droplet Combustion in CO 2 Enriched Environments at Elevated Pressures.pdf":
        "https://ntrs.nasa.gov/citations/20070017858",
    "Combustion Characteristics in a Non-Premixed Cool-Flame Regime of n-Heptane in Microgravity.pdf":
        "https://ntrs.nasa.gov/citations/20150018397",
}

# --- Textes anglais, par numero de ligne ------------------------------------
# Traductions des colonnes descriptives du CSV, sans rien y ajouter.

INCIDENTS_EN = {
    1: (
        "Bondarenko chamber fire, Moscow",
        "Reported: in an isolation chamber of at least 50% oxygen (total "
        "pressure not reported), an alcohol-soaked cotton ball fell on an "
        "electric hotplate. Wool clothing caught fire and the oxygen-rich air "
        "kept it burning. The door was opened after nearly half an hour "
        "because of the pressure difference; death from shock 16 hours "
        "later. Kept secret by Soviet authorities until 1980.",
    ),
    2: (
        "Apollo 1 fire",
        "Command module ground test on the launch pad, 16.7 psia of 100% "
        "oxygen. Most probable initiator: an electrical arc. Many combustible "
        "materials sat near ignition sources, and the fast spread raised "
        "pressure and temperature. The inner hatch could not be opened "
        "against the internal pressure; opening all hatches took about 5 "
        "minutes (Review Board summary). The crew died of asphyxia from toxic "
        "gases. Changes: restrict and control combustibles, shorten egress "
        "time, remove live electrical connections from the cabin.",
    ),
    3: (
        "Fire aboard Mir",
        "Kvant-1 module; cabin atmosphere not given in the source. A "
        "solid-fuel oxygen generator caught fire during activation (probable "
        "cause: latex glove material inside the cartridge). An intense flame "
        "lasted several minutes, dense smoke filled the station and the "
        "flame blocked access to the Soyuz escape vehicle. The crew used "
        "oxygen masks and foam extinguishers; the flame finally went out on "
        "its own, with no structural damage. Changes: emergency training and "
        "cartridge preparation; the lessons were carried into ISS oxygen "
        "systems.",
    ),
    4: (
        "Saffire IV and V, deliberate fires in a Cygnus",
        "Large samples (PMMA, cotton, cotton/fiberglass blend) were set on "
        "fire on purpose inside a Cygnus cargo vehicle after it left the ISS, "
        "to study fire spread and growth. Atmosphere not given in the page "
        "read. Heat release reached 10 kW without the average vehicle "
        "conditions becoming unacceptable, but pressure, temperature and CO2 "
        "all rose measurably.",
    ),
}

INCIDENT_CAVEATS = {
    1: "Secondary source, very few documents. Not an established fact.",
    3: "The source page has no verbatim quote on the cause or the extinguishers.",
}

LITERATURE_EN = {
    1: "n-decane droplet, 4.03 mm, 21% O2 in nitrogen, 0.5 atm. The hot flame "
       "went out by radiative extinction at about 14.6 s, then the droplet "
       "kept burning with a cool flame. K = 0.41 mm²/s hot, 0.33 cool.",
    2: "n-decane droplet, 3.79 mm, 25% O2, 0.5 atm. No radiative extinction: "
       "the droplet micro-exploded 26 s after ignition. K = 0.52 mm²/s.",
    3: "n-decane droplet, 3.86 mm, 17% O2, 0.5 atm. Radiative extinction at "
       "7 s. K = 0.31 mm²/s hot, 0.27 cool.",
    4: "n-decane at 14% O2: both the hot and the cool flame went out earlier. "
       "K = 0.29 mm²/s hot, 0.25 cool. Pressure and droplet size are not "
       "stated in the passage read.",
    5: "n-alkanes (heptane, octane, decane), 14–25% O2, 0.5–1 atm. The size "
       "at which the cool flame goes out scales with oxygen concentration to "
       "the power −3/4 in oxygen/nitrogen. Helium and CO2 do not follow it.",
    6: "Limiting oxygen index in oxygen/nitrogen: 0.12 for methanol, 0.15 for "
       "heptane (O2 mole fraction). Below it, droplet flames do not survive. "
       "Pressure not stated for this result.",
    7: "Adding CO2 raises the limiting oxygen index, to about 0.14 for "
       "methanol (with 30% CO2) and 0.17 for heptane (with 20% CO2), at 1 atm.",
    8: "With helium mixed into air at 1 atm, the limiting oxygen index is "
       "about 0.14 for methanol and 0.15 for heptane.",
    9: "Large heptane droplets: after the visible hot flame goes out by "
       "radiation, the droplet keeps burning with a cool flame, invisible to "
       "the cameras and inferred from other measurements.",
    10: "Methanol and heptane flames, 0.1–0.4 O2, 0.7–3 atm, with nitrogen, "
        "CO2 or helium, oscillated at about 1 Hz just before radiative "
        "extinction.",
    11: "Methanol droplets (about 1.5–4 mm), 12–30% O2, 0.7 and 1 atm, with "
        "CO2 or helium added to nitrogen: the limiting oxygen index is about "
        "12–13% O2 by volume.",
    12: "Methanol droplet, 2.0 mm, 21% O2, at 1, 3 and 5 atm with up to 70% "
        "CO2: the burning rate decreases slightly as CO2 increases.",
    13: "n-heptane droplet, 2.0 mm, 21% O2, at 1, 3 and 5 atm with up to 70% "
        "CO2: the burning rate increases slightly as CO2 increases, the "
        "opposite of what quasi-steady theory predicts.",
    14: "Gaseous n-heptane source in nearly still air at zero gravity. The "
        "flame goes out from radiative heat loss at about 1200 K, then a cool "
        "flame near 700 K stays close to the fuel source for about 18 s.",
}

# Onglet de chaque resultat publie : « limit » = indice limite d'oxygene,
# affiche avec l'oxygene ; « diluent » = effet du CO2 ; « fuel » = le reste.
LITERATURE_TOPIC = {
    1: "fuel", 2: "fuel", 3: "fuel", 4: "fuel", 5: "fuel", 9: "fuel",
    10: "fuel", 14: "fuel",
    6: "limit", 7: "limit", 8: "limit", 11: "limit",
    12: "diluent", 13: "diluent",
}

ATMOSPHERES_EN = {
    1: "ISS, nominal",
    2: "Space Shuttle",
    3: "Apollo and Gemini, in flight",
    4: "Apollo 1, launch pad test",
    5: "Apollo after the fire, on the ground",
    6: "Skylab SMEAT ground test",
    7: "Proposed exploration atmosphere",
    8: "Spacesuit EVA test profile",
    9: "NASA-STD-3001 limits",
    10: "Mars surface habitat, study concept",
    11: "Lunar surface habitat, study concept",
    12: "Space Shuttle, before a spacewalk",
    13: "ISS airlock campout, before a spacewalk",
    14: "Orion, design atmosphere for lunar missions",
    15: "Orion, full cabin pressure at its oxygen ceiling",
    16: "Soyuz, top of its oxygen range",
}

# Cle = premier mot de la colonne « type » (« essai au sol » -> « essai »).
# Noms courts pour les etiquettes de la carte, ou la place manque.
ATMOSPHERES_SHORT = {
    1: "ISS", 2: "Shuttle", 3: "Apollo & Gemini flight", 4: "Apollo 1 pad",
    5: "Apollo after fire", 6: "Skylab SMEAT", 7: "Exploration",
    8: "EVA suit test", 9: "NASA-STD-3001", 10: "Mars habitat",
    11: "Moon habitat", 12: "Shuttle EVA prep", 13: "ISS campout",
    14: "Orion", 15: "Orion max", 16: "Soyuz",
}

ATMOSPHERE_STATUS = {
    "vaisseau": "Spacecraft",
    "essai": "Ground test",
    "proposition": "Proposal",
    "norme": "Standard",
}


def _status_key(cell) -> str:
    return str(cell).split()[0].lower() if isinstance(cell, str) and cell else ""

GRAVITY_EN = {
    1: "Earth",
    2: "Moon",
    3: "Mars",
    4: "ISS microgravity",
}

GRAVITY_CAVEATS = {
    2: "This fact sheet does not cover fire behaviour at partial gravity.",
    4: "A design requirement, not a measured value.",
}

FUEL_WORD = {"Methanol": "methanol", "Heptane": "heptane", "n-decane": "decane"}


@dataclass
class Card:
    """Une fiche de contexte, prete a afficher (tout en anglais)."""

    kind: str  # fire | experiment | result | simulation | reference
    title: str
    summary: str
    quote: str
    source: str
    url: str | None
    verified: bool = False
    labels: list[str] = field(default_factory=list)
    year: str = ""


@dataclass
class Section:
    key: str
    name: str
    why: list[str]
    cards: list[Card] = field(default_factory=list)
    empty: str = ""


# --- Donnees ----------------------------------------------------------------

@st.cache_data
def tables() -> dict[str, pd.DataFrame]:
    return {
        "incidents": load_incidents(),
        "atmospheres": load_atmospheres(),
        "gravity": load_gravity(),
        "literature": load_literature(),
    }


def untranslated(t: dict[str, pd.DataFrame] | None = None) -> list[str]:
    """Lignes de contexte sans texte anglais : elles s'afficheraient a moitie."""
    t = tables() if t is None else t
    gaps = []
    for name, table, known in (
        ("incidents", t["incidents"], INCIDENTS_EN),
        ("atmospheres", t["atmospheres"], ATMOSPHERES_EN),
        ("atmospheres (short)", t["atmospheres"], ATMOSPHERES_SHORT),
        ("gravity", t["gravity"], GRAVITY_EN),
        ("literature", t["literature"], LITERATURE_EN),
    ):
        gaps += [f"{name} #{i}" for i in table["id"] if i not in known]
    lit = t["literature"]
    gaps += [f"literature #{i} (topic)" for i in lit["id"] if i not in LITERATURE_TOPIC]
    gaps += [f"atmospheres type « {k} »" for k in t["atmospheres"]["type"].unique()
             if _status_key(k) not in ATMOSPHERE_STATUS]
    gaps += [f"incidents type « {k} »" for k in t["incidents"]["type"].unique()
             if _incident_kind(k) is None]
    return gaps


def _row(table: pd.DataFrame, row_id: int) -> pd.Series:
    match = table[table["id"] == row_id]
    if match.empty:
        raise KeyError(f"ligne id={row_id} absente de data/context")
    return match.iloc[0]


def _incident_kind(cell) -> str | None:
    """« incident » -> fire ; « experience controlee » -> experiment."""
    text = str(cell).lower()
    if text.startswith("incident"):
        return "fire"
    if text.startswith("exp"):
        return "experiment"
    return None


def _incident(row: pd.Series) -> Card:
    i = int(row["id"])
    title, summary = INCIDENTS_EN.get(i, (f"Recorded fire #{i}", ""))
    labels = []
    if not row["established"]:
        labels.append("Secondary source, not an established fact")
    caveat = INCIDENT_CAVEATS.get(i)
    if caveat and caveat not in summary:
        summary = f"{summary} {caveat}".strip()
    return Card(
        kind=_incident_kind(row["type"]) or "fire",
        title=title,
        summary=summary,
        quote=str(row["citation"]),
        source=english_source(str(row["source"])),
        url=str(row["url"]),
        verified=bool(row["verified"]),
        labels=labels,
        year="" if pd.isna(row["year"]) else str(row["year"]),
    )


def _publication(row: pd.Series) -> Card:
    i = int(row["id"])
    page = "" if pd.isna(row["page"]) else f", p. {row['page']}"
    year = re.search(r"(?:19|20)\d\d", str(row["article"]))
    return Card(
        kind="simulation" if row["is_simulation"] else "result",
        title=f"{row['article']}{page}",
        summary=LITERATURE_EN.get(i, ""),
        quote=str(row["citation"]),
        source=str(row["article"]),
        url=LITERATURE_LINKS.get(str(row["fichier"])),
        labels=["Simulation, not a measurement"] if row["is_simulation"] else [],
        year=year.group(0) if year else "",
    )


def _fuel_matches(row: pd.Series, fuel: str) -> bool:
    return FUEL_WORD[fuel] in str(row["combustible"]).lower()


def _diluent_matches(row: pd.Series, conditions: dict) -> bool:
    """Le diluant de la ligne correspond-il a celui choisi ?"""
    text = str(row["diluant"]).lower()
    has_co2 = "co2" in text
    has_he = re.search(r"\bhe\b", text) is not None
    if conditions["co2_frac"] > 0:
        return has_co2
    if conditions["he_frac"] > 0:
        return has_he
    return not has_co2 and not has_he


def _publications(t, conditions: dict, topic: str, diluent_filter: bool) -> list[Card]:
    cards = []
    for _, row in t["literature"].iterrows():
        if LITERATURE_TOPIC.get(int(row["id"])) != topic:
            continue
        if not _fuel_matches(row, conditions["fuel"]):
            continue
        if diluent_filter and not _diluent_matches(row, conditions):
            continue
        cards.append(_publication(row))
    return cards


# --- Atmospheres, norme, comparaison ----------------------------------------

def atmosphere_points(t) -> list[dict]:
    """Les atmospheres de vaisseaux, pour la carte pression x oxygene.

    Une pression ou un O2 absents restent None : la carte ne place pas le
    point et le dit. La ligne « norme » n'est pas un point, c'est un cadre.
    """
    points = []
    for _, r in t["atmospheres"].iterrows():
        status = ATMOSPHERE_STATUS.get(_status_key(r["type"]), "Other")
        if status == "Standard":
            continue
        lo = r["pressure_atm_min"]
        o2 = r["o2_frac"]
        points.append({
            "id": int(r["id"]),
            "name": ATMOSPHERES_EN.get(int(r["id"]), f"Row {r['id']}"),
            "short": ATMOSPHERES_SHORT.get(int(r["id"]), f"Row {r['id']}"),
            "status": status,
            "pressure": None if pd.isna(lo) or r["pressure_is_range"] else float(lo),
            "o2": None if pd.isna(o2) else float(o2),
            "source": english_source(str(r["source"])),
            "url": str(r["url"]),
            "quote": str(r["citation"]),
            "verified": bool(r["verified"]),
        })
    return points


def standard_check(t, conditions: dict) -> dict:
    """Le reglage tient-il dans les limites NASA-STD-3001 (ligne 9) ?

    Pression dans ]min, max] et au moins X % de diluant, X lu dans la colonne
    « autre ». Le diluant est tout ce qui n'est pas de l'oxygene.
    """
    row = _row(t["atmospheres"], STANDARD_ROW)
    lo, hi = row["pressure_atm_min"], row["pressure_atm_max"]
    numbers = re.findall(r"\d+(?:\.\d+)?", str(row["autre"]))
    out = {
        "source": english_source(str(row["source"])),
        "url": str(row["url"]),
        "quote": str(row["citation"]),
        "verified": bool(row["verified"]),
        "note": "Crew health limits for long exposure. This document gives no "
                "oxygen flammability limit.",
        "inside": None,
        "verdict": "The limits could not be read from the source row.",
        "box": None,
    }
    if pd.isna(lo) or pd.isna(hi) or not numbers:
        return out
    min_diluent = float(numbers[0]) / 100.0
    out["box"] = {"p_min": float(lo), "p_max": float(hi), "o2_max": 1.0 - min_diluent}
    p, o2 = conditions["pressure_atm"], conditions["o2_frac"]
    problems = []
    if p <= lo:
        problems.append(f"pressure {p * PSIA_PER_ATM:.1f} psia is at or below "
                        f"{lo * PSIA_PER_ATM:.1f}")
    if p > hi:
        problems.append(f"pressure {p * PSIA_PER_ATM:.1f} psia is above "
                        f"{hi * PSIA_PER_ATM:.1f}")
    if 1.0 - o2 < min_diluent - 1e-9:
        problems.append(f"only {1.0 - o2:.0%} of the gas is not oxygen "
                        f"(at least {min_diluent:.0%} required)")
    out["inside"] = not problems
    out["verdict"] = ("Outside: " + "; ".join(problems) + "."
                      if problems else "Inside the limits for long crew exposure.")
    return out


def iss_comparison(engine: Engine, t, conditions: dict,
                   prediction: Prediction) -> dict | None:
    """Le moteur, a ce reglage puis dans l'air de l'ISS, meme goutte.

    None quand le reglage EST l'air de l'ISS (un rapport de 1,00 ne dit rien)
    ou quand la ligne ISS n'a pas de chiffres.
    """
    iss = _row(t["atmospheres"], ISS_ROW)
    if pd.isna(iss["pressure_atm_min"]) or pd.isna(iss["o2_frac"]):
        return None
    ref = dict(conditions, pressure_atm=float(iss["pressure_atm_min"]),
               o2_frac=float(iss["o2_frac"]), co2_frac=0.0, he_frac=0.0,
               gravity_g=0.0)
    if all(abs(ref[k] - conditions[k]) < 5e-3
           for k in ("pressure_atm", "o2_frac", "co2_frac", "he_frac")):
        return None
    base = engine.predict(ref)
    if base.k_mm2_s <= 0:
        return None
    return {
        "ratio": prediction.k_mm2_s / base.k_mm2_s,
        "k_iss": base.k_mm2_s,
        "p_iss": base.extinction_probability,
    }


# --- Sections ---------------------------------------------------------------

def sections(conditions: dict, prediction: Prediction,
             t: dict[str, pd.DataFrame] | None = None) -> list[Section]:
    t = tables() if t is None else t
    inc = t["incidents"]
    o2, g = conditions["o2_frac"], conditions["gravity_g"]

    # Oxygene et pression
    why = ["Oxygen feeds the flame and pressure sets how much of it is there. "
           "Spacecraft have flown everything from 21% to 100% oxygen."]
    cards = []
    if o2 >= RICH_OXYGEN:
        rich = [_incident(r) for _, r in inc.iterrows() if int(r["id"]) in (2, 1)]
        cards += sorted(rich, key=lambda c: bool(c.labels))
    cards += _publications(t, conditions, "limit", diluent_filter=True)
    oxygen = Section("oxygen", "Oxygen & pressure", why, cards)

    # Gravite
    grav = t["gravity"]
    gcards = []
    if g < MICROGRAVITY_BELOW:
        gwhy = ["In microgravity hot gas does not rise, so the flame stays a "
                "sphere around the droplet. All the NASA droplet tests behind "
                "Fotia were run in this regime."]
        gcards += [_incident(r) for _, r in inc.iterrows() if int(r["id"]) in (3, 4)]
        # La ligne ISS n'a pas de fraction chiffree (« 1e-6 (exigence) »).
        near = grav[grav["is_requirement"]]
    else:
        gwhy = [f"No droplet experiment in our data was run at {g:.3f} g. "
                "NASA's droplet tests were all in microgravity, so Fotia keeps "
                "microgravity numbers and only the drawn flame shape changes."]
        near = grav[(grav["fraction_of_earth"] - g).abs() <= 0.02]
    for _, r in near.iterrows():
        i = int(r["id"])
        gcards.append(Card(
            kind="reference",
            title=f"{GRAVITY_EN.get(i, f'Gravity row {i}')} gravity",
            summary=GRAVITY_CAVEATS.get(i, ""),
            quote=str(r["citation"]),
            source=english_source(str(r["source"])),
            url=str(r["url"]),
            verified=bool(r["verified"]),
        ))
    gravity = Section("gravity", "Gravity", gwhy, gcards)

    # Diluant
    diluted = conditions["co2_frac"] > 0 or conditions["he_frac"] > 0
    if conditions["co2_frac"] > 0:
        dwhy = ["CO2 was tested on the ISS as a fire suppression agent: adding "
                "it raises the oxygen a droplet flame needs to survive."]
    elif conditions["he_frac"] > 0:
        dwhy = ["Helium is the other inert gas tested on the ISS as a diluent."]
    else:
        dwhy = ["Nitrogen only. Pick CO2 or helium to see what a diluent "
                "changes, and which published results back it."]
    dcards = _publications(t, conditions, "diluent", diluent_filter=True)
    if diluted:
        # L'indice limite d'oxygene avec ce diluant concerne aussi cet onglet.
        dcards += _publications(t, conditions, "limit", diluent_filter=True)
    diluent = Section("diluent", "Diluent", dwhy, dcards,
                      empty="No published result in our sources for this fuel "
                            "with this diluent." if diluted else "")

    # Carburant et goutte
    fwhy = ["Each fuel burns at its own rate and goes out in its own way. "
            "Large droplets can keep burning with an invisible cool flame "
            "after the visible flame goes out."]
    if prediction.k_basis == "surrogate":
        fwhy.append("Fotia has no NASA test for this fuel in its training "
                    "data: its numbers come from heptane and combustion theory. "
                    "The published results below are what was measured.")
    fcards = _publications(t, conditions, "fuel", diluent_filter=False)
    fuel = Section("fuel", "Fuel & droplet", fwhy, fcards,
                   empty="No published result in our sources for this fuel.")

    return [oxygen, gravity, diluent, fuel]
