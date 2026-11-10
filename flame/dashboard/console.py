"""La console Fotia — tout l'ecran tient dans UN composant Streamlit.

POURQUOI UN SEUL COMPOSANT. L'ancienne interface empilait des widgets
Streamlit, des iframes et du markdown : trois styles, aucune unite. Ici la
page entiere (lieux, commandes, flamme, reponse, contexte) est un seul
composant `st.components.v2`, dessine par flame/viz/console/. Une seule
feuille de style, une seule palette, un seul ordre de lecture :

    01 Where   02 The air   03 The flame   04 Prediction   05 Why it matters

QUI CALCULE QUOI. Le navigateur ne calcule rien de physique. Il envoie les
reglages (`setStateValue("conditions", ...)`), Python appelle le moteur
hybride et renvoie tout le resultat dans `data`. Le moteur reste ainsi en un
seul exemplaire, en Python. Le JavaScript anime et affiche.

ALLER-RETOUR ET NUMERO DE SEQUENCE. Chaque envoi porte un `seq`. Python le
renvoie avec le resultat ; le navigateur ignore un resultat plus ancien que
son dernier envoi, sinon un curseur deplace vite « sauterait » en arriere.

LA PALETTE. Celle du Horizon Design System de la NASA (le systeme de
nasa.gov), relevee le 2026-11-09 sur terra-ui.netlify.app, qui le reprend :
nasa-red hsl(3 91% 59%), nasa-blue hsl(217 78% 50%), international-orange
hsl(23 83% 53%), active-green hsl(145 67% 57%), gris « carbon ». Polices du
meme systeme : Public Sans et DM Mono (Inter, aussi au systeme, n'est
plus charge : deux familles suffisent). Le guide de marque imprime donne
d'autres valeurs (#0032A0, #E4002B, PMS 286 / 185) : ce sont les couleurs de
l'insigne, pas celles de l'ecran. Aucun logo NASA n'est utilise.
"""

from __future__ import annotations

import math
from pathlib import Path

import streamlit as st

from flame.dashboard.predict import build
from flame.dashboard.presets import PRESETS, gravity_label, resolve
from flame.dashboard.registry import MODULES
from flame.dashboard.why import (
    atmosphere_points,
    iss_comparison,
    sections,
    standard_check,
    tables,
)
from flame.loaders.context import PSIA_PER_ATM
from flame.models.hybrid import FUELS, LABELS, MICROGRAVITY_BELOW, SURROGATE, Engine
from flame.viz.fotia_flame import BADGES, payload as flame_payload
from flame.loaders.msl1 import load as _load_msl1

ASSETS = Path(__file__).resolve().parents[1] / "viz" / "console"
KEY = "fotia_console"
STATE = "fotia_conditions"
CUSTOM = "custom"

# MASQUES POUR L'INSTANT (decision du 2026-11-09). Aucune donnee de goutte
# n'existe a gravite lunaire ou martienne, et les atmospheres des habitats
# sont des concepts d'etude. Rien n'est supprime : les preregles restent dans
# presets.py et les lignes dans data/context ; il suffit de vider ces deux
# ensembles pour les faire revenir.
# Gouttes MSL-1 qui nourrissent la constante K (voir flame/loaders/msl1.py).
MSL1_TESTS = len(_load_msl1())

# Apollo 1 et Apollo en vol masques le meme jour : 100 % d'oxygene, bien
# au-dela de tout essai de goutte connu (FLEX s'arrete a 34 %, MSL-1 a 40 %
# avec K mesure). La console ne propose que des lieux qu'elle sait
# repondre a partir d'essais NASA. Ils restent sur la carte, comme contexte.
MASKED_PRESETS = {"moon", "mars", "apollo1_pad", "apollo_flight"}
MASKED_ATMOSPHERES = {10, 11}  # habitats martien et lunaire (contexte_atmospheres)


def visible_presets():
    return [p for p in PRESETS if p.key not in MASKED_PRESETS]

# Bornes des commandes. Elles depassent volontairement les essais : sortir
# des donnees est permis, c'est signale.
LIMITS = {
    "pressure_atm": (0.10, 5.00),
    "o2_frac": (0.10, 1.00),
    "gravity_g": (0.0, 1.0),
    "diluent_frac": (0.0, 0.70),
    "d0_mm": (0.5, 6.0),
}
DILUENTS = ("none", "co2", "he")

BADGE_TEXT = {
    "inside": "A NASA test was run close to this setting",
    "extrapolation": "Between NASA tests: each value was tested, not this mix",
    "outside": "No NASA test this far out. Treat the numbers as a trend",
}

BASIS = {
    "learned": "learned from NASA tests",
    "physics": "NASA tests + combustion theory",
    "surrogate": "theory, from heptane",
}


@st.cache_resource
def _engine() -> tuple[Engine, int]:
    bundle = build(MODULES["suppression"])
    return Engine(bundle), len(bundle.frame)


def _component():
    """Enregistre le composant a CHAQUE passage, sans cache.

    Le registre appartient au runtime Streamlit : mis en cache, le composant
    resterait enregistre dans un runtime precedent (AppTest en cree un par
    rendu) et ce runtime-ci repondrait « not registered ». Reenregistrer une
    definition identique ne produit aucun avertissement, et une modification
    du HTML, du CSS ou du JS est prise au rendu suivant.
    """
    read = lambda name: (ASSETS / name).read_text(encoding="utf-8")  # noqa: E731
    return st.components.v2.component(
        "fotia_console",
        html=read("console.html"),
        css=read("console.css"),
        # flame3d.js declare la classe Flame3D, console.js l'utilise.
        js=read("flame3d.js") + "\n" + read("console.js"),
    )


# --- Reglages ---------------------------------------------------------------

def _clamp(value, lo: float, hi: float, fallback: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(v):
        return fallback
    return min(max(v, lo), hi)


def initial() -> dict:
    """Le premier ecran : l'ISS, heptane, goutte de 2,5 mm."""
    iss = resolve("iss", *_context())
    return {
        "preset": "iss",
        "fuel": "Heptane",
        "pressure_atm": round(iss.pressure_atm or 1.0, 3),
        "o2_frac": round(iss.o2_frac or 0.21, 3),
        "gravity_g": round(iss.gravity_g or 0.0, 3),
        "diluent": "none",
        "diluent_frac": 0.0,
        "d0_mm": 2.5,
        "seq": 0,
    }


def sanitize(raw: dict | None) -> dict:
    """Ce que le navigateur envoie, ramene a des valeurs sures."""
    base = initial()
    if not isinstance(raw, dict):
        return base
    out = dict(base)
    for name, (lo, hi) in LIMITS.items():
        out[name] = _clamp(raw.get(name), lo, hi, base[name])
    out["fuel"] = raw.get("fuel") if raw.get("fuel") in FUELS else base["fuel"]
    out["diluent"] = raw.get("diluent") if raw.get("diluent") in DILUENTS else "none"
    if out["diluent"] == "none":
        out["diluent_frac"] = 0.0
    keys = {p.key for p in visible_presets()} | {CUSTOM}
    out["preset"] = raw.get("preset") if raw.get("preset") in keys else CUSTOM
    try:
        out["seq"] = int(raw.get("seq", 0))
    except (TypeError, ValueError):
        out["seq"] = 0
    return out


def engine_conditions(c: dict) -> dict:
    frac = c["diluent_frac"] if c["diluent"] != "none" else 0.0
    return {
        "fuel": c["fuel"],
        "pressure_atm": c["pressure_atm"],
        "o2_frac": c["o2_frac"],
        "gravity_g": c["gravity_g"],
        "co2_frac": frac if c["diluent"] == "co2" else 0.0,
        "he_frac": frac if c["diluent"] == "he" else 0.0,
        "d0_mm": c["d0_mm"],
    }


# --- Donnees envoyees au navigateur -----------------------------------------

def _context():
    t = tables()
    return t["atmospheres"], t["gravity"]


def presets_payload() -> list[dict]:
    atmospheres, gravity = _context()
    out = []
    for preset in visible_presets():
        r = resolve(preset.key, atmospheres, gravity)
        out.append({
            "key": preset.key,
            "name": preset.name,
            "pressure_atm": r.pressure_atm,
            "o2_frac": r.o2_frac,
            "gravity_g": r.gravity_g,
            "missing": r.missing,
            "caveat": preset.caveat,
            "gravity_note": r.gravity_note,
            "citations": [
                {"what": c.what, "source": c.source, "url": c.url,
                 "quote": c.quote, "verified": c.verified}
                for c in r.citations
            ],
        })
    return out


def ranges_payload(engine: Engine) -> dict:
    """La plage testee de chaque grandeur, par carburant. None = jamais teste."""
    out = {}
    for fuel in FUELS:
        if fuel in SURROGATE:
            out[fuel] = None
            continue
        r = engine.ranges[fuel]
        out[fuel] = {name: [float(lo), float(hi)] for name, (lo, hi) in r.items()}
        out[fuel]["gravity_g"] = [0.0, MICROGRAVITY_BELOW]
    return out


def _card(card) -> dict:
    return {
        "kind": card.kind, "title": card.title, "summary": card.summary,
        "quote": "" if card.quote in ("", "nan") else card.quote,
        "source": card.source, "url": card.url, "verified": card.verified,
        "labels": card.labels, "year": card.year,
    }


def result_payload(c: dict) -> dict:
    """Le moteur, le contexte et tout ce que la console affiche pour `c`."""
    engine, _ = _engine()
    t = tables()
    cond = engine_conditions(c)
    gas = cond["o2_frac"] + cond["co2_frac"] + cond["he_frac"]
    out = {
        "pressure_psia": c["pressure_atm"] * PSIA_PER_ATM,
        "o2_partial_atm": c["pressure_atm"] * c["o2_frac"],
        "gravity_label": gravity_label(c["gravity_g"]),
        "standard": standard_check(t, cond),
        "error": None,
        "prediction": None,
        "why": [],
    }
    if gas > 1.0 + 1e-9:
        out["error"] = (f"Oxygen plus added diluent make {gas:.0%} of the gas. "
                        "Lower one of them to light the flame.")
        return out

    p = engine.predict(cond)
    label, colour = BADGES[p.verdict]
    compare = iss_comparison(engine, t, cond, p)
    out["prediction"] = {
        "verdict": p.verdict,
        "badge": label,
        "badge_text": BADGE_TEXT[p.verdict],
        "k": p.k_mm2_s,
        "k_basis": BASIS[p.k_basis],
        "p_ext": p.extinction_probability,
        "ext_basis": p.extinction_basis,
        "extinguishes": p.extinguishes,
        "dext": p.dext_mm,
        "duration": p.duration_s,
        "extrapolated": [LABELS[name] for name in p.extrapolated],
        "notes": p.notes,
        "compare": compare,
        "flame": flame_payload(p, c["d0_mm"], c["gravity_g"]),
    }
    out["why"] = [
        {"key": s.key, "name": s.name, "why": s.why, "empty": s.empty,
         "cards": [_card(card) for card in s.cards]}
        for s in sections(cond, p, t)
    ]
    return out


def payload(c: dict) -> dict:
    engine, tests = _engine()
    return {
        "conditions": c,
        "limits": {k: list(v) for k, v in LIMITS.items()},
        "fuels": FUELS,
        "surrogate": SURROGATE,
        "presets": presets_payload(),
        "ranges": ranges_payload(engine),
        "atmospheres": [a for a in atmosphere_points(tables())
                        if a["id"] not in MASKED_ATMOSPHERES],
        "tests": tests,
        "msl1_tests": MSL1_TESTS,
        "result": result_payload(c),
    }


# --- Streamlit ----------------------------------------------------------------

def _on_change() -> None:
    state = st.session_state.get(KEY) or {}
    st.session_state[STATE] = sanitize(state.get("conditions"))


def render() -> None:
    if STATE not in st.session_state:
        st.session_state[STATE] = initial()
    conditions = sanitize(st.session_state[STATE])
    _component()(
        key=KEY,
        data=payload(conditions),
        default={"conditions": conditions},
        on_conditions_change=_on_change,
    )
