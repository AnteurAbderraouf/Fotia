"""Les outils de l'assistant — le moteur et les tables, jamais sa memoire.

POURQUOI DES OUTILS. Un modele de langage invente volontiers un chiffre
plausible. Ici, tout chiffre sur une flamme vient de `Engine.predict` (le
meme appel que la console) et toute source vient de data/context. Le modele
ne fait que choisir l'outil, puis expliquer le resultat.

Chaque outil :
  * prend des arguments simples (nombres, mots), decrits par un schema JSON
    neutre, que gemini.py traduit pour l'API ;
  * repond par un dict serialisable en JSON, en anglais ;
  * ne leve jamais : une erreur revient comme {"error": ...}, que le modele
    lit et explique.

Les reglages passent par `sanitize` de la console : l'assistant ne peut pas
demander au moteur une valeur que les curseurs refuseraient.
"""

from __future__ import annotations

import math

import pandas as pd

from flame.dashboard.console import (
    LIMITS,
    MASKED_ATMOSPHERES,
    MSL1_TESTS,
    _engine,
    presets_payload,
    ranges_payload,
    result_payload,
    sanitize,
)
from flame.dashboard.presets import english_source
from flame.dashboard.why import (
    ATMOSPHERE_STATUS,
    ATMOSPHERES_EN,
    GRAVITY_CAVEATS,
    GRAVITY_EN,
    INCIDENT_CAVEATS,
    INCIDENTS_EN,
    LITERATURE_EN,
    LITERATURE_LINKS,
    _status_key,
    tables,
)
from flame.loaders.context import PSIA_PER_ATM
from flame.models.hybrid import FUELS, SURROGATE

DILUENT_NAMES = {"none": "nitrogen only", "co2": "CO2 added", "he": "helium added"}


# --- Mise en forme ------------------------------------------------------------

def _clean(value):
    """Un nombre lisible, sans NaN ni numpy, pour le JSON."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 4)
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _setting(c: dict) -> dict:
    return {
        "fuel": c["fuel"],
        "pressure_atm": round(c["pressure_atm"], 3),
        "pressure_psia": round(c["pressure_atm"] * PSIA_PER_ATM, 2),
        "oxygen_percent": round(100 * c["o2_frac"], 1),
        "gravity_g": round(c["gravity_g"], 4),
        "diluent": DILUENT_NAMES[c["diluent"]],
        "diluent_percent": round(100 * c["diluent_frac"], 1),
        "droplet_diameter_mm": round(c["d0_mm"], 2),
        "place_button": c.get("preset", "custom"),
    }


def _prediction(c: dict) -> dict:
    """Ce que la console affiche pour `c`, en clair."""
    r = result_payload(c)
    out = {"setting": _setting(c)}
    if r["error"]:
        out["error"] = r["error"]
        return out
    p = r["prediction"]
    out["confidence_badge"] = p["badge"]
    out["confidence_meaning"] = p["badge_text"]
    out["burning_rate_constant_K_mm2_per_s"] = _clean(p["k"])
    out["K_basis"] = p["k_basis"]
    if p["compare"]:
        out["K_relative_to_ISS_air"] = round(p["compare"]["ratio"], 3)
    if p["p_ext"] is None:
        out["extinction"] = "not modelled: no NASA extinction test for this fuel"
    else:
        out["extinction_probability"] = round(p["p_ext"], 3)
        out["extinction_basis"] = {
            "learned": "learned from NASA FLEX tests",
            "edge": "value at the edge of the tested range (not extrapolated)",
        }.get(p["ext_basis"], p["ext_basis"])
        out["predicted_outcome"] = (
            f"extinguishes at {p['dext']:.2f} mm" if p["extinguishes"] and p["dext"]
            else "burns to completion")
    out["burn_time_s"] = _clean(p["duration"])
    out["values_beyond_the_tests"] = p["extrapolated"]
    out["engine_notes"] = p["notes"]
    out["nasa_std_3001_crew_limits"] = r["standard"]["verdict"]
    if c["gravity_g"] >= 0.01:
        out["gravity_warning"] = (
            "No droplet test exists above microgravity: the numbers are the "
            "microgravity ones; only the drawn flame shape changes, and it is "
            "illustrative.")
    return out


# --- Les outils -----------------------------------------------------------------

def get_current_setting(current: dict) -> dict:
    """Le reglage que le visiteur regarde, et ce que la console en dit."""
    return _prediction(sanitize(current))


def predict(current: dict, fuel: str | None = None, pressure_atm: float | None = None,
            o2_percent: float | None = None, gravity_g: float | None = None,
            diluent: str | None = None, diluent_percent: float | None = None,
            droplet_mm: float | None = None) -> dict:
    """Le moteur a un autre reglage. Ce qui n'est pas donne reste celui de
    la console, pour que « et avec plus d'oxygene ? » garde le reste."""
    c = dict(sanitize(current))
    if fuel is not None:
        match = {f.lower(): f for f in FUELS}
        key = str(fuel).strip().lower().replace("n-", "").replace("normal ", "")
        alias = {"heptane": "Heptane", "methanol": "Methanol", "decane": "n-decane"}
        chosen = match.get(str(fuel).strip().lower()) or alias.get(key)
        if chosen is None:
            return {"error": f"Unknown fuel '{fuel}'. Fotia has: {', '.join(FUELS)}."}
        c["fuel"] = chosen
    if pressure_atm is not None:
        c["pressure_atm"] = pressure_atm
    if o2_percent is not None:
        c["o2_frac"] = o2_percent / 100.0
    if gravity_g is not None:
        c["gravity_g"] = gravity_g
    if diluent is not None:
        d = str(diluent).strip().lower()
        c["diluent"] = {"nitrogen": "none", "n2": "none", "none": "none",
                        "co2": "co2", "carbon dioxide": "co2",
                        "he": "he", "helium": "he"}.get(d, "invalid")
        if c["diluent"] == "invalid":
            return {"error": "diluent must be nitrogen, co2 or helium."}
    if diluent_percent is not None:
        c["diluent_frac"] = diluent_percent / 100.0
    if droplet_mm is not None:
        c["d0_mm"] = droplet_mm
    c["preset"] = "custom"
    clamped = sanitize(c)
    out = _prediction(clamped)
    moved = [k for k in LIMITS if abs(float(c[k]) - clamped[k]) > 1e-9]
    if moved:
        out["clamped_to_console_limits"] = {
            k: {"asked": c[k], "used": clamped[k], "allowed": list(LIMITS[k])}
            for k in moved}
    return out


def list_places(current: dict | None = None) -> dict:
    """Les boutons de lieu de la section 01 Where, avec leurs sources."""
    places = []
    for p in presets_payload():
        places.append({
            "button": p["name"],
            "pressure_atm": _clean(p["pressure_atm"]),
            "pressure_psia": _clean(p["pressure_atm"] * PSIA_PER_ATM) if p["pressure_atm"] else None,
            "oxygen_percent": _clean(100 * p["o2_frac"]) if p["o2_frac"] is not None else None,
            "gravity": ("left as set by the visitor" if p["gravity_g"] is None
                        else "microgravity" if p["gravity_g"] < 0.01 else f"{p['gravity_g']} g"),
            "caveat": p["caveat"],
            "gravity_note": p["gravity_note"],
            "sources": [{"source": s["source"], "url": s["url"], "quote": s["quote"],
                         "verified_against_original": s["verified"]}
                        for s in p["citations"]],
        })
    places.append({"button": "Your own mix",
                   "caveat": "Keeps the current values and lets the visitor set every slider."})
    return {"places": places,
            "hidden": ("Apollo 1, Apollo in flight, Moon and Mars habitats are not "
                       "offered as buttons: 100% oxygen and partial gravity are beyond "
                       "every droplet test. They remain on the atmosphere map.")}


def tested_ranges(current: dict | None = None) -> dict:
    """Ce que NASA a reellement teste, par carburant."""
    engine, tests = _engine()
    out = {"flex_tests_for_extinction": tests, "msl1_heptane_droplets_for_K": MSL1_TESTS,
           "fuels": {}}
    for fuel, r in ranges_payload(engine).items():
        if r is None:
            out["fuels"][fuel] = (f"never tested by NASA in these data sets; computed "
                                  f"from {SURROGATE[fuel]} with combustion theory")
            continue
        out["fuels"][fuel] = {
            "pressure_atm": [round(v, 3) for v in r["pressure_atm"]],
            "oxygen_percent": [round(100 * v, 1) for v in r["o2_frac"]],
            "added_co2_percent": [round(100 * v, 1) for v in r["co2_frac"]],
            "added_helium_percent": [round(100 * v, 1) for v in r["he_frac"]],
            "droplet_mm": [round(v, 2) for v in r["d0_mm"]],
            "gravity": "microgravity only",
        }
        er = engine.ext_ranges.get(fuel)
        if er:
            out["fuels"][fuel]["extinction_model_range"] = {
                "pressure_atm": [round(float(v), 3) for v in er["pressure_atm"]],
                "oxygen_percent": [round(100 * float(v), 1) for v in er["o2_frac"]],
            }
    return out


def _match(text: str, query: str) -> bool:
    words = [w for w in str(query or "").lower().split() if len(w) > 2]
    return not words or any(w in text.lower() for w in words)


def search_sources(current: dict | None = None, topic: str = "all",
                   query: str = "") -> dict:
    """Les fiches de contexte : incendies, atmospheres, gravite, articles."""
    t = tables()
    topic = (topic or "all").lower()
    out: dict[str, list] = {}

    if topic in ("all", "incidents", "fires"):
        rows = []
        for _, r in t["incidents"].iterrows():
            name, summary = INCIDENTS_EN.get(int(r["id"]), (str(r["vaisseau_lieu"]), ""))
            if not _match(f"{name} {summary} {r['date']}", query):
                continue
            rows.append({"name": name, "date": str(r["date"]), "summary": summary,
                         "caveat": INCIDENT_CAVEATS.get(int(r["id"]), ""),
                         "source": english_source(str(r["source"])), "url": str(r["url"]),
                         "verified_against_original": bool(r["verified"])})
        out["fires"] = rows

    if topic in ("all", "atmospheres", "spacecraft"):
        rows = []
        for _, r in t["atmospheres"].iterrows():
            name = ATMOSPHERES_EN.get(int(r["id"]), str(r["vaisseau_situation"]))
            if not _match(name, query):
                continue
            rows.append({
                "name": name,
                "status": ATMOSPHERE_STATUS.get(_status_key(r["type"]), "Other"),
                "pressure_psia": _clean(r["pression_psia"]) if not isinstance(r["pression_psia"], str) else r["pression_psia"],
                "pressure_atm": _clean(r["pressure_atm_min"]),
                "oxygen_percent": _clean(100 * r["o2_frac"]) if pd.notna(r["o2_frac"]) else None,
                "quote": str(r["citation"]),
                "source": english_source(str(r["source"])), "url": str(r["url"]),
                "verified_against_original": bool(r["verified"]),
                "shown_on_map": int(r["id"]) not in MASKED_ATMOSPHERES,
            })
        out["atmospheres"] = rows

    if topic in ("all", "gravity"):
        rows = []
        for _, r in t["gravity"].iterrows():
            name = GRAVITY_EN.get(int(r["id"]), str(r["regime"]))
            if not _match(name, query):
                continue
            rows.append({"name": name, "fraction_of_earth": _clean(r["fraction_of_earth"]),
                         "caveat": GRAVITY_CAVEATS.get(int(r["id"]), ""),
                         "quote": str(r["citation"]),
                         "source": english_source(str(r["source"])), "url": str(r["url"])})
        out["gravity"] = rows

    if topic in ("all", "literature", "papers", "results"):
        rows = []
        for _, r in t["literature"].iterrows():
            summary = LITERATURE_EN.get(int(r["id"]), "")
            if not _match(f"{summary} {r['article']} {r['combustible']}", query):
                continue
            rows.append({"result": summary, "paper": str(r["article"]),
                         "page": _clean(r["page"]),
                         "quote": str(r["citation"]),
                         "url": LITERATURE_LINKS.get(str(r["fichier"]), ""),
                         "is_simulation": bool(r["is_simulation"])})
        out["published_results"] = rows

    if not any(out.values()):
        return {"found": 0, "note": "Nothing in Fotia's sources matches. Say so; do not guess."}
    return out


# --- Registre -----------------------------------------------------------------
# Schemas JSON neutres ; gemini.py les traduit. `current` n'y figure pas :
# c'est le reglage de la console, ajoute par le code, jamais par le modele.

TOOLS = {
    "get_current_setting": {
        "fn": get_current_setting,
        "description": (
            "The setting the visitor is looking at right now in the console "
            "(place, fuel, oxygen, pressure, gravity, diluent, droplet size) and "
            "everything the console shows for it: confidence badge, burning rate "
            "constant K, extinction probability and outcome, burn time, values "
            "beyond NASA tests, engine notes, crew-limit check. Call this first "
            "for any question about 'this', 'my setting' or what the screen shows."),
        "parameters": {"type": "object", "properties": {}},
    },
    "predict": {
        "fn": predict,
        "description": (
            "Run Fotia's engine at another setting. Only give the values that "
            "change; the others stay as in the console. Use it for every number "
            "about a flame (K, extinction, burn time, confidence). Never estimate "
            "these yourself."),
        "parameters": {
            "type": "object",
            "properties": {
                "fuel": {"type": "string", "enum": FUELS},
                "pressure_atm": {"type": "number", "description": "Total pressure in atm (1 atm = 14.7 psia)."},
                "o2_percent": {"type": "number", "description": "Oxygen, percent of the gas."},
                "gravity_g": {"type": "number", "description": "0 = microgravity, 1 = Earth."},
                "diluent": {"type": "string", "enum": ["nitrogen", "co2", "helium"]},
                "diluent_percent": {"type": "number", "description": "Percent of CO2 or helium added; the rest is nitrogen."},
                "droplet_mm": {"type": "number", "description": "Initial droplet diameter in mm."},
            },
        },
    },
    "list_places": {
        "fn": list_places,
        "description": (
            "The place buttons in section 01 Where: each spacecraft atmosphere, "
            "its values, caveat and cited source, plus which places are hidden and why."),
        "parameters": {"type": "object", "properties": {}},
    },
    "tested_ranges": {
        "fn": tested_ranges,
        "description": (
            "What NASA actually tested, per fuel: pressure, oxygen, CO2, helium "
            "and droplet-size ranges, and how many tests each model learned from."),
        "parameters": {"type": "object", "properties": {}},
    },
    "search_sources": {
        "fn": search_sources,
        "description": (
            "Fotia's cited context: spacecraft fires (Apollo 1, Mir, Saffire...), "
            "spacecraft atmospheres, gravity levels, and published droplet "
            "combustion results with quotes and links."),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "enum": ["all", "fires", "atmospheres", "gravity", "papers"]},
                "query": {"type": "string", "description": "Optional keywords, e.g. 'Apollo' or 'helium'."},
            },
        },
    },
}


def run(name: str, args: dict, current: dict) -> dict:
    """Execute un outil ; une erreur revient au modele, pas a l'ecran."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"No tool named {name}."}
    allowed = set(tool["parameters"]["properties"])
    clean = {k: v for k, v in (args or {}).items() if k in allowed}
    try:
        return tool["fn"](current, **clean)
    except Exception as exc:  # le modele doit pouvoir dire « ca n'a pas marche »
        return {"error": f"{type(exc).__name__}: {exc}"}
