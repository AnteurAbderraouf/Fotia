"""PSI-25 / BASS-II — matériaux solides brûlés à bord de l'ISS.

129 combustions réelles en microgravité sur des matériaux qu'on embarque
vraiment : PMMA (films, plaques, barreaux), tissu SIBAL coton-fibre de verre,
Nomex, une bougie de cire. Six investigateurs principaux, deux ans de
campagne.

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-25/csv/BASS-II.csv

CONCLUSION QUI CONTREDIT LE PLAN DU HANDOFF, établie le 2026-09-22.
Le §8 prévoyait un module « Matériaux : ce matériau brûle-t-il dans
l'espace ? » entraîné sur PSI-25. Ce n'est pas possible honnêtement : ce jeu
de données NE PORTE PAS D'ÉTIQUETTE EXPLOITABLE. Deux pistes ont été
examinées, aucune ne tient.

  1. L'issue en texte libre. Le handoff annonce « 18 lignes le disent
     explicitement, le reste doit être inféré ». Le compte est bon — une
     vingtaine de cellules « Air display » portent « ext », « blowoff »,
     « no ign at any flow » ou « would not stay lit » — mais les 105 autres ne
     contiennent qu'une suite de débits, sans conclusion. Déduire une
     extinction d'une liste de nombres serait une invention, pas une
     inférence.

  2. La consommation d'oxygène. Elle paraissait prometteuse puisqu'elle est
     mesurée. Elle ne tient pas physiquement : l'O2 final monte jusqu'à 40.4 %
     et l'écart initial-final va de -19.7 à +20.6 points, c'est-à-dire que de
     l'oxygène apparaît. La chambre a manifestement été repurgée entre les
     deux relevés sur une partie des essais. La médiane est de 0.3 point,
     noyée dans ce bruit. Le CO porte par ailleurs des valeurs négatives
     (-3 ppm, décalage de capteur) et censurées (« >500 », « over »).

CE QUE CE JEU DE DONNÉES VAUT MALGRÉ TOUT. Un catalogue de matériaux, et il
est précieux : 129 combustions réelles en microgravité, avec la géométrie de
chaque échantillon une fois démêlée. C'est ce que produit ce loader — une
description propre, un `outcome` honnêtement vide là où rien n'a été écrit, et
des drapeaux de qualité sur les mesures de gaz.

LE DÉMÊLAGE DU MATÉRIAU. La colonne « Fuel Sample Material » compte 51
orthographes pour une petite famille d'échantillons. Le même film est écrit
« 2 cm 100 micron thick PMMA », « 100 micron PMMA film 2 cm wide » et
« 100 micron film 2 cm wide ». Les barreaux sont en pouces (1/4", 3/8", 1/2")
avec des variantes (« 1/4 black rod », « 3/8"clear rod », « 1/4 blac rod »).
Les épaisseurs sont tantôt en microns, tantôt en millimètres. On en extrait la
famille, la forme, l'épaisseur en mm, la largeur en cm, le diamètre en mm, la
couleur et le nombre de faces exposées — cette dernière compte, brûler une
face ou deux n'est pas la même chose.

LA RAMPE DE DÉBIT. « Air display » décrit la consigne d'air au fil de l'essai
(« 10 5 3 1 decr »). On en extrait la suite de valeurs, mais certaines
cellules mêlent des valeurs de potentiomètre ou de ventilateur entre
parenthèses (« 5 2 0 (49 62 49 44 42) ») ou renvoient ailleurs
(« see fan »). Ces cas sont signalés par `flow_ramp_ambiguous` au lieu d'être
interprétés de force.
"""

from __future__ import annotations

import re

import pandas as pd

from flame.common.clean import add_provenance, as_text
from flame.common.paths import processed_path, psi_dir

CSV = psi_dir("PSI-25") / "csv" / "BASS-II.csv"
ENCODING = "utf-8-sig"

RENAME = {
    "PI": "principal_investigator",
    "Test #": "test_id",
    "Date": "test_date",
    "GMT": "gmt_day",
    "Sample #": "sample_number",
    "Fuel Sample Material": "material_raw",
    "Flow restrictor": "flow_restrictor",
    "Fan display": "fan_display_raw",
    "Air display": "air_display_raw",
    "Radiometer gain (1000 250 50 10)": "radiometer_gain",
    "Total Frames Shot": "frames_shot_raw",
    "Calibrated initial O2 % by vol": "o2_initial_pct",
    "Calibrated final O2 % by vol": "o2_final_pct",
    "Initial CO2 % by vol": "co2_initial_pct",
    "Final CO2 % by vol": "co2_final_pct",
    "Initial CO (ppm)": "co_initial_ppm",
    "Final CO (ppm)": "co_final_ppm",
}

GAS_COLUMNS = [
    "o2_initial_pct",
    "o2_final_pct",
    "co2_initial_pct",
    "co2_final_pct",
    "co_initial_ppm",
    "co_final_ppm",
]

# Mots par lesquels NASA note une issue. Tout le reste ne dit rien.
OUTCOME_PATTERNS = [
    (r"no ign", "no_ignition"),
    (r"would not stay lit", "no_sustained_ignition"),
    (r"blowoff|blow off", "blowoff"),
    (r"\bext\b|0ext|extinguish", "extinction"),
]

# Une cellule qui mele des valeurs d'un autre instrument n'est pas une rampe
# de debit lisible telle quelle.
AMBIGUOUS_MARKERS = r"pot|see fan|\(|cm/s|\?"

INCH_TO_MM = 25.4


def _parse_material(text: str) -> dict:
    """Démêle une description d'échantillon en propriétés physiques."""
    if not isinstance(text, str):
        return {}
    lowered = text.lower()
    out: dict = {"material_family": None, "material_form": None}

    if "sibal" in lowered:
        out["material_family"] = "SIBAL cotton-fiberglass"
        out["material_form"] = "fabric"
    elif "nomex" in lowered:
        out["material_family"] = "Nomex"
        out["material_form"] = "fabric"
    elif "wax" in lowered or "candle" in lowered:
        out["material_family"] = "wax"
        out["material_form"] = "candle"
    elif "spherical" in lowered:
        out["material_family"] = "PMMA"
        out["material_form"] = "sphere"
    else:
        out["material_family"] = "PMMA"
        out["material_form"] = "rod" if "rod" in lowered else "sheet"

    if "clear" in lowered:
        out["sample_color"] = "clear"
    elif "black" in lowered or "blac " in lowered:  # « 1/4 blac rod »
        out["sample_color"] = "black"

    # Diametre des barreaux, ecrit en pouces sous forme de fraction.
    fraction = re.search(r"(\d)\s*/\s*(\d)", lowered)
    if fraction and out["material_form"] == "rod":
        out["diameter_mm"] = round(
            int(fraction.group(1)) / int(fraction.group(2)) * INCH_TO_MM, 3
        )
    elif out["material_form"] == "rod":
        decimal = re.search(r"(\d*\.?\d+)\s*\"", lowered)
        if decimal:
            out["diameter_mm"] = round(float(decimal.group(1)) * INCH_TO_MM, 3)

    # Epaisseur : microns ou millimetres selon la cellule.
    microns = re.search(r"(\d+)\s*micron", lowered)
    if microns:
        out["thickness_mm"] = int(microns.group(1)) / 1000
    else:
        millimetres = re.search(r"(\d*\.?\d+)\s*mm", lowered)
        if millimetres:
            out["thickness_mm"] = float(millimetres.group(1))

    width = re.search(r"(\d*\.?\d+)\s*cm", lowered)
    if width:
        out["width_cm"] = float(width.group(1))

    if "2 sided" in lowered or "2sided" in lowered:
        out["exposed_faces"] = 2
    elif "1 sided" in lowered or "1sided" in lowered:
        out["exposed_faces"] = 1

    return out


def _parse_air_display(text) -> dict:
    """Extrait l'issue déclarée et la rampe de débit d'une cellule."""
    if not isinstance(text, str):
        return {"outcome": None, "flow_ramp_ambiguous": None}

    lowered = text.lower()
    outcome = None
    for pattern, name in OUTCOME_PATTERNS:
        if re.search(pattern, lowered):
            outcome = name
            break

    ambiguous = bool(re.search(AMBIGUOUS_MARKERS, lowered))
    # On ne lit les nombres qu'en dehors des parentheses, qui contiennent des
    # valeurs d'un autre instrument.
    outside = re.sub(r"\([^)]*\)", " ", lowered)
    values = [float(v) for v in re.findall(r"\d*\.?\d+", outside)]

    result = {
        "outcome": outcome,
        "flow_ramp_ambiguous": ambiguous,
        "flow_steps": len(values) if values else None,
        "flow_first": values[0] if values else None,
        "flow_min": min(values) if values else None,
        "flow_max": max(values) if values else None,
    }
    if len(values) > 1:
        result["flow_decreasing"] = values[-1] < values[0]
    return result


def load() -> pd.DataFrame:
    """Table descriptive des 129 combustions, matériau et rampe démêlés."""
    df = pd.read_csv(CSV, encoding=ENCODING)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise KeyError(f"colonnes attendues absentes : {sorted(missing)}")
    df = df.rename(columns=RENAME)[list(RENAME.values())]

    df["material_raw"] = as_text(df["material_raw"])
    parsed = pd.DataFrame(
        [_parse_material(v) for v in df["material_raw"]], index=df.index
    )
    unparsed = df.loc[parsed["material_family"].isna(), "material_raw"].dropna().unique()
    if len(unparsed):
        raise ValueError(f"materiaux non reconnus : {list(unparsed)}")
    df = pd.concat([df, parsed], axis=1)

    df["air_display_raw"] = as_text(df["air_display_raw"])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [_parse_air_display(v) for v in df["air_display_raw"]], index=df.index
            ),
        ],
        axis=1,
    )

    for column in GAS_COLUMNS:
        text = as_text(df[column])
        df[f"{column}_censored"] = text.str.contains(
            r">|over|OR", case=False, na=False
        )
        df[column] = pd.to_numeric(text, errors="coerce")

    # Drapeaux de qualite : ces cas sont physiquement impossibles pour une
    # combustion en chambre close, la chambre a ete repurgee entre les relevés.
    df["o2_gained"] = df["o2_final_pct"] > df["o2_initial_pct"]
    df["co_negative"] = (df["co_initial_ppm"] < 0) | (df["co_final_ppm"] < 0)
    df["o2_consumed_pct"] = (df["o2_initial_pct"] - df["o2_final_pct"]).where(
        ~df["o2_gained"]
    )

    df["fan_display"] = pd.to_numeric(df["fan_display_raw"], errors="coerce")
    df["frames_shot"] = pd.to_numeric(df["frames_shot_raw"], errors="coerce")

    return add_provenance(df, investigation="PSI-25", gravity="microgravity")


def main() -> None:
    df = load()
    path = processed_path("psi25_materials_catalogue.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"{path.name} : {len(df)} combustions ISS\n")

    print("matériaux démêlés (51 orthographes ramenées à) :")
    print(
        df.groupby(["material_family", "material_form"])
        .size()
        .rename("essais")
        .to_string()
    )
    print("\ngeometries relevees :")
    for column, unit in [
        ("thickness_mm", "mm"),
        ("width_cm", "cm"),
        ("diameter_mm", "mm"),
    ]:
        values = sorted(df[column].dropna().unique())
        print(f"  {column:14} {len(values):2d} valeurs ({unit}) : {values[:9]}")
    faces = df["exposed_faces"].value_counts(dropna=False).to_dict()
    print(f"  faces exposees : {faces}  (non precise = NaN)")

    print("\nissue declaree dans « Air display » :")
    stated = df["outcome"].value_counts(dropna=False)
    for value, count in stated.items():
        label = "RIEN D'ECRIT" if pd.isna(value) else value
        print(f"  {str(label):24} {count:3d}")
    print(
        f"  -> {df['outcome'].notna().sum()} essais sur {len(df)} portent une issue."
        "\n     Trop peu pour entrainer, et le reste ne se devine pas."
    )

    print("\nqualite des mesures de gaz :")
    print(f"  O2 final > O2 initial (impossible) : {int(df['o2_gained'].sum())} essais")
    print(f"  CO negatif (decalage capteur)      : {int(df['co_negative'].sum())} essais")
    censored = int(df[[c for c in df if c.endswith('_censored')]].any(axis=1).sum())
    print(f"  valeur censuree (« >500 », « over »): {censored} essais")
    print(f"  rampe de debit ambigue             : {int(df['flow_ramp_ambiguous'].sum())} essais")

    print("\npar investigateur principal :")
    print(df["principal_investigator"].value_counts().to_string())


if __name__ == "__main__":
    main()
