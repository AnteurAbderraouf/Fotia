"""PSI-159 / ACME CFI-G — flammes de gaz sur brûleur sphérique (ISS).

Module « Sustainment » : une flamme de gaz se maintient-elle d'elle-même, ou
s'éteint-elle seule ? L'étiquette est directement observée et n'invente aucun
seuil : SE (self-extinguished, la flamme est morte seule) contre FT (flow
terminated, l'équipage a coupé le débit alors qu'elle brûlait encore).

Sources (lecture seule) :
    combustion_science/microgravity_investigation/PSI-159/csv/
        CFI-G_Normal Flames.csv     179 essais
        CFI-G_Inverse Flames.csv    154 essais utiles (le fichier fait
                                    1015 lignes, le reste est du vide Excel)

DEUX CONFIGURATIONS, PAS UN SEUL JEU DE DONNÉES.
Le handoff annonçait « 179 + 154 = 333 » comme si les deux fichiers avaient le
même schéma. Ils ne l'ont pas, et la différence est physique, pas cosmétique :

    flamme normale  : du carburant est injecté dans une ambiance oxydante.
                      Ce qui varie est la fraction d'O2 ambiante (XO2).
    flamme inverse  : de l'oxygène est injecté dans une ambiance carburée.
                      Ce qui varie est la fraction de carburant ambiante (Xf).

Ce sont des expériences en miroir. Leurs colonnes de composition ne se
correspondent donc pas une à une, et les empiler telles quelles produirait une
colonne qui veut dire deux choses différentes selon la ligne.

Ce qui se met légitimement en commun, ce sont les grandeurs thermochimiques
calculées, qui décrivent la flamme indépendamment de la configuration :
Tad (température de flamme adiabatique), Zst (fraction de mélange
stœchiométrique), la pression, l'espèce du carburant. La table produite porte
une colonne `flame_type` et garde les colonnes propres à chaque configuration,
vides pour l'autre. On peut donc modéliser les deux ensemble sur les features
communes, ou chaque configuration séparément — mais jamais par accident.

CORRECTION au handoff sur le nombre de lignes. L'étiquette n'existe pas
partout : 61 essais portent « - » au lieu de SE ou FT (46 en normal,
15 en inverse). Le total modélisable est de 272 essais, pas 333.

Autres pièges relevés dans les fichiers le 2026-09-22 :
  * Des erreurs de formule Excel sont figées comme valeurs : « #DIV/0! » dans
    Tad et Zst du fichier normal, « #VALUE! » dans la température du fichier
    inverse. Traitées comme des mesures absentes.
  * « Hot ignition » du fichier inverse mélange les casses (Yes/yes, No/no) et
    contient 8 « flash » — un allumage qui n'a pas pris. Ce n'est ni oui ni
    non, donc la colonne reste à trois modalités plutôt que d'écraser
    l'information dans un booléen.
  * « Flame Type » du fichier inverse alterne « Inverse » et « inverse ».
  * Le carburant du fichier normal encode la dilution dans son nom
    (« 30% butane » = butane dilué à 30 % dans N2, « butane » = pur). On
    sépare en espèce + taux de dilution, faute de quoi le même gaz compte
    comme trois carburants distincts et n'est comparable ni entre lignes ni
    avec le fichier inverse.
  * Le sens de la colonne « TFP » n'est documenté nulle part : l'info.md de
    PSI-159 est un squelette vide (seul de tout le catalogue) et les deux
    rapports PDF ne sont pas dépouillés. Colonne conservée telle quelle,
    signalée comme non documentée.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance, as_text, to_numeric
from flame.common.paths import processed_path, psi_dir

CSV_DIR = psi_dir("PSI-159") / "csv"
NORMAL_CSV = CSV_DIR / "CFI-G_Normal Flames.csv"
INVERSE_CSV = CSV_DIR / "CFI-G_Inverse Flames.csv"

# Le fichier normal encode la dilution dans le nom du carburant. On la sort
# pour que « butane » veuille dire la même chose partout.
FUEL_DILUTION = {
    "30% butane": ("butane", 0.30),
    "50% butane": ("butane", 0.50),
    "30% ethane": ("ethane", 0.30),
    "30% propane": ("propane", 0.30),
    "ethane": ("ethane", 1.00),
    "propane": ("propane", 1.00),
    "butane": ("butane", 1.00),
}

NORMAL_RENAME = {
    "Flame Type": "flame_type",
    "Day": "test_day",
    "Fuel": "fuel_raw",
    "Test": "test_id",
    "XO2 before test": "xo2_before",
    "XO2  after test": "xo2_after",
    "approx p (bar)": "pressure_bar",
    "Xf": "xf_jet",
    "mHC (mg/s)": "fuel_flow_mg_s",
    "TFP": "tfp",
    "Tad (K)": "tad_k",
    "Zst": "zst",
    "Hot ignition": "hot_ignition",
    "burn time (sec)": "burn_time_s",
    "Self-ext or flow term": "outcome_raw",
    "Peak burner T (C )": "peak_burner_temp_c",
    "Comments": "comments",
    "Cool flame?": "cool_flame_note",
}

INVERSE_RENAME = {
    "Flame Type": "flame_type",
    "Day": "test_day",
    "Fuel": "fuel_raw",
    "Test": "test_id",
    "Xf  before test": "xf_ambient_before",
    "Xf  after test": "xf_ambient_after",
    "approx p (bar)": "pressure_bar",
    "XO2": "xo2_jet",
    "HC burnrate (mg/s)": "fuel_flow_mg_s",
    "O2 flow rate (mg/s)": "o2_flow_mg_s",
    "TFP": "tfp",
    "Tad (K)": "tad_k",
    "Zst": "zst",
    "Hot ignition": "hot_ignition",
    "burn time (sec)": "burn_time_s",
    "Self -ext or flow term": "outcome_raw",
    "Peak burner Temp (C)": "peak_burner_temp_c",
    "Color": "flame_color",
    "Comments": "comments",
}

# --- Séparation entrées / sorties --------------------------------------------
#
# FEATURES_COMMON vaut pour les deux configurations. C'est le seul jeu qui
# autorise à modéliser normal et inverse ensemble.
#
# Les fractions « before test » sont des conditions initiales, donc des
# entrées ; les « after test » sont des mesures de fin, donc des sorties.
#
# `burn_time_s` est écarté des features et c'est le point délicat : la durée de
# combustion est fortement liée à l'issue, puisqu'une flamme qui s'éteint seule
# brûle par définition jusqu'à son extinction. La donner en entrée reviendrait
# à annoncer le résultat.

FEATURES_COMMON = [
    "fuel",
    "fuel_dilution",
    "pressure_bar",
    "tad_k",
    "zst",
    "fuel_flow_mg_s",
    "tfp",
    "flame_type",
]
FEATURES_NORMAL_ONLY = ["xo2_before", "xf_jet"]
FEATURES_INVERSE_ONLY = ["xf_ambient_before", "xo2_jet", "o2_flow_mg_s"]
FEATURES = FEATURES_COMMON

POST_BURN = [
    "xo2_after",
    "xf_ambient_after",
    "burn_time_s",
    "peak_burner_temp_c",
    "flame_color",
    "hot_ignition",
]
LABEL = "self_extinguished"

NUMERIC_COLS = [
    "pressure_bar",
    "tad_k",
    "zst",
    "fuel_flow_mg_s",
    "burn_time_s",
    "peak_burner_temp_c",
    "xo2_before",
    "xo2_after",
    "xf_jet",
    "xf_ambient_before",
    "xf_ambient_after",
    "xo2_jet",
    "o2_flow_mg_s",
]


def _read(path, rename: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    keep = {src: dst for src, dst in rename.items() if src in df.columns}
    missing = set(rename) - set(keep)
    if missing:
        raise KeyError(f"{path.name} : colonnes attendues absentes {missing}")
    # Le fichier inverse traîne des centaines de lignes Excel vides.
    return df[list(keep)].rename(columns=keep).dropna(subset=["test_id"])


def load(labelled_only: bool = True) -> pd.DataFrame:
    """Charge et harmonise les deux configurations en une table.

    labelled_only : True écarte les 61 essais dont l'issue est « - », qui ne
        peuvent pas porter d'étiquette.
    """
    df = pd.concat(
        [_read(NORMAL_CSV, NORMAL_RENAME), _read(INVERSE_CSV, INVERSE_RENAME)],
        ignore_index=True,
    )

    df["flame_type"] = as_text(df["flame_type"]).str.lower()

    fuel = as_text(df["fuel_raw"])
    unknown = sorted(set(fuel.dropna()) - set(FUEL_DILUTION))
    if unknown:
        raise ValueError(f"carburants NASA non repertories : {unknown}")
    df["fuel"] = fuel.map(lambda v: FUEL_DILUTION[v][0] if pd.notna(v) else pd.NA)
    df["fuel_dilution"] = fuel.map(
        lambda v: FUEL_DILUTION[v][1] if pd.notna(v) else pd.NA
    ).astype("Float64")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col], _ = to_numeric(df[col])

    df["tfp"] = as_text(df["tfp"]).str.lower()
    # « flash » est une troisieme modalite reelle, pas un oui/non mal ecrit :
    # l'allumage a eu lieu mais n'a pas pris.
    df["hot_ignition"] = as_text(df["hot_ignition"]).str.lower()

    # SE = la flamme est morte seule, FT = l'equipage a coupe le debit alors
    # qu'elle brulait encore. « - » n'est ni l'un ni l'autre.
    outcome = as_text(df["outcome_raw"]).str.upper()
    df[LABEL] = outcome.map({"SE": 1, "FT": 0}).astype("Int64")

    df = add_provenance(df, investigation="PSI-159", gravity="microgravity")

    if labelled_only:
        df = df[df[LABEL].notna()].reset_index(drop=True)

    return df


def main() -> None:
    everything = load(labelled_only=False)
    modeling = load(labelled_only=True)

    all_path = processed_path("psi159_all_tests.csv")
    model_path = processed_path("psi159_sustainment.csv")
    everything.to_csv(all_path, index=False, encoding="utf-8")
    modeling.to_csv(model_path, index=False, encoding="utf-8")

    print(f"{all_path.name:26} {len(everything):3d} essais (campagne complete)")
    print(f"{model_path.name:26} {len(modeling):3d} essais etiquetes (modelisables)")
    print(f"{'':26} {len(everything) - len(modeling):3d} ecartes : issue notee « - »")
    print()
    print("par configuration :")
    print(
        pd.crosstab(modeling["flame_type"], modeling[LABEL]).rename(
            columns={0: "flow term (FT)", 1: "auto-extinction (SE)"}
        ).to_string()
    )
    counts = modeling[LABEL].value_counts().sort_index()
    total = int(counts.sum())
    print(f"\nplancher majoritaire : {counts.max() / total:.0%}")
    print(f"carburants : {sorted(modeling['fuel'].dropna().unique())}")
    print(f"dilutions  : {sorted(modeling['fuel_dilution'].dropna().unique())}")


if __name__ == "__main__":
    main()
