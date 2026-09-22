"""PSI-39 / CFI — flammes froides sur gouttelettes (ISS).

Module « Flammes froides » : après l'extinction de la flamme visible, une
combustion invisible persiste-t-elle ? C'est un enjeu de sécurité direct — un
feu déclaré éteint qui continue à consommer du carburant sans être vu.

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-39/csv/
    CFI_DataSummary_Spreadsheet_Jan2021.xlsx

Le classeur a six feuilles, trois sont utiles et se joignent en 1:1 sur
« Flex Number » (227 identifiants, aucun doublon, mêmes clés partout) :
    Master Sheet  conditions ambiantes (pression, gaz, carburant, fibre)
    DATA          mesures de combustion + état d'avancement de l'analyse
    Igniter       énergie et durée d'allumage

CORRECTION IMPORTANTE au handoff, vérifiée sur les données le 2026-09-22.
Le handoff annonçait « 120 des 227 ont un diamètre d'extinction de flamme
froide, 107 non, étiquette à peu près équilibrée ». C'est faux, et l'erreur
aurait produit un modèle qui apprend la mauvaise chose.

  * 81 des 227 essais portent « Completed Analysis? = No » et sont vides :
    seuls 3 d'entre eux ont la moindre mesure. Compter ces lignes comme
    « pas de flamme froide » revient à entraîner le modèle à prédire si un
    essai a été dépouillé, et non si une flamme froide a eu lieu.

  * Sur les 146 essais réellement analysés, le diamètre d'extinction seul est
    trompeur : 7 essais ont un taux de combustion froide (kcool) sans diamètre
    relevé. Une flamme froide qui a un taux de combustion a bel et bien
    existé, son extinction n'a simplement pas été mesurée.

  * Étiquette honnête : flamme froide observée si kcool OU Dext,cool est
    présent, sur les seuls essais analysés. Soit 124 oui / 22 non sur 146
    lignes, c'est-à-dire 85/15 — nettement déséquilibrée, pas équilibrée.

Autre écart au handoff, celui-ci en notre faveur : il signalait la colonne
Pressure comme du texte incohérent (« .5 atm » contre « 0.5 atm »). C'est vrai
dans Master Sheet, mais DATA porte une colonne « Pressure (atm) » déjà
numérique, et les deux concordent sur les 227 lignes sans une seule exception.
On utilise la numérique et le problème disparaît.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import (
    add_provenance,
    as_text,
    check_fractions_sum_to_one,
    strip_unit,
)
from flame.common.paths import processed_path, psi_dir

XLSX = psi_dir("PSI-39") / "csv" / "CFI_DataSummary_Spreadsheet_Jan2021.xlsx"
JOIN_KEY = "flex_number"

# NASA nomme les carburants de quatre façons, avec des espaces de fin
# irréguliers. On normalise pour l'usage et on garde la chaîne NASA exacte
# dans `fuel_raw` — le projet conserve ses traces.
FUEL_LABELS = {
    "C12H26": "n-dodecane",
    "Farnesane": "farnesane",
    "[.75]/[.25] C12H26/iso-dodecane": "dodecane75-isododecane25",
    "[.60]/[.40] C12H26/iso-dodecane": "dodecane60-isododecane40",
}

MASTER_RENAME = {
    "Flex Number": JOIN_KEY,
    "Test Date": "test_date",
    "Local Time": "local_time",
    "CIR": "cir_id",
    "Pressure": "pressure_raw",
    "Oxygen": "o2_frac",
    "Nitrogen": "n2_frac",
    "Helium": "he_frac",
    "Fiber": "fiber",
    "Fuel": "fuel_raw",
}

DATA_RENAME = {
    "Flex Number": JOIN_KEY,
    "D0 ( mm )": "d0_mm",
    "Dext, hot ( mm )": "dext_hot_mm",
    "khot ( mm^2/s )": "k_hot_mm2_s",
    "kcool ( mm^2/s )": "k_cool_mm2_s",
    "Dext,cool ( mm )": "dext_cool_mm",
    "Pressure (atm)": "pressure_atm",
    "Reignitions": "reignitions",
    "Extrap CF?": "cool_flame_extrapolated",
    "HF Burn Time": "hot_flame_burn_time_s",
    "Peak HF Rad": "peak_hot_flame_radiance",
    "Peak Water Vapor Rad": "peak_water_vapor_radiance",
    "Rad Ratio": "radiance_ratio",
    "Completed Analysis?": "analysis_completed",
}

IGNITER_RENAME = {
    "Flex Number": JOIN_KEY,
    "Ignition Power ( W )": "ignition_power_w",
    "Ignition Time ( ms )": "ignition_time_ms",
}

# --- Séparation entrées / sorties --------------------------------------------
#
# Même règle que pour PSI-69 : ne sont features que les grandeurs connues avant
# l'allumage. Le diamètre initial en fait partie, c'est une consigne
# expérimentale — la gouttelette est déployée à une taille voulue puis allumée.
#
# Tout ce qui décrit la combustion elle-même est écarté. `reignitions` mérite
# une mention : les réallumages sont fortement liés à la flamme froide (tous
# les essais à deux réallumages ou plus en ont une). C'est précisément pour ça
# qu'il ne peut pas servir d'entrée — un réallumage EST un comportement de
# flamme froide, pas une condition qui la précède.

FEATURES = [
    "fuel",
    "pressure_atm",
    "o2_frac",
    "he_frac",
    "fiber",
    "d0_mm",
    "ignition_power_w",
    "ignition_time_ms",
]
POST_BURN = [
    "dext_hot_mm",
    "k_hot_mm2_s",
    "k_cool_mm2_s",
    "dext_cool_mm",
    "reignitions",
    "hot_flame_burn_time_s",
    "peak_hot_flame_radiance",
    "peak_water_vapor_radiance",
    "radiance_ratio",
]
LABEL = "cool_flame"


def _sheet(name: str, rename: dict) -> pd.DataFrame:
    """Lit une feuille, nettoie ses en-têtes, ne garde que les colonnes utiles."""
    df = pd.read_excel(XLSX, sheet_name=name)
    df.columns = [str(c).strip() for c in df.columns]
    return df[list(rename)].rename(columns=rename)


def load(analysed_only: bool = True) -> pd.DataFrame:
    """Charge et joint les feuilles utiles.

    analysed_only : True ne garde que les 146 essais dépouillés, seuls capables
        de porter une étiquette. False renvoie les 227 avec `cool_flame` à NaN
        sur les non-dépouillés — utile pour inventorier la campagne complète,
        jamais pour entraîner.
    """
    df = (
        _sheet("Master Sheet", MASTER_RENAME)
        .merge(_sheet("DATA", DATA_RENAME), on=JOIN_KEY, validate="one_to_one")
        .merge(_sheet("Igniter", IGNITER_RENAME), on=JOIN_KEY, validate="one_to_one")
    )

    df["fuel_raw"] = as_text(df["fuel_raw"])
    df["fuel"] = df["fuel_raw"].map(FUEL_LABELS)
    unmapped = df.loc[df["fuel"].isna() & df["fuel_raw"].notna(), "fuel_raw"].unique()
    if len(unmapped):
        raise ValueError(f"carburants NASA non repertories : {list(unmapped)}")

    df["fiber"] = as_text(df["fiber"])
    df["ignition_power_w"] = strip_unit(df["ignition_power_w"])
    df["ignition_time_ms"] = strip_unit(df["ignition_time_ms"])
    df["analysis_completed"] = as_text(df["analysis_completed"]).eq("Yes")

    check_fractions_sum_to_one(df, ["o2_frac", "n2_frac", "he_frac"])

    # Une flamme froide a existé si l'une OU l'autre de ses deux signatures a
    # été mesurée. Voir l'en-tête du module : le diamètre seul en manque 7.
    observed = df["k_cool_mm2_s"].notna() | df["dext_cool_mm"].notna()
    df[LABEL] = observed.astype("Int64").where(df["analysis_completed"])

    df["cool_flame_extrapolated"] = df["cool_flame_extrapolated"].astype("Int64")
    df = add_provenance(df, investigation="PSI-39", gravity="microgravity")

    if analysed_only:
        df = df[df["analysis_completed"]].reset_index(drop=True)

    return df


def main() -> None:
    everything = load(analysed_only=False)
    modeling = load(analysed_only=True)

    all_path = processed_path("psi39_all_tests.csv")
    model_path = processed_path("psi39_cool_flames.csv")
    everything.to_csv(all_path, index=False, encoding="utf-8")
    modeling.to_csv(model_path, index=False, encoding="utf-8")

    print(f"{all_path.name:26} {len(everything):3d} essais (campagne complete)")
    print(f"{model_path.name:26} {len(modeling):3d} essais depouilles (modelisables)")
    print(f"{'':26} {len(everything) - len(modeling):3d} ecartes : analyse non faite")
    print()
    counts = modeling[LABEL].value_counts().sort_index()
    total = int(counts.sum())
    print("Etiquette :")
    for value, count in counts.items():
        name = "flamme froide" if value == 1 else "aucune"
        print(f"  {name:16} {count:3d}  ({count / total:.0%})")
    print(f"  -> plancher majoritaire : {counts.max() / total:.0%}")
    extrap = int(modeling["cool_flame_extrapolated"].fillna(0).sum())
    print(f"  dont {extrap} diametres extrapoles, non mesures directement")


if __name__ == "__main__":
    main()
