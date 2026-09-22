"""PSI-107 / SPICE — point de fumée, propension d'un carburant à faire de la suie.

Module « Suie ». Le point de fumée est la longueur qu'atteint une flamme de
diffusion juste avant de commencer à émettre de la suie. Plus il est court,
plus le carburant encrasse : c'est une mesure directe de production de fumée,
donc de ce qui obscurcit une cabine et déclenche — ou non — un détecteur.

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-107/csv/
    SmokePointReview-0g-3.xlsx, feuille « Smoke Point Datasheet »

CORRECTION AU HANDOFF, vérifiée le 2026-09-22.
Le handoff décrivait « sheet Smoke Point Datasheet (~115 rows) ». Le compte est
juste mais trompeur : la feuille n'est pas un tableau, c'est QUATRE tableaux
empilés verticalement, issus de TROIS PLATEFORMES DE MICROGRAVITÉ DIFFÉRENTES,
sans qu'aucun en-tête ne l'annonce. Les empiler sans distinction reviendrait à
mélanger des régimes de gravité comme le §9 interdit de le faire.

    lignes   4-58   55 essais  SPICE en vol, coflow 5.4-64.6 cm/s
    lignes  62-67    6 essais  KC-135, l'avion à trajectoires paraboliques
                               (microgravité par bouffées d'environ 20 s),
                               coflow nul
    lignes  69-100  31 essais  STS 83 et 94, deux missions de la navette
                               spatiale, coflow nul
    lignes 104-118  15 essais  SPICE, série mesurée à la caméra Nikon

La qualité de microgravité n'est pas la même sur les trois : l'ISS offre une
microgravité continue, le KC-135 une vingtaine de secondes par parabole, la
navette un entre-deux. La colonne `platform` conserve cette distinction.

La table de référence du module est donc la campagne SPICE — 70 essais en
réunissant les deux séries — et non 115.

AUTRES PIÈGES DE LA FEUILLE :
  * L'en-tête tient sur DEUX lignes : le nom en ligne 2, l'unité en ligne 3.
  * Deux lignes de moyennes (« Ethylene Avg », « Propane Avg. ») sont posées au
    milieu des données de la navette. Ce sont des agrégats, pas des essais.
  * Un second en-tête est réinséré en ligne 102, au début du bloc Nikon.
  * Le nom du carburant change d'orthographe d'un bloc à l'autre :
    « 75 % Propylene », « 75% Propylene », « Propylene », « 100% Propylene ».
    On sépare l'espèce et sa fraction pour que le même mélange compte pour un.
  * Le diamètre de buse aussi : 1.6002 et 1.6, 0.762 et 0.764, sont les mêmes
    buses arrondies différemment.

Les bornes de lignes sont codées en dur. C'est assumé : ce fichier est une
archive figée, il ne changera plus. `_validate_layout` vérifie malgré tout les
repères au chargement, pour échouer bruyamment plutôt que de découper de
travers si le fichier venait à bouger.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance, as_text
from flame.common.paths import processed_path, psi_dir

XLSX = psi_dir("PSI-107") / "csv" / "SmokePointReview-0g-3.xlsx"
SHEET = "Smoke Point Datasheet"

# (première ligne, dernière ligne incluse, nom de série, plateforme)
BLOCKS = [
    (4, 58, "spice_flight", "ISS"),
    (62, 67, "zero_coflow_reference", "KC-135"),
    (69, 100, "sts_83_94", "Space Shuttle"),
    (104, 118, "spice_nikon", "ISS"),
]
SPICE_SERIES = {"spice_flight", "spice_nikon"}

COLUMNS = {
    5: "test_id",
    6: "fuel_raw",
    7: "nozzle_mm",
    8: "pressure_pa",
    11: "xf",
    12: "fuel_display",
    13: "air_display",
    14: "fan_setting",
    17: "smoke_point_pixels",
    18: "smoke_point_mm",
    20: "flame_width_mm",
    22: "tad_k",
    25: "total_fuel_sccm",
    26: "hc_only_sccm",
    27: "mdot_total_mg_s",
    28: "mdot_hc_mg_s",
    29: "coflow_velocity_cm_s",
    30: "fuel_jet_velocity_cm_s",
}

# Le propylène est dilué dans l'éthylène. NASA écrit la même chose de quatre
# façons ; on ramène à une espèce plus une fraction.
FUELS = {
    "ethylene": ("ethylene", 1.00),
    "propane": ("propane", 1.00),
    "propylene": ("propylene", 1.00),
    "100% propylene": ("propylene", 1.00),
    "75 % propylene": ("propylene", 0.75),
    "75% propylene": ("propylene", 0.75),
    "50 % propylene": ("propylene", 0.50),
    "50% propylene": ("propylene", 0.50),
}

# Les buses sont les mêmes, arrondies différemment selon le bloc.
NOZZLE_CANONICAL = {1.6: 1.6002, 0.764: 0.762}

NUMERIC = [
    "nozzle_mm",
    "pressure_pa",
    "xf",
    "fuel_display",
    "air_display",
    "fan_setting",
    "smoke_point_pixels",
    "smoke_point_mm",
    "flame_width_mm",
    "tad_k",
    "total_fuel_sccm",
    "hc_only_sccm",
    "mdot_total_mg_s",
    "mdot_hc_mg_s",
    "coflow_velocity_cm_s",
    "fuel_jet_velocity_cm_s",
]

# Conditions réglées avant l'essai. Le point de fumée et la largeur de flamme
# sont des mesures faites sur l'image de la flamme : ce sont les sorties.
FEATURES = [
    "fuel",
    "fuel_fraction",
    "nozzle_mm",
    "coflow_velocity_cm_s",
    "fuel_jet_velocity_cm_s",
    "mdot_hc_mg_s",
]
POST_BURN = ["smoke_point_mm", "smoke_point_pixels", "flame_width_mm"]
OUTCOME = "smoke_point_mm"


def _validate_layout(grid: pd.DataFrame) -> None:
    """Échoue bruyamment si le fichier n'a plus la forme attendue."""
    checks = {
        (2, 18): "smk pt",
        (4, 5): "E16-1",
        (61, 0): "KC-135",
        (69, 0): "STS 83 & 94",
        (102, 7): "Nozzle ID (mm)",
    }
    for (row, column), expected in checks.items():
        found = str(grid.iat[row, column]).strip()
        if expected.lower() not in found.lower():
            raise ValueError(
                f"{XLSX.name} : repere de mise en page perdu en "
                f"({row}, {column}) — attendu « {expected} », trouve « {found} ». "
                "Les bornes de blocs sont a revalider."
            )


def load(spice_only: bool = True) -> pd.DataFrame:
    """Charge les blocs de la feuille et les réunit avec leur plateforme.

    spice_only : True ne garde que la campagne SPICE (ISS). False ajoute les
        essais KC-135 et navette, qui sont de la microgravité d'une autre
        qualité et ne doivent pas être mêlés sans intention.
    """
    grid = pd.read_excel(XLSX, sheet_name=SHEET, header=None)
    _validate_layout(grid)

    frames = []
    for first, last, series, platform in BLOCKS:
        block = grid.iloc[first : last + 1, list(COLUMNS)].copy()
        block.columns = list(COLUMNS.values())
        block["series"] = series
        block["platform"] = platform
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)

    df["fuel_raw"] = as_text(df["fuel_raw"])

    # Les lignes de moyennes posees au milieu des essais navette sont des
    # agregats, pas des observations.
    is_average = df["fuel_raw"].str.contains("avg", case=False, na=False)
    df = df[~is_average]

    key = df["fuel_raw"].str.lower().str.strip()
    unknown = sorted(set(key.dropna()) - set(FUELS))
    if unknown:
        raise ValueError(f"carburants non repertories : {unknown}")
    df["fuel"] = key.map(lambda v: FUELS[v][0] if pd.notna(v) else pd.NA)
    df["fuel_fraction"] = key.map(
        lambda v: FUELS[v][1] if pd.notna(v) else pd.NA
    ).astype("Float64")

    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["nozzle_mm"] = df["nozzle_mm"].replace(NOZZLE_CANONICAL)

    df = df[df[OUTCOME].notna() & df["fuel"].notna()].reset_index(drop=True)
    df = add_provenance(df, investigation="PSI-107", gravity="microgravity")

    if spice_only:
        df = df[df["series"].isin(SPICE_SERIES)].reset_index(drop=True)

    return df


def main() -> None:
    everything = load(spice_only=False)
    spice = load(spice_only=True)

    all_path = processed_path("psi107_all_platforms.csv")
    spice_path = processed_path("psi107_smoke_point.csv")
    everything.to_csv(all_path, index=False, encoding="utf-8")
    spice.to_csv(spice_path, index=False, encoding="utf-8")

    print(f"{all_path.name:30} {len(everything):3d} essais, toutes plateformes")
    print(f"{spice_path.name:30} {len(spice):3d} essais SPICE (ISS)")
    print("\npar serie et plateforme :")
    print(
        everything.groupby(["platform", "series"]).size().rename("essais").to_string()
    )
    print("\ncarburants SPICE :")
    print(spice.groupby(["fuel", "fuel_fraction"]).size().rename("essais").to_string())
    print("\nbuses (mm) :", sorted(spice["nozzle_mm"].dropna().unique()))
    print(
        f"point de fumee : {spice[OUTCOME].min():.1f} -> {spice[OUTCOME].max():.1f} mm "
        "(court = suie abondante)"
    )
    print("\npoint de fumee moyen par carburant, en mm :")
    print(spice.groupby("fuel")[OUTCOME].agg(["count", "mean"]).round(1).to_string())


if __name__ == "__main__":
    main()
