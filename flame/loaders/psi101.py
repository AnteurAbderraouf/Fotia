"""PSI-101 / SAME-R — fumée et réponse des détecteurs de l'ISS.

Module « Détection ». Des matériaux réellement embarqués sont chauffés jusqu'à
émettre de la fumée, et l'on mesure à la fois ce que la fumée est
physiquement — masse, nombre et taille des particules — et ce que les
détecteurs de bord en perçoivent.

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-101/csv/
    Combined SAMER Results 06 02 14.xlsx, feuille « detection June 14 »

CHOIX DE LA FEUILLE. Le classeur compte 43 feuilles. « detection June 14 » et
« detection Mar 14 » sont quasi identiques ; la première est un sur-ensemble
strict (deux colonnes de seuil d'alarme en plus), c'est donc elle qu'on lit.
Les autres feuilles sont des vues par matériau, des versions intermédiaires et
des tables destinées à des articles ; elles n'apportent pas de ligne nouvelle.

DEUX CAMPAGNES DANS UNE SEULE FEUILLE. La première colonne,
« Flight 1=same 2=sameR, g=ground », sépare 45 essais SAME et 109 essais
SAME-R. Autrement dit les résultats de vol de PSI-102 (SAME d'origine) sont
ici, ce qui confirme qu'il n'y a rien à retélécharger pour PSI-102, et le
compte de 109 annoncé pour PSI-101 correspond exactement au groupe SAME-R.

CORRECTION AU HANDOFF, vérifiée le 2026-09-22. Le handoff annonce
« Label : did the actual ISS smoke detector respond », c'est-à-dire une
classification. Aucune colonne du classeur n'enregistre un déclenchement.
Ce qui existe est de deux natures :

    la réponse mesurée des détecteurs, en volts, continue
        ISS Detector Scatter, ISS Obscuration, STS Detector, Ion A et Ion B

    des seuils calculés, du type « quelle concentration faudrait-il pour
    déclencher telle alarme », qui sont des grandeurs dérivées

Ce module est donc une régression — prédire la réponse d'un détecteur à partir
du matériau et des conditions — et non une classification. Fabriquer un seuil
pour en faire un oui/non reviendrait à inventer l'étiquette, ce que le §9
interdit explicitement.

LE DBP N'EST PAS UN MATÉRIAU DE VAISSEAU. Les 20 essais « DBP » utilisent du
phtalate de dibutyle, un aérosol d'étalonnage servant à vérifier les
instruments. Les confondre avec les cinq matériaux réels (Kapton, Silicone,
Lampwick, Pyrell, Teflon) fausserait toute comparaison. La colonne
`is_calibration_aerosol` les distingue.

CE QUE DISENT DÉJÀ LES DONNÉES. Teflon et Kapton produisent les nombres de
particules les plus élevés mais les signaux de diffusion les plus faibles
(1.4 et 2.4 volts, contre 6.25 pour le silicone). Une fumée composée de fines
particules échappe largement à un détecteur à diffusion. C'est un résultat de
sécurité en soi, et il est déjà lisible avant tout modèle.

SÉLECTION PAR POSITION. La feuille comporte des en-têtes en double
(« Diluting A », « GMT », « Aerosol Concentration To P-Trak » apparaissent
deux fois). Les colonnes sont donc désignées par leur indice, pas par leur
nom, sous peine de récupérer la mauvaise.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance, as_text
from flame.common.paths import processed_path, psi_dir

XLSX = psi_dir("PSI-101") / "csv" / "Combined SAMER Results 06 02 14.xlsx"
SHEET = "detection June 14"
HEADER_ROW = 1

CAMPAIGNS = {"1": "SAME", "2": "SAME-R", "g": "ground"}
CALIBRATION_AEROSOL = "DBP"

# Indice de colonne -> nom retenu. Les indices sont ceux de la feuille brute.
CONDITIONS = {
    0: "campaign_code",
    2: "material",
    4: "test_id",
    6: "same_number",
    7: "description",
    13: "inlet_velocity_cm_s",
    15: "duration_s",
    16: "primary_aging_s",
    17: "primary_mixing_s",
    20: "net_aging_s",
    35: "secondary_aging_s",
    36: "secondary_mixing_s",
    82: "sample_temperature_c",
}

DETECTOR_RESPONSE = {
    47: "iss_obscuration_volts",
    48: "iss_scatter_volts",
    49: "sts_detector_volts",
    55: "ion_b_volts",
    62: "ion_b_delta_volts",
}

SMOKE_PHYSICS = {
    57: "ptrak_particles_cc",
    61: "dusttrak_b_mg_m3",
    63: "m0_best_diameter",
    64: "m1_mm_cm3",
    69: "amd_micro_m",
    70: "mass_average_diameter_micro_m",
    72: "hatch_choate_sigma_g",
    73: "hatch_choate_dg",
    74: "cpc_100nm",
    75: "cpc_600nm",
    76: "cpc_1200nm",
    83: "mass_loss_mg",
}

ALARM_THRESHOLDS = {
    94: "count_for_mass_alarm",
    95: "mass_for_count_alarm",
    97: "count_for_scattering_alarm",
    98: "scattering_for_count_alarm",
    100: "scattering_for_ion_alarm",
    101: "count_for_ion_alarm",
    102: "ion_for_scattering_alarm",
    103: "ion_for_count_alarm",
}

COLUMNS = {**CONDITIONS, **DETECTOR_RESPONSE, **SMOKE_PHYSICS, **ALARM_THRESHOLDS}

TEXT_COLUMNS = {"campaign_code", "material", "test_id", "description", "same_number"}

# Conditions reglees avant l'essai. Tout le reste est mesure pendant ou apres,
# ou calcule a partir des mesures.
FEATURES = [
    "material",
    "inlet_velocity_cm_s",
    "duration_s",
    "primary_aging_s",
    "primary_mixing_s",
    "net_aging_s",
]
OUTCOMES = list(DETECTOR_RESPONSE.values())
DERIVED = list(ALARM_THRESHOLDS.values())


def _verify_headers(grid: pd.DataFrame) -> None:
    """Contrôle que les indices désignent toujours les bonnes colonnes."""
    expected = {
        0: "Flight",
        2: "Sample Type",
        48: "ISS Detector Scatter",
        57: "Ptrak",
        94: "Number count to produce mass alarm",
    }
    header = grid.iloc[HEADER_ROW]
    for index, fragment in expected.items():
        found = str(header.iloc[index])
        if fragment.lower() not in found.lower():
            raise ValueError(
                f"{XLSX.name} / {SHEET} : la colonne {index} devait contenir "
                f"« {fragment} », elle contient « {found} ». "
                "La selection par indice est a revalider."
            )


def load(exclude_calibration: bool = True) -> pd.DataFrame:
    """Charge la feuille de détection et nomme ses colonnes.

    exclude_calibration : True écarte les 20 essais DBP, qui sont de
        l'étalonnage d'instrument et non des matériaux de vaisseau.
    """
    grid = pd.read_excel(XLSX, sheet_name=SHEET, header=None)
    _verify_headers(grid)

    body = grid.iloc[HEADER_ROW + 1 :, list(COLUMNS)].copy()
    body.columns = list(COLUMNS.values())

    body["material"] = as_text(body["material"])
    body = body[body["material"].notna()].reset_index(drop=True)

    for column in body.columns:
        if column in TEXT_COLUMNS:
            body[column] = as_text(body[column])
        else:
            body[column] = pd.to_numeric(body[column], errors="coerce")

    body["campaign"] = body["campaign_code"].map(CAMPAIGNS)
    unknown = body.loc[body["campaign"].isna(), "campaign_code"].dropna().unique()
    if len(unknown):
        raise ValueError(f"codes de campagne inconnus : {list(unknown)}")

    # SAME-R est PSI-101, SAME d'origine est PSI-102 : la feuille porte les deux.
    body["investigation_source"] = body["campaign"].map(
        {"SAME-R": "PSI-101", "SAME": "PSI-102", "ground": "PSI-101"}
    )
    body["is_calibration_aerosol"] = body["material"].eq(CALIBRATION_AEROSOL)

    body = add_provenance(body, investigation="PSI-101", gravity="microgravity")

    if exclude_calibration:
        body = body[~body["is_calibration_aerosol"]].reset_index(drop=True)

    return body


def main() -> None:
    everything = load(exclude_calibration=False)
    materials = load(exclude_calibration=True)

    all_path = processed_path("psi101_all_tests.csv")
    mat_path = processed_path("psi101_detection.csv")
    everything.to_csv(all_path, index=False, encoding="utf-8")
    materials.to_csv(mat_path, index=False, encoding="utf-8")

    print(f"{all_path.name:26} {len(everything):3d} essais, etalonnage DBP inclus")
    print(f"{mat_path.name:26} {len(materials):3d} essais sur materiaux reels")
    print("\npar campagne (et investigation d'origine) :")
    print(
        everything.groupby(["campaign", "investigation_source"])
        .size()
        .rename("essais")
        .to_string()
    )
    print("\npar materiau :")
    print(materials["material"].value_counts().to_string())
    print("\nreponse des detecteurs et physique de la fumee, moyennes par materiau :")
    view = materials.groupby("material")[
        ["iss_scatter_volts", "iss_obscuration_volts", "ptrak_particles_cc", "dusttrak_b_mg_m3"]
    ].mean().round(2)
    view.columns = ["diffusion V", "obscurcis. V", "particules/cc", "masse mg/m3"]
    print(view.to_string())
    print(
        "\n-> Teflon et Kapton : beaucoup de particules, peu de signal de diffusion."
        "\n   Une fumee fine echappe a ce type de detecteur."
    )


if __name__ == "__main__":
    main()
