"""PSI-142 / Princeton — limites d'extinction en contre-courant (AU SOL, 1g).

Référence terrestre du projet. Un brûleur à contre-courant mesure à quelle
vitesse d'étirement (strain rate) une flamme de diffusion s'éteint, pour cinq
alcanes et avec ou sans ozone ajouté.

ATTENTION GRAVITÉ. Ces essais sont faits SUR TERRE, à 1g, sur un brûleur qui
n'a rien d'un essai ISS. La colonne `gravity` vaut « 1g » et toute mise en
commun avec les données ISS sans en tenir compte mélangerait deux régimes
physiques distincts — c'est la mise en garde du §9 du handoff, et c'est ici
qu'elle s'applique.

Source (lecture seule) :
    combustion_science/ground_investigation/PSI-142/csv/
    PSI data summary Princeton with links.xlsx

CORRECTION AU HANDOFF, vérifiée le 2026-09-22 sur la feuille « data summary »,
qui est l'index rédigé par Princeton lui-même. Le handoff annonçait
« data ID 1 : avec ozone, data ID 2 : sans ozone ». C'est l'inverse :

    data ID 1   limites d'extinction de flamme froide, SANS ozone   (Fig. 4)
    data ID 2   limites d'extinction de flamme froide, AVEC ozone   (Fig. 5)
    data ID 3   limites d'extinction de flamme chaude, sans ozone   (Fig. 7)

L'ozone est l'intérêt scientifique du jeu de données — il sensibilise le
mélange et permet à une flamme froide d'exister là où elle ne tiendrait pas.
Intervertir les deux conditions inverserait la conclusion.

STRUCTURE. Chaque feuille empile ses blocs VERTICALEMENT : une cellule de
gauche porte le nom du carburant (nC14, nC12, nC10, nC8, nC7) et les lignes qui
suivent sont ses mesures, jusqu'au carburant suivant. C'est l'inverse de
PSI-117, dont les blocs sont côte à côte.

FEUILLES ÉCARTÉES. `data ID 5`, `6` et `7` sont des profils spatiaux — un
signal en fonction de la distance dans la flamme, environ 2 300 points. Ce sont
des courbes au sein d'une condition, pas des observations par condition, et
elles n'entrent pas dans une table par essai.

NATURE DE LA SORTIE. Contrairement aux autres modules, l'observation n'est pas
une classe mais une grandeur continue : la vitesse d'étirement à l'extinction.
Ce module relève d'une régression, pas d'une classification.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance
from flame.common.paths import processed_path, psi_dir

XLSX = psi_dir("PSI-142") / "csv" / "PSI data summary Princeton with links.xlsx"

FUELS = {"nC7": "n-heptane", "nC8": "n-octane", "nC10": "n-decane",
         "nC12": "n-dodecane", "nC14": "n-tetradecane"}

# Lu dans la feuille « data summary », l'index rédigé par Princeton.
SHEET_CONTEXT = {
    "data ID 1": {
        "extinction_type": "cool",
        "ozone": False,
        "publication": "C.B. Reuter et al., Combustion and Flame 179 (2017) 23-32",
        "figure": "Fig. 4",
    },
    "data ID 2": {
        "extinction_type": "cool",
        "ozone": True,
        "publication": "C.B. Reuter et al., Combustion and Flame 179 (2017) 23-32",
        "figure": "Fig. 5",
    },
    "data ID 3": {
        "extinction_type": "hot",
        "ozone": False,
        "publication": "C.B. Reuter et al., Combustion and Flame 179 (2017) 23-32",
        "figure": "Fig. 7",
    },
}

PREMIXED_CONTEXT = {
    "publication": "C.B. Reuter et al., Proc. Combustion Institute 37 (2019) 1851",
    "figure": "Fig. 2",
}
THERMOMETRY_CONTEXT = {
    "publication": "O.R. Yehia et al., Proc. Combustion Institute 37 (2019) 1717",
    "figure": "Fig. 9",
}


def _grid(sheet: str) -> pd.DataFrame:
    grid = pd.read_excel(XLSX, sheet_name=sheet, header=None).dropna(
        axis=1, how="all"
    )
    grid.columns = range(grid.shape[1])
    return grid


def _extinction_sheet(sheet: str) -> pd.DataFrame:
    """Feuilles 1 / 2 / 3 : blocs empilés verticalement, un par carburant."""
    grid = _grid(sheet)
    context = SHEET_CONTEXT[sheet]
    rows, current_fuel = [], None

    for _, row in grid.iterrows():
        label = row[0]
        if pd.notna(label) and str(label).strip() in FUELS:
            current_fuel = FUELS[str(label).strip()]
            continue
        yf = pd.to_numeric(row[1], errors="coerce")
        strain = pd.to_numeric(row[2], errors="coerce")
        if pd.isna(yf) or pd.isna(strain) or current_fuel is None:
            continue
        rows.append(
            {
                "fuel": current_fuel,
                "fuel_mass_fraction": float(yf),
                "extinction_strain_rate_1_s": float(strain),
                "data_id": sheet,
                **context,
            }
        )
    return pd.DataFrame(rows)


def _premixed_sheet() -> pd.DataFrame:
    """data ID 4 : limites d'extinction en prémélange.

    Grandeurs différentes des feuilles 1-3 : ici l'abscisse est la vitesse
    d'étirement et la mesure est une richesse (phi) à l'extinction. Le
    carburant n'est pas indiqué dans la feuille.
    """
    grid = _grid("data ID 4")
    rows = []
    for _, row in grid.iterrows():
        strain = pd.to_numeric(row[0], errors="coerce")
        if pd.isna(strain):
            continue
        for column, name in [(1, "phi_hot_extinction"), (2, "phi_cool_extinction")]:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                continue
            rows.append(
                {
                    "strain_rate_1_s": float(strain),
                    "measurement": name,
                    "value": float(value),
                    "data_id": "data ID 4",
                    **PREMIXED_CONTEXT,
                }
            )
    return pd.DataFrame(rows)


def _thermometry_sheet() -> pd.DataFrame:
    """data ID 8 : température maximale, blocs « 550,O3 » / « 550, no O3 » ...

    L'étiquette de bloc porte deux informations collées : une température de
    référence et la présence ou non d'ozone.
    """
    grid = _grid("data ID 8")
    rows, reference_temp, ozone = [], None, None

    for _, row in grid.iterrows():
        label = str(row[0]).strip() if pd.notna(row[0]) else ""
        if "," in label and any(ch.isdigit() for ch in label.split(",")[0]):
            reference_temp = float(label.split(",")[0])
            ozone = "no o3" not in label.lower()
            continue
        xf = pd.to_numeric(row[0], errors="coerce")
        t_max = pd.to_numeric(row[1], errors="coerce")
        if pd.isna(xf) or pd.isna(t_max) or reference_temp is None:
            continue
        rows.append(
            {
                "fuel": "n-dodecane",
                "fuel_mole_fraction": float(xf),
                "reference_temp_k": reference_temp,
                "ozone": ozone,
                "max_temperature_k": float(t_max),
                "data_id": "data ID 8",
                **THERMOMETRY_CONTEXT,
            }
        )
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """Table principale : limites d'extinction, feuilles 1 / 2 / 3."""
    df = pd.concat(
        [_extinction_sheet(s) for s in SHEET_CONTEXT], ignore_index=True
    )
    return add_provenance(df, investigation="PSI-142", gravity="1g")


def load_other() -> pd.DataFrame:
    """Mesures annexes : prémélange (ID 4) et thermométrie (ID 8)."""
    df = pd.concat([_premixed_sheet(), _thermometry_sheet()], ignore_index=True)
    return add_provenance(df, investigation="PSI-142", gravity="1g")


def main() -> None:
    limits = load()
    other = load_other()

    limits_path = processed_path("psi142_extinction_limits.csv")
    other_path = processed_path("psi142_other_measurements.csv")
    limits.to_csv(limits_path, index=False, encoding="utf-8")
    other.to_csv(other_path, index=False, encoding="utf-8")

    print(f"{limits_path.name:34} {len(limits):3d} limites d'extinction")
    print(f"{other_path.name:34} {len(other):3d} mesures annexes (ID 4 + ID 8)")
    print("\nlimites d'extinction, par feuille :")
    print(
        limits.groupby(["data_id", "extinction_type", "ozone"])
        .size()
        .rename("lignes")
        .to_string()
    )
    print("\npar carburant :")
    print(limits["fuel"].value_counts().to_string())
    print(
        "\nvitesse d'etirement a l'extinction : "
        f"{limits['extinction_strain_rate_1_s'].min():.1f} "
        f"-> {limits['extinction_strain_rate_1_s'].max():.1f} 1/s"
    )
    print(f"gravite : {limits['gravity'].unique().tolist()}  <- au sol, pas ISS")


if __name__ == "__main__":
    main()
