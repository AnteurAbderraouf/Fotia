"""PSI-117 / FLEX — flammes froides, données derrière les figures publiées.

Complément du module « Flammes froides » (PSI-39). Mêmes grandeurs — diamètres
d'extinction de flamme chaude et de flamme froide — sur d'autres alcanes
(n-heptane, n-decane, n-dodecane) et sur une plage de pression plus large.

Source (lecture seule) :
    combustion_science/ground_investigation/PSI-117/csv/

CE QUE SONT CES FICHIERS, ET POURQUOI ÇA CHANGE TOUT.
`ReadMe_List_of_Data.xlsx` est l'index de NASA : chaque fichier y est associé à
un DOI et à un numéro de figure. Ce ne sont donc pas des relevés bruts de
campagne, mais les données derrière les figures de trois articles publiés.
Deux conséquences.

  1. Chaque fichier mélange des blocs « Expt » (mesures) et « Sim »/« Model »
     (simulations numériques) POUR LES MÊMES CONDITIONS. C'est très exactement
     la non-indépendance annoncée au §9 du handoff. Une simulation à 1 atm et
     la mesure à 1 atm ne sont pas deux observations : c'est une observation et
     sa prédiction. Les mettre des deux côtés d'un découpage train/test ferait
     grimper le score sans qu'aucun apprentissage ait eu lieu.
     La colonne `source` distingue les deux, et tout découpage doit se faire
     par condition physique, jamais par ligne.

  2. `PROCI_2023_C12_Extinction_Diameter_vs_pressure` contient aussi deux blocs
     de barres d'erreur. Ce ne sont pas des observations. Ils sont marqués
     `source = error_bar` et doivent être écartés de toute modélisation.

STRUCTURE DES FICHIERS.
Chaque fichier est une grille où plusieurs blocs sont posés côte à côte,
séparés par une colonne entièrement vide — une mise en page de tableur, pas un
format de données. Les quatre fichiers scalaires utilisent quatre conventions
différentes pour dire de quel bloc il s'agit, d'où une lecture explicite par
fichier plutôt qu'un parseur générique qui devinerait :

    MST_2024_Dext        le type est en première cellule de l'en-tête
    MST_2024_Kavg        idem
    PROCI_2023_Alkane    le carburant est en en-tête, le type en première
                         cellule de la PREMIÈRE LIGNE DE DONNÉES
    PROCI_2023_C12_...   le type est encodé dans le nom de la colonne
                         (« _EXPT », « _model », « _ErrorBar »)

FICHIERS NON TRAITÉS ICI.
Les cinq autres CSV du dossier sont des séries temporelles au sein d'un seul
essai — diamètre ou radiance en fonction du temps — et non des observations
par condition. Ils relèvent d'un autre usage (tracer une histoire de
combustion) et n'entrent pas dans une table par essai :
    MST_2024_Burning_history_Flame_diameter, MST_2024_Radiance_Flame_Diameter,
    PROCI_2019_Stable_Cool, PROCI_2019_Oscillatory_Cool_Flame,
    PROCI_C12_2019_Three_Stage
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance, as_text
from flame.common.paths import processed_path, psi_dir

CSV_DIR = psi_dir("PSI-117") / "csv"

# Conditions et références tirées de ReadMe_List_of_Data.xlsx, l'index NASA.
FILE_CONTEXT = {
    "MST_2024_Dext.csv": {
        "fuel": "n-dodecane",
        "d0_mm": 4.0,
        "xo2": 0.21,
        "doi": "10.1007/s12217-024-10115-x",
        "figure": "Fig. 5",
    },
    "MST_2024_Kavg.csv": {
        "fuel": "n-dodecane",
        "d0_mm": 4.5,
        "xo2": 0.21,
        "doi": "10.1007/s12217-024-10115-x",
        "figure": "Fig. 4",
    },
    "PROCI_2023_Alkane_Extinction_Diameter_1atm.csv": {
        "pressure_atm": 1.0,
        "xo2": 0.21,
        "doi": "10.1016/j.proci.2022.07.094",
        "figure": "Fig. 3",
    },
    "PROCI_2023_C12_Extinction_Diameter_vs_pressure.csv": {
        "fuel": "n-dodecane",
        "d0_mm": 4.0,
        "xo2": 0.21,
        "doi": "10.1016/j.proci.2022.07.094",
        "figure": "Fig. 7",
    },
}

FUEL_CODES = {"C7": "n-heptane", "C10": "n-decane", "C12": "n-dodecane"}

# Vocabulaire NASA -> nom de mesure unique dans la table de sortie.
MEASUREMENT_NAMES = {
    "dext_cool": "dext_cool_mm",
    "dext_cool_expt": "dext_cool_mm",
    "dext_cf": "dext_cool_mm",
    "dext_cf_expt": "dext_cool_mm",
    "dext_cf_model": "dext_cool_mm",
    "dext_hot": "dext_hot_mm",
    "dext_hf": "dext_hot_mm",
    "dext_hf_expt": "dext_hot_mm",
    "dext_hf_model": "dext_hot_mm",
    "kavg_cf": "k_cool_mm2_s",
    "kb (mm2/s)": "k_burn_mm2_s",
}

SOURCE_NAMES = {
    "expt": "experiment",
    "experiment": "experiment",
    "sim": "simulation",
    "model": "simulation",
}


def _split_blocks(grid: pd.DataFrame) -> list[pd.DataFrame]:
    """Découpe une grille de tableur sur ses colonnes entièrement vides.

    Une colonne vide n'est pas une donnée, c'est une gouttière de mise en page.
    Elle sépare des tableaux indépendants posés côte à côte dans la même
    feuille.
    """
    blocks, current = [], []
    for column in range(grid.shape[1]):
        if grid[column].isna().all():
            if current:
                blocks.append(grid[current])
                current = []
        else:
            current.append(column)
    if current:
        blocks.append(grid[current])
    return blocks


def _read_grid(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        CSV_DIR / filename, header=None, dtype=str, encoding="utf-8-sig"
    )


def _emit(rows: list, filename: str, **fields) -> None:
    rows.append({**FILE_CONTEXT[filename], **fields, "data_file": filename})


def _parse_header_typed(filename: str, x_name: str) -> list[dict]:
    """MST_2024_Dext et MST_2024_Kavg : le type est en tête de bloc.

    En-tête : [type, abscisse, mesure...]. La première colonne des données est
    vide, elle ne sert qu'à porter l'étiquette du bloc.
    """
    rows: list[dict] = []
    for block in _split_blocks(_read_grid(filename)):
        header = [as_text(pd.Series([v])).iloc[0] for v in block.iloc[0]]
        source = SOURCE_NAMES[str(header[0]).strip().lower()]
        body = block.iloc[1:]
        x_values = pd.to_numeric(body.iloc[:, 1], errors="coerce")
        for offset, raw_name in enumerate(header[2:], start=2):
            if raw_name is None or pd.isna(raw_name):
                continue
            measurement = MEASUREMENT_NAMES[str(raw_name).strip().lower()]
            values = pd.to_numeric(body.iloc[:, offset], errors="coerce")
            for x, value in zip(x_values, values):
                if pd.isna(x) or pd.isna(value):
                    continue
                _emit(
                    rows,
                    filename,
                    **{x_name: float(x)},
                    measurement=measurement,
                    value=float(value),
                    source=source,
                )
    return rows


def _parse_alkane(filename: str) -> list[dict]:
    """PROCI_2023_Alkane : carburant en en-tête, type en première ligne.

    Chaque bloc est [code carburant, Do, Dext_HF, Dext_CF] et ses lignes sont
    de vraies observations appariées — le seul fichier du lot où un diamètre
    initial, un diamètre d'extinction chaud et un froid appartiennent au même
    essai.
    """
    rows: list[dict] = []
    for block in _split_blocks(_read_grid(filename)):
        header = [str(v).strip() for v in block.iloc[0]]
        fuel = FUEL_CODES[header[0]]
        source = SOURCE_NAMES[str(block.iloc[1, 0]).strip().lower()]
        body = block.iloc[1:]
        d0 = pd.to_numeric(body.iloc[:, 1], errors="coerce")
        for offset, raw_name in enumerate(header[2:], start=2):
            measurement = MEASUREMENT_NAMES[raw_name.lower()]
            values = pd.to_numeric(body.iloc[:, offset], errors="coerce")
            for initial, value in zip(d0, values):
                if pd.isna(initial) or pd.isna(value):
                    continue
                _emit(
                    rows,
                    filename,
                    fuel=fuel,
                    d0_mm=float(initial),
                    measurement=measurement,
                    value=float(value),
                    source=source,
                )
    return rows


def _parse_named_columns(filename: str) -> list[dict]:
    """PROCI_2023_C12 : le type est encodé dans le nom de la colonne.

    Les deux blocs « ErrorBar » sont des barres d'erreur et non des
    observations : marqués comme tels pour qu'aucune modélisation ne les
    ramasse par inadvertance.
    """
    rows: list[dict] = []
    for block in _split_blocks(_read_grid(filename)):
        name = str(block.iloc[0, 1]).strip()
        lowered = name.lower()
        if "errorbar" in lowered:
            source, measurement = "error_bar", "dext_cool_mm"
        else:
            source = "simulation" if "model" in lowered else "experiment"
            measurement = MEASUREMENT_NAMES[lowered]
        body = block.iloc[1:]
        pressures = pd.to_numeric(body.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(body.iloc[:, 1], errors="coerce")
        for pressure, value in zip(pressures, values):
            if pd.isna(pressure) or pd.isna(value):
                continue
            _emit(
                rows,
                filename,
                pressure_atm=float(pressure),
                measurement=measurement,
                value=float(value),
                source=source,
                error_bar_side="+" if name.endswith("+") else ("-" if name.endswith("-") else None),
            )
    return rows


def load(drop_error_bars: bool = True) -> pd.DataFrame:
    """Table longue : une ligne par (condition, mesure, origine).

    Le format long est délibéré. Les quatre fichiers ne mesurent pas les mêmes
    grandeurs aux mêmes conditions, et seul PROCI_2023_Alkane apparie
    réellement plusieurs mesures au sein d'un même essai. Élargir la table
    reviendrait à inventer des jointures que les données ne portent pas.
    """
    rows: list[dict] = []
    rows += _parse_header_typed("MST_2024_Dext.csv", "pressure_atm")
    rows += _parse_header_typed("MST_2024_Kavg.csv", "pressure_atm")
    rows += _parse_alkane("PROCI_2023_Alkane_Extinction_Diameter_1atm.csv")
    rows += _parse_named_columns("PROCI_2023_C12_Extinction_Diameter_vs_pressure.csv")

    df = pd.DataFrame(rows)
    ordered = [
        "fuel",
        "d0_mm",
        "pressure_atm",
        "xo2",
        "measurement",
        "value",
        "source",
        "error_bar_side",
        "data_file",
        "doi",
        "figure",
    ]
    df = df.reindex(columns=ordered)

    # NASA classe PSI-117 sous ground_investigation mais le decrit comme une
    # investigation en vol (handoff §6). On suit la description, pas le
    # rangement, et on le signale plutot que de trancher en silence.
    # source=None : la provenance varie ligne a ligne (mesure / simulation /
    # barre d'erreur) et c'est toute la valeur de cette table.
    df = add_provenance(
        df, investigation="PSI-117", gravity="microgravity", source=None
    )

    if drop_error_bars:
        df = df[df["source"] != "error_bar"].reset_index(drop=True)

    return df


def cool_flame_observations() -> pd.DataFrame:
    """Sous-ensemble appariable : le seul fichier à vraies observations.

    PROCI_2023_Alkane donne, pour un même essai, le diamètre initial, le
    diamètre d'extinction de la flamme chaude et celui de la flamme froide.
    Un diamètre de flamme froide nul signifie qu'aucune flamme froide n'a
    persisté — c'est la même étiquette que PSI-39, mais observée sur trois
    alcanes à 1 atm.
    """
    long = load()
    alkane = long[long["data_file"].str.startswith("PROCI_2023_Alkane")]
    wide = alkane.pivot_table(
        index=["fuel", "d0_mm", "pressure_atm", "source"],
        columns="measurement",
        values="value",
    ).reset_index()
    wide.columns.name = None
    if "dext_cool_mm" in wide:
        wide["cool_flame"] = (wide["dext_cool_mm"] > 0).astype("Int64")
    return add_provenance(
        wide, investigation="PSI-117", gravity="microgravity", source=None
    )


def main() -> None:
    long = load(drop_error_bars=False)
    observations = cool_flame_observations()

    long_path = processed_path("psi117_measurements.csv")
    obs_path = processed_path("psi117_cool_flame_observations.csv")
    long.to_csv(long_path, index=False, encoding="utf-8")
    observations.to_csv(obs_path, index=False, encoding="utf-8")

    print(f"{long_path.name:38} {len(long):3d} mesures (format long)")
    print(f"{obs_path.name:38} {len(observations):3d} essais apparies (PROCI_2023_Alkane)")
    print("\npar origine :")
    print(long["source"].value_counts().to_string())
    print("\npar grandeur mesuree :")
    print(long["measurement"].value_counts().to_string())
    print("\nmesures x origine (hors barres d'erreur) :")
    real = long[long["source"] != "error_bar"]
    print(pd.crosstab(real["measurement"], real["source"]).to_string())
    print("\nessais apparies -> flamme froide observee :")
    print(pd.crosstab(observations["source"], observations["cool_flame"]).to_string())


if __name__ == "__main__":
    main()
