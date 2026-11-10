"""MSL-1 / DCE — gouttes d'heptane libres brulees en vol (Spacelab, 1997).

SOURCE. Ackerman, Colantonio, Crouch, Dryer, Haggard, Linteris, Marchese,
Nayagam, Voss, Williams, Zhang, « A Treatment of Measurements of Heptane
Droplet Combustion Aboard MSL-1 », NASA/TM-2003-212553, octobre 2003,
Table 1, p. 17-18. https://ntrs.nasa.gov/citations/20040000941

Le tableau a ete recopie a la main dans data/context/msl1_dce_heptane.csv,
une ligne par essai, avec la colonne `usable` et sa raison :

    - 3 essais sans vue de la goutte (17, 27, 33) : pas de constante K ;
    - l'essai 18, ou la goutte a heurte l'allumeur : convection forcee ;
    - 2 gouttes attachees a une fibre (35, 37) : la fibre change K.

Reste 32 essais avec une constante K « overall » mesuree.

POURQUOI CE JEU COMPTE. FLEX (PSI-69) s'arrete a 0,69 atm et, pour
l'heptane, a 30 % d'oxygene. MSL-1 a brule de l'heptane a 0,25 et 0,50 bar,
jusqu'a 40 % d'oxygene (K mesure), dans de l'helium au lieu de l'azote.

CE QU'IL NE DONNE PAS. Aucune etiquette extinction / combustion complete
comparable a celle de FLEX : il ne sert qu'a la constante K, jamais au
modele d'extinction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flame.common.paths import PROJECT_ROOT

CSV = PROJECT_ROOT / "data" / "context" / "msl1_dce_heptane.csv"
MMHG_PER_BAR = 750.062
TARGET = "burning_rate_mm2_s"
SOURCE = "MSL-1 DCE (NASA/TM-2003-212553)"


def load(usable_only: bool = True) -> pd.DataFrame:
    """Les essais MSL-1, aux noms de colonnes de PSI-69."""
    raw = pd.read_csv(CSV)
    if usable_only:
        raw = raw[raw["usable"] == "yes"]
    o2 = raw["o2_pct"] / 100.0
    he = np.where(raw["diluent"] == "He", 1.0 - o2, 0.0)
    out = pd.DataFrame({
        "test_id": raw["test"].astype(str).radd("MSL1-"),
        "fuel": "Heptane",
        "pressure_mmhg": raw["pressure_bar"] * MMHG_PER_BAR,
        "o2_frac": o2,
        "n2_frac": np.where(raw["diluent"] == "He", 0.0, 1.0 - o2),
        "co2_frac": 0.0,
        "he_frac": he,
        "d0_mm": raw["d0_mm"],
        TARGET: raw["k_overall_mm2_s"],
        "burn_time_s": raw["burn_time_s"],
        "investigation": SOURCE,
        "gravity": "microgravity",
    })
    out["pressure_atm"] = out["pressure_mmhg"] / 760.0
    return out.reset_index(drop=True)
