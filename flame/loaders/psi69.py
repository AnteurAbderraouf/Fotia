"""PSI-69 / FLEX-1 — efficacité des suppresseurs sur gouttelettes (ISS).

Module « Suppression » : quelles conditions ambiantes éteignent une flamme de
gouttelette en microgravité ? C'est le seul jeu de données du catalogue qui
porte sur la suppression d'incendie (CO2 dans 123 burns, hélium dans 50).

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-69/csv/FLEX.csv

Particularités du fichier NASA, toutes vérifiées sur les données le 2026-09-22 :

  * Encodé en cp1252, pas en UTF-8.

  * Les indices chimiques ont été perdus dans les en-têtes : « O » désigne O2,
    « N » désigne N2 et « CO » désigne **CO2**, pas le monoxyde de carbone.
    Vérifié plutôt que supposé : les quatre fractions molaires somment à
    0.98–1.01 (ce sont donc les seuls gaz du mélange) et les effectifs
    non nuls — 123 pour CO2, 50 pour He — correspondent exactement à la
    description NASA de l'expérience.

  * « Burning rate; mm » a perdu son unité de la même façon : les valeurs vont
    de 0.05 à 1.06, ce qui correspond à la constante de combustion K d'une
    gouttelette en mm^2/s.

  * Le tiret cadratin (U+2013) sert de marqueur de valeur manquante :
    6 cellules en diamètre initial, 28 en diamètre d'extinction.

  * Deux temps de combustion sont notés approximatifs (« ~6 », « ~9 »).

  * Un burn porte une pression ambiante de 0 mmHg (test #114), physiquement
    impossible — il n'y a pas de flamme dans le vide, et la ligne est par
    ailleurs normale. Traité comme une saisie manquante.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import (
    add_provenance,
    check_fractions_sum_to_one,
    to_numeric,
)
from flame.common.paths import processed_path, psi_dir

CSV = psi_dir("PSI-69") / "csv" / "FLEX.csv"
ENCODING = "cp1252"

# Marqueur NASA de valeur absente. Écrit en échappement plutôt qu'en littéral
# pour qu'il survive à toute manipulation du fichier source.
RENAME = {
    "FLEX Test #": "test_id",
    "FLEX Identifier": "flex_identifier",
    "Test Date": "test_date",
    "Test GMT": "test_gmt",
    "Fuel": "fuel",
    "Ambient pressure; mmHg": "pressure_mmhg",
    "O initial ambient composition; mole fraction": "o2_frac",
    "N initial ambient composition; mole fraction": "n2_frac",
    "CO initial ambient composition; mole fraction": "co2_frac",
    "He initial ambient composition; mole fraction": "he_frac",
    "Droplet initial diameter; mm": "d0_mm",
    "Visible flame extinction diameter; mm": "dext_mm",
    "Burning rate; mm": "burning_rate_mm2_s",
    "Burn time; s": "burn_time_s",
    "Test end": "test_end",
}

# --- La séparation qui compte -------------------------------------------------
#
# FEATURES ne contient que ce qui est connu AVANT d'allumer la gouttelette.
#
# POST_BURN est mesuré pendant ou après la combustion. Ces colonnes restent
# dans la table — elles sont réelles et utiles pour afficher un burn historique
# dans le dashboard — mais les utiliser comme entrées d'un modèle qui prédit
# l'extinction serait circulaire : on prédirait l'extinction à partir de la
# preuve qu'elle a déjà eu lieu. Le score serait excellent et le modèle
# inutilisable, puisqu'au moment où on voudrait la prédiction aucune de ces
# valeurs n'existe encore.
#
# n2_frac est volontairement hors FEATURES : les quatre fractions molaires
# somment à 1, donc elles sont linéairement dépendantes. Les inclure toutes
# les quatre rend les coefficients d'une régression logistique instables et
# ininterprétables. N2 est le gaz de remplissage, c'est lui qu'on retire.

FEATURES = ["fuel", "pressure_mmhg", "o2_frac", "co2_frac", "he_frac", "d0_mm"]
POST_BURN = ["dext_mm", "burning_rate_mm2_s", "burn_time_s"]
LABEL = "extinction"


def load(drop_disruption: bool = True) -> pd.DataFrame:
    """Charge FLEX.csv et renvoie une table propre.

    drop_disruption : les 61 burns marqués « Disruption » sont des essais
        ratés — la gouttelette s'est fragmentée ou a dérivé hors du champ.
        Ce n'est pas un résultat de combustion mais un incident de mesure,
        donc ils ne peuvent pas porter d'étiquette. True les écarte
        (il reste 213 burns), False les garde avec `extinction` à NaN.
    """
    df = pd.read_csv(CSV, encoding=ENCODING).rename(columns=RENAME)

    for col in ["d0_mm", "dext_mm", "burn_time_s"]:
        df[col], approx = to_numeric(df[col])
        if approx.any():
            df[f"{col}_approx"] = approx

    # Pression nulle = saisie manquante, pas un vide physique.
    df.loc[df["pressure_mmhg"] <= 0, "pressure_mmhg"] = pd.NA
    df["pressure_atm"] = df["pressure_mmhg"] / 760.0

    # Étiquette binaire. Disruption n'est ni un 0 ni un 1 : c'est une absence
    # de résultat, d'où le NaN plutôt qu'une troisième classe.
    # Int64 (entier « nullable ») et non int : Disruption doit pouvoir rester
    # NaN quand drop_disruption=False, ce qu'un int classique ne permet pas.
    df[LABEL] = df["test_end"].map({"Extinction": 1, "Completion": 0}).astype("Int64")

    # Confirme que O2/N2/CO2/He sont bien les seuls gaz du mélange — c'est ce
    # test qui prouve que la colonne « CO » est du CO2. Échoue bruyamment si
    # l'hypothèse est fausse, plutôt que de produire une table silencieusement
    # erronée.
    check_fractions_sum_to_one(df, ["o2_frac", "n2_frac", "co2_frac", "he_frac"])

    df = add_provenance(df, investigation="PSI-69", gravity="microgravity")

    if drop_disruption:
        df = df[df["test_end"] != "Disruption"].reset_index(drop=True)

    return df


def main() -> None:
    everything = load(drop_disruption=False)
    modeling = load(drop_disruption=True)

    all_path = processed_path("psi69_all_burns.csv")
    model_path = processed_path("psi69_suppression.csv")
    everything.to_csv(all_path, index=False, encoding="utf-8")
    modeling.to_csv(model_path, index=False, encoding="utf-8")

    print(f"{all_path.name:28} {len(everything):3d} burns (tout, Disruption inclus)")
    print(f"{model_path.name:28} {len(modeling):3d} burns (modélisables)")
    print()
    print("Étiquette :")
    counts = modeling[LABEL].value_counts().sort_index()
    total = int(counts.sum())
    for value, count in counts.items():
        name = "extinction" if value == 1 else "combustion complète"
        print(f"  {name:20} {count:3d}  ({count / total:.0%})")
    print(f"  -> plancher majoritaire : {counts.max() / total:.0%}")
    print()
    print("Features (connues avant allumage) :", ", ".join(FEATURES))
    print("Écartées car post-combustion   :", ", ".join(POST_BURN))


if __name__ == "__main__":
    main()
