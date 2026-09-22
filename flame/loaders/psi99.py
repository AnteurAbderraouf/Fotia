"""PSI-99 / Saffire-II — le même matériau brûlé en microgravité et sur Terre.

NEUF ÉCHANTILLONS. C'est beaucoup trop peu pour entraîner quoi que ce soit, et
ce n'est pas l'intérêt de ce jeu de données. Sa valeur est qu'il porte, sur la
MÊME LIGNE, le comportement d'un matériau en microgravité et son comportement
à 1g. Aucune autre source du catalogue ne fait ça. C'est la carte d'affichage
du dashboard : « le feu dans l'espace n'est pas le feu sur Terre », démontré
plutôt qu'affirmé.

Source (lecture seule) :
    combustion_science/microgravity_investigation/PSI-99/csv/SAFFIRE-2.csv

PRÉCISION SUR L'HISTOIRE, vérifiée ligne à ligne le 2026-09-22.
Le handoff résume : « le silicone brûle complètement sur Terre et environ pas
en microgravité ; le SIBAL se propage à 2.1-2.6 mm/s dans les deux ». La
seconde moitié est inexacte et c'est dommage, parce que la vérité est plus
intéressante. Le SIBAL n'a pas de vitesse de propagation chiffrée à 1g : la
colonne porte « Acceleratory ». Autrement dit,

    en microgravité   il se propage à vitesse constante, 2.1 puis 2.6 mm/s
    à 1g              il accélère, sans vitesse stable à donner

La flottabilité entretient et emballe la flamme sur Terre. En microgravité,
privée de convection naturelle, elle avance régulièrement. Ce n'est pas
« pareil dans les deux cas », c'est deux régimes de propagation différents.

Le silicone, lui, confirme bien le handoff : à peu près rien en microgravité
(4 échantillons sur 4), alors qu'à 1g deux brûlent complètement.

PIÈGES DU FICHIER :
  * Les en-têtes « ?-g Burn Length » et « ?-g Spread Length » portent un « ? »
    qui est un « µ » perdu à l'encodage — même accident que les indices de
    PSI-69.
  * Quatre lignes portent « 037 mm » là où l'épaisseur est 0.37 mm : le point
    décimal a sauté. Confirmé par la ligne 2-7 qui écrit « 0.85 and 0.37 mm ».
  * La colonne Material n'est renseignée qu'au premier échantillon d'une série
    (2-6 hérite de 2-5, 2-9 hérite de 2-8) : il faut la propager.
  * Les six dernières lignes du fichier ne sont pas des échantillons : deux
    visualisations d'écoulement avant essai, et quatre notes explicatives.
  * Toutes les grandeurs portent leur unité dans la valeur (« 29 cm »,
    « 20 cm/s », « 9.2 s »).
  * Plusieurs résultats sont qualitatifs et ne se chiffrent pas :
    « Complete », « Acceleratory », « Insignificant », « n/a (Nomex) ».
    Ils sont conservés tels quels dans les colonnes `_raw`, parce qu'écraser
    « Complete » en nombre perdrait ce que l'essai a réellement montré.
"""

from __future__ import annotations

import pandas as pd

from flame.common.clean import add_provenance, as_text, strip_unit
from flame.common.paths import processed_path, psi_dir

CSV = psi_dir("PSI-99") / "csv" / "SAFFIRE-2.csv"
ENCODING = "utf-8-sig"

RENAME = {
    "Sample Number": "sample_id",
    "Material": "material",
    "Samle Thickness": "thickness_raw",  # la faute de frappe est dans le fichier NASA
    "Sample Length": "length_raw",
    "Sample Width (cm)": "width_raw",
    "Air Flow (cm/s)": "air_flow_raw",
    "Percent O2 (Note 1)": "o2_percent_raw",
    "Ignition Power (W)": "ignition_power_raw",
    "Ignition Time (s)": "ignition_time_raw",
    "Burn Time (s)": "burn_time_raw",
    "Flow Direction": "flow_direction",
    "?-g Burn Length": "ug_burn_length_raw",
    "?-g Spread Length": "ug_spread_rate_raw",
    "1-g Burn Length": "g1_burn_length_raw",
    "1-g Spread Rate": "g1_spread_rate_raw",
    "Camera 1": "camera_1",
    "Camera 2": "camera_2",
}

# « 037 mm » est 0.37 mm : le point decimal a saute. Verifie contre la ligne
# 2-7, qui ecrit « 0.85 and 0.37 mm » pour le meme materiau.
THICKNESS_FIXES = {"037 mm": "0.37 mm"}

# Resultats qualitatifs qu'aucun nombre ne remplace.
QUALITATIVE = {"complete", "acceleratory", "insignificant", "n/a (nomex)", "note 4"}

PAIRS = [
    ("ug_burn_length_raw", "ug_burn_length_cm"),
    ("g1_burn_length_raw", "g1_burn_length_cm"),
    ("ug_spread_rate_raw", "ug_spread_rate_mm_s"),
    ("g1_spread_rate_raw", "g1_spread_rate_mm_s"),
]


def _length_in_mm(series: pd.Series) -> pd.Series:
    """Longueur en millimetres, quelle que soit l'unite ecrite.

    Renvoie NaN quand la cellule n'est pas une mesure mais un renvoi vers une
    figure : « See Fig. 5 » contient un 5 qui n'est pas une epaisseur.
    """
    text = as_text(series)
    is_reference = text.str.contains(r"see|fig|note", case=False, na=False)
    value = strip_unit(text.mask(is_reference))
    # Test sans expression reguliere : l'unite est en fin de cellule.
    # Une regex etait ici un risque inutile — une sequence d'echappement
    # mal ecrite y avait glisse un caractere de controle invisible, et le
    # motif ne pouvait plus rien reconnaitre. Un endswith ne ment pas.
    in_cm = text.str.strip().str.endswith("cm").fillna(False)
    return value.mask(in_cm, value * 10)


def load() -> pd.DataFrame:
    """Les 9 échantillons, sans les lignes de service ni les notes."""
    df = pd.read_csv(CSV, encoding=ENCODING)
    # Plusieurs en-tetes NASA trainent un espace final (« Samle Thickness »).
    df.columns = [str(c).strip() for c in df.columns]
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise KeyError(f"colonnes attendues absentes de {CSV.name} : {sorted(missing)}")
    df = df.rename(columns=RENAME)

    # Un echantillon reel porte un identifiant de la forme « 2-n ». Les autres
    # lignes sont des visualisations d'ecoulement et des notes de bas de table.
    df["sample_id"] = as_text(df["sample_id"])
    df = df[df["sample_id"].str.fullmatch(r"2-\d+", na=False)].reset_index(drop=True)

    # Le materiau n'est ecrit qu'au premier echantillon de chaque serie.
    df["material"] = as_text(df["material"]).ffill()

    df["thickness_raw"] = as_text(df["thickness_raw"]).replace(THICKNESS_FIXES)

    # L'epaisseur melange les unites et contient des renvois : « 0.27 mm »,
    # « 1 cm » (soit 10 mm) et « See Fig. 5 », qu'un simple extracteur de
    # nombre lirait comme 5 mm. D'ou une conversion explicite plutot qu'un
    # strip_unit aveugle.
    # 2-7 est un echantillon composite (« 0.85 and 0.37 mm » : PMMA + Nomex).
    # On garde la premiere epaisseur mais on le signale, comme pour la longueur.
    df["thickness_composite"] = (
        df["thickness_raw"].str.contains(" and ", case=False, na=False)
    )
    df["thickness_mm"] = _length_in_mm(df["thickness_raw"])

    # « 5 and 24 cm » decrit un echantillon composite PMMA + Nomex : deux
    # longueurs, pas une. Retenir le premier nombre serait faux, on laisse
    # vide et on conserve le texte.
    length_text = as_text(df["length_raw"])
    df["length_composite"] = length_text.str.contains(" and ", case=False, na=False)
    df["length_cm"] = strip_unit(length_text).mask(df["length_composite"])
    df["width_cm"] = strip_unit(df["width_raw"])
    df["air_flow_cm_s"] = strip_unit(df["air_flow_raw"])
    # « 22.1 to 22.0 » est une derive au cours de l'essai, « ~ 21.5 » une
    # approximation : on garde la premiere valeur et on signale les deux cas.
    o2_text = as_text(df["o2_percent_raw"])
    df["o2_percent"] = strip_unit(o2_text)
    df["o2_is_range"] = o2_text.str.contains(" to ", case=False, na=False)
    df["o2_is_approx"] = o2_text.str.startswith("~").fillna(False)
    df["ignition_power_w"] = strip_unit(df["ignition_power_raw"])
    df["ignition_time_s"] = strip_unit(df["ignition_time_raw"])
    df["burn_time_s"] = strip_unit(df["burn_time_raw"])

    for raw_column, parsed_column in PAIRS:
        text = as_text(df[raw_column])
        df[raw_column] = text
        numeric = strip_unit(text)
        # Un resultat qualitatif ne doit pas etre transforme en nombre.
        is_qualitative = text.str.lower().isin(QUALITATIVE)
        df[parsed_column] = numeric.mask(is_qualitative)
        df[parsed_column.replace("_cm", "").replace("_mm_s", "") + "_qualitative"] = (
            text.where(is_qualitative)
        )

    df["burned_in_microgravity"] = (df["ug_burn_length_cm"].fillna(0) > 0).astype("Int64")

    return add_provenance(df, investigation="PSI-99", gravity="both")


def main() -> None:
    df = load()
    path = processed_path("psi99_microgravity_vs_earth.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"{path.name} : {len(df)} echantillons\n")

    view = df[
        [
            "sample_id",
            "material",
            "thickness_mm",
            "flow_direction",
            "ug_burn_length_raw",
            "ug_spread_rate_raw",
            "g1_burn_length_raw",
            "g1_spread_rate_raw",
        ]
    ]
    view.columns = [
        "ech.",
        "materiau",
        "ep. mm",
        "ecoulement",
        "ug longueur",
        "ug vitesse",
        "1g longueur",
        "1g vitesse",
    ]
    print(view.to_string(index=False))

    print("\nle contraste, materiau par materiau :")
    for material, block in df.groupby("material"):
        burned = int(block["burned_in_microgravity"].sum())
        print(
            f"  {material[:34]:<36} {burned}/{len(block)} ont brule en microgravite"
        )


if __name__ == "__main__":
    main()
