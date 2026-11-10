"""Contexte releve a la main — incidents, atmospheres, gravite, litterature.

Source (lecture seule, NON regenerable) :
    data/context/contexte_incidents.csv
    data/context/contexte_atmospheres.csv
    data/context/contexte_gravite.csv
    data/context/literature_crosscheck.csv
    data/context/proprietes_combustibles.csv

CE LOADER N'ECRIT JAMAIS. Contrairement aux loaders PSI, il n'a pas de sortie
dans data/processed/ : ces fichiers ont ete releves dans des sources par une
personne, un script ne peut pas les reconstruire. Il les lit, les normalise et
dit clairement ce qu'on a le droit d'en faire.

TROIS GARDE-FOUS PORTES PAR LES DONNEES ELLES-MEMES, PAS PAR L'INTERFACE.

  * `verified` : la colonne « verifie » vaut « non » partout au 2026-11-09.
    Une citation non recoupee sur le texte brut ne doit pas s'afficher comme
    verifiee. Le booleen ne passe a True que si la cellule commence par
    « oui », ce qui est la convention du README de data/context.

  * `established` (incidents) : l'incident Bondarenko vient d'une source
    secondaire, de confiance « faible ». Toute ligne dont la confiance commence
    par « faible » est marquee non etablie, pas seulement celle-la.

  * `pressure_certain` (litterature) : une pression « supposee », « non
    precisee », « plusieurs » ou « non relevee » ne se compare pas a un
    chiffre. Les lignes 4 et 6 tombent ici, et le controle `_check_known_rows`
    echoue si une modification future les faisait repasser.

LES UNITES. Les atmospheres sont donnees en psia, parfois en kPa, parfois en
intervalle (« 5.0 a 15.0 ») ; les modeles parlent en atm. On convertit en
gardant la cellule d'origine a cote : rien n'est remplace, tout est ajoute.
Une pression absente reste absente — elle n'est jamais deduite d'une autre
ligne.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from flame.common.paths import CONTEXT, context_path

ENCODING = "utf-8-sig"  # les quatre fichiers commencent par un BOM
PSIA_PER_ATM = 14.696
KPA_PER_ATM = 101.325

# Colonnes sans lesquelles la ligne ne peut pas etre affichee honnetement :
# la source et l'URL sont exigees partout, c'est la condition d'affichage.
REQUIRED = {
    "incidents": {
        "file": "contexte_incidents.csv",
        "columns": ["id", "type", "date", "vaisseau_lieu", "atmosphere",
                    "source", "url", "citation", "confiance", "verifie"],
    },
    "atmospheres": {
        "file": "contexte_atmospheres.csv",
        "columns": ["id", "type", "vaisseau_situation", "pression_psia",
                    "pression_kpa", "o2_pct", "source", "url", "citation",
                    "confiance", "verifie"],
    },
    "gravity": {
        "file": "contexte_gravite.csv",
        "columns": ["id", "regime", "g_m_s2", "fraction_de_terre", "source",
                    "url", "citation", "confiance", "verifie"],
    },
    "literature": {
        "file": "literature_crosscheck.csv",
        "columns": ["id", "type", "article", "fichier", "page", "combustible",
                    "o2_pct", "pression_atm", "diluant", "gravite",
                    "resultat_observe", "valeur", "citation", "remarque"],
    },
    "fuels": {
        "file": "proprietes_combustibles.csv",
        "columns": ["id", "combustible", "propriete", "valeur", "unite",
                    "temperature_k", "source", "url", "citation", "confiance",
                    "verifie"],
    },
}

# Proprietes dont la partie physique du moteur a besoin, par carburant. Un
# carburant auquel il en manque une n'est pas propose : on ne complete pas.
FUEL_PROPERTIES = [
    "formule",
    "masse_molaire",
    "T_ebullition",
    "enthalpie_vaporisation",
    "enthalpie_combustion_liquide",
    "masse_volumique_liquide",
]

# Lignes de literature_crosscheck dont la pression est connue pour etre
# incertaine (voir leur colonne remarque). Sert de controle, pas de filtre :
# le filtre est generique, ceci verifie qu'il les attrape bien.
KNOWN_UNCERTAIN_PRESSURE = {4, 6}

# Mots qui rendent une cellule de pression inutilisable pour un chiffre.
UNCERTAIN_WORDS = ("suppose", "non precis", "non releve", "plusieurs")

NUMBER = re.compile(r"\d+(?:[.,]\d+)?(?:e-?\d+)?", re.IGNORECASE)


class ContextFileError(FileNotFoundError):
    """Un fichier de contexte manque. Il ne se regenere pas : il se recupere."""


def _normalise_header(name: str) -> str:
    """« O2_pct » -> « o2_pct », « pression_kPa » -> « pression_kpa »."""
    return re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")


def _read(key: str) -> pd.DataFrame:
    spec = REQUIRED[key]
    path = context_path(spec["file"])
    if not path.is_file():
        raise ContextFileError(
            f"Fichier de contexte manquant : {path}\n"
            f"  data/context/ est releve a la main et NE SE REGENERE PAS "
            f"(scripts/build_all.py n'y peut rien).\n"
            f"  Le recuperer depuis git s'il y est versionne "
            f"(`git checkout -- data/context/{spec['file']}`), sinon depuis "
            f"la copie du designer, nasa/Gravity context/."
        )
    df = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)
    df.columns = [_normalise_header(c) for c in df.columns]

    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} : colonnes attendues absentes {missing}. "
            f"Colonnes lues : {list(df.columns)}"
        )

    # Cellules vides -> NA, espaces parasites retires. dtype=str d'abord pour
    # qu'aucune cellule (« 5.0 a 15.0 », « 1e-6 (exigence) ») ne soit
    # convertie en silence par pandas.
    df = df.apply(lambda col: col.str.strip()).replace("", pd.NA)
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    df.attrs["path"] = str(path)
    return df


def _numbers(cell) -> list[float]:
    if cell is None or pd.isna(cell):
        return []
    return [float(n.replace(",", ".")) for n in NUMBER.findall(str(cell))]


def _verified(cell) -> bool:
    return isinstance(cell, str) and cell.lower().startswith("oui")


def _gravity_regime(cell) -> str | None:
    """Ramene les libelles a un vocabulaire fixe, sans en inventer."""
    if cell is None or pd.isna(cell):
        return None
    text = str(cell).lower()
    if "micro" in text or "zero-g" in text or "0 g" in text:
        return "microgravity"
    if text in {"1g", "1 g", "terre", "sol"}:
        return "1g"
    return None


def _add_verification(df: pd.DataFrame) -> pd.DataFrame:
    df["verified"] = df["verifie"].map(_verified)
    return df


def load_incidents() -> pd.DataFrame:
    """Feux de vaisseaux et experiences de feu controle."""
    df = _add_verification(_read("incidents"))
    df["year"] = df["date"].str.extract(r"(\d{4})", expand=False).astype("Int64")
    # Une source « faible » ne s'affiche pas comme un fait etabli.
    df["established"] = ~df["confiance"].str.lower().str.startswith("faible")
    return df


def load_atmospheres() -> pd.DataFrame:
    """Pression et oxygene par vaisseau, essai ou proposition.

    Ajoute pressure_atm_min / pressure_atm_max (egales sauf pour une plage),
    o2_frac, et `pressure_source` qui dit de quelle colonne vient le chiffre.
    """
    df = _add_verification(_read("atmospheres"))
    low, high, origin = [], [], []
    for psia, kpa in zip(df["pression_psia"], df["pression_kpa"]):
        values, unit = _numbers(psia), PSIA_PER_ATM
        source = "psia"
        if not values:
            values, unit, source = _numbers(kpa), KPA_PER_ATM, "kpa"
        if not values:
            low.append(np.nan)
            high.append(np.nan)
            origin.append(None)
            continue
        low.append(min(values) / unit)
        high.append(max(values) / unit)
        origin.append(source)
    df["pressure_atm_min"] = np.round(low, 4)
    df["pressure_atm_max"] = np.round(high, 4)
    df["pressure_source"] = origin
    df["pressure_is_range"] = df["pressure_atm_min"] != df["pressure_atm_max"]
    df.loc[df["pressure_atm_min"].isna(), "pressure_is_range"] = False

    o2 = [(_numbers(cell) or [np.nan])[0] for cell in df["o2_pct"]]
    df["o2_frac"] = np.round(np.array(o2, dtype=float) / 100.0, 4)
    return df


def load_gravity() -> pd.DataFrame:
    """Regimes de gravite. La valeur ISS est une EXIGENCE, pas une mesure."""
    df = _add_verification(_read("gravity"))
    df["g_m_s2_value"] = [(_numbers(c) or [np.nan])[0] for c in df["g_m_s2"]]
    df["fraction_of_earth"] = [
        (_numbers(c) or [np.nan])[0] for c in df["fraction_de_terre"]
    ]
    df["is_requirement"] = df["fraction_de_terre"].str.contains(
        "exigence", case=False, na=False
    )
    return df


def _o2_bounds(cell) -> tuple[float, float]:
    """Bornes d'O2 en fraction molaire, NaN si la cellule n'est pas chiffree.

    « 21 » -> 0.21 ; « 14-25 » -> 0.14..0.25 ; « 0.1-0.4 (fraction molaire) »
    -> 0.1..0.4. « air », « LOI » et « plusieurs » restent NaN : « air » n'est
    pas une mesure, et y mettre 0.21 serait une hypothese non ecrite.
    """
    values = _numbers(cell)
    if not values:
        return np.nan, np.nan
    fractions = [v if v <= 1 else v / 100.0 for v in values]
    return min(fractions), max(fractions)


def load_literature() -> pd.DataFrame:
    """Resultats publies, avec ce qui est comparable chiffre a chiffre.

    `pressure_certain` est False quand la cellule est supposee, imprecise ou
    absente. Une ligne sans pression certaine peut s'afficher, jamais entrer
    dans une comparaison chiffree.
    """
    df = _read("literature")
    df["page"] = pd.to_numeric(df["page"], errors="coerce").astype("Int64")

    lows, highs, certain = [], [], []
    for cell in df["pression_atm"]:
        text = "" if pd.isna(cell) else str(cell).lower()
        values = _numbers(text)
        uncertain = not values or any(word in text for word in UNCERTAIN_WORDS)
        lows.append(min(values) if values else np.nan)
        highs.append(max(values) if values else np.nan)
        certain.append(not uncertain)
    df["pressure_atm_min"] = lows
    df["pressure_atm_max"] = highs
    df["pressure_certain"] = certain

    bounds = [_o2_bounds(cell) for cell in df["o2_pct"]]
    df["o2_frac_min"] = [b[0] for b in bounds]
    df["o2_frac_max"] = [b[1] for b in bounds]

    df["gravity"] = df["gravite"].map(_gravity_regime)
    df["is_simulation"] = df["type"].str.lower().eq("simulation")
    _check_known_rows(df)
    return df


def _check_known_rows(df: pd.DataFrame) -> None:
    """Echoue si une ligne connue pour etre incertaine passe pour certaine."""
    leaking = df[df["id"].isin(KNOWN_UNCERTAIN_PRESSURE) & df["pressure_certain"]]
    if not leaking.empty:
        raise ValueError(
            "literature_crosscheck : les lignes "
            f"{sorted(leaking['id'].tolist())} ont une pression incertaine "
            "(voir remarque) mais le filtre les juge certaines. Si la "
            "cellule a ete levee, retirer l'id de KNOWN_UNCERTAIN_PRESSURE."
        )


def load_fuel_properties() -> pd.DataFrame:
    """Proprietes des carburants, une ligne par (carburant, propriete).

    Format long, comme le fichier : chaque valeur garde sa source, son URL et
    sa citation. `fuel_table()` donne la vue large dont le calcul a besoin.
    """
    df = _add_verification(_read("fuels"))
    df["value"] = pd.to_numeric(df["valeur"], errors="coerce")
    return df


def fuel_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Une ligne par carburant complet, une colonne par propriete.

    Echoue si deux lignes donnent la meme propriete au meme carburant :
    choisir entre deux valeurs est une decision, pas un detail de lecture.
    """
    df = load_fuel_properties() if df is None else df
    duplicated = df[df.duplicated(["combustible", "propriete"], keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "proprietes_combustibles : propriete en double "
            f"{duplicated[['combustible', 'propriete']].values.tolist()}"
        )
    wide = df.pivot(index="combustible", columns="propriete", values="valeur")
    complete = wide.reindex(columns=FUEL_PROPERTIES).dropna()
    numeric = [c for c in FUEL_PROPERTIES if c != "formule"]
    complete[numeric] = complete[numeric].astype(float)
    return complete


def constant(df: pd.DataFrame, substance: str, prop: str) -> float:
    """Une constante hors carburant (eau, oxygene), avec erreur claire."""
    match = df[(df["combustible"] == substance) & (df["propriete"] == prop)]
    if match.empty:
        raise KeyError(f"proprietes_combustibles : {substance}/{prop} absent")
    return float(match["value"].iloc[0])


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "incidents": load_incidents(),
        "atmospheres": load_atmospheres(),
        "gravity": load_gravity(),
        "literature": load_literature(),
        "fuels": load_fuel_properties(),
    }


def main() -> None:
    print(f"data/context : {CONTEXT}\n")
    tables = load_all()
    for name, df in tables.items():
        verified = int(df["verified"].sum()) if "verified" in df else None
        tail = f", verifiees {verified}/{len(df)}" if verified is not None else ""
        print(f"  {name:12} {len(df):3d} lignes{tail}")

    incidents = tables["incidents"]
    print("\nincidents non etablis (source faible) :")
    for _, row in incidents[~incidents["established"]].iterrows():
        print(f"  #{row['id']} {row['date']}  {row['vaisseau_lieu'][:60]}")

    atm = tables["atmospheres"]
    print("\natmospheres (atm, O2) :")
    for _, row in atm.iterrows():
        if pd.isna(row["pressure_atm_min"]):
            pressure = "pression ABSENTE"
        elif row["pressure_is_range"]:
            pressure = f"{row['pressure_atm_min']:.3f}-{row['pressure_atm_max']:.3f} atm"
        else:
            pressure = f"{row['pressure_atm_min']:.3f} atm"
        o2 = "O2 ABSENT" if pd.isna(row["o2_frac"]) else f"O2 {row['o2_frac']:.2f}"
        print(f"  #{row['id']} {row['vaisseau_situation'][:44]:44} {pressure:22} {o2}")

    print("\ncarburants complets :")
    print(fuel_table(tables["fuels"]).to_string())

    lit = tables["literature"]
    print("\nlitterature : pression utilisable pour un chiffre ?")
    for _, row in lit.iterrows():
        flag = "oui" if row["pressure_certain"] else "NON"
        print(f"  #{row['id']:2d} {flag:3}  {str(row['pression_atm'])[:30]:30} "
              f"{row['combustible'][:30]}")


if __name__ == "__main__":
    main()
