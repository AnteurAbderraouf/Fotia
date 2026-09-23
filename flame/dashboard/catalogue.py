"""Le catalogue des investigations — la moitié du livrable.

POURQUOI CE N'EST PAS UN ACCESSOIRE.

Le défi porte sur la fragmentation : les résultats de NASA existent, ils sont
publics, mais ils vivent dans des dizaines d'investigations séparées sans
format commun. Rassembler et rendre comparable EST la demande, pas le travail
préparatoire qui y mène.

Ce module produit donc l'inventaire complet, et pas seulement les jeux qui ont
servi à modéliser. Savoir qu'une investigation ne mène nulle part est un
résultat : c'est du temps que la personne suivante n'aura pas à perdre.

DEUX SOURCES, STRICTEMENT SÉPARÉES.

    ce que NASA dit    scanne depuis les info.md, reproduits verbatim :
                       titre, plateforme, dates, financeur, objectifs.
                       On ne resume pas, on ne corrige pas.

    ce qu'on a trouve  notre evaluation : le jeu porte-t-il des resultats par
                       essai, combien de lignes en sortent, pourquoi certaines
                       pistes s'arretent. C'est notre travail, pas celui de
                       NASA, et l'interface l'affiche comme tel.

Mélanger les deux serait attribuer à NASA des jugements qu'elle n'a pas
portés.

L'INVENTAIRE EST SCANNÉ, PAS ÉCRIT À LA MAIN. Les comptes de fichiers et les
métadonnées viennent du disque à chaque appel. Si un dossier change, le
catalogue suit — une liste figée dans le code aurait diverge au premier ajout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flame.common.paths import GROUND, MICROGRAVITY, PROCESSED

# --- Notre evaluation, investigation par investigation -----------------------
# Ce bloc est le seul contenu redige par nous. Tout le reste est scanne.
VERDICTS: dict[str, tuple[str, str]] = {
    "PSI-69": ("exploite", "213 essais etiquetes. Le seul jeu portant sur la suppression d'incendie."),
    "PSI-39": ("exploite", "146 essais depouilles sur 227. Les 81 autres sont vides, et le farnesane n'a aucune ligne exploitable."),
    "PSI-159": ("exploite", "272 essais etiquetes. Meilleur equilibre du catalogue, plancher 67 %."),
    "PSI-101": ("exploite", "134 essais sur materiaux reels. Contient aussi les resultats de vol de PSI-102."),
    "PSI-107": ("exploite", "70 essais SPICE. La feuille melange trois plateformes de microgravite differentes."),
    "PSI-25": ("partiel", "129 combustions ISS, mais AUCUNE etiquette entrainable : 20 issues explicites sur 129. Catalogue de materiaux."),
    "PSI-117": ("exploite", "141 mesures, donnees derriere les figures de quatre articles. Melange mesures et simulations."),
    "PSI-142": ("exploite", "95 limites d'extinction. Donnees AU SOL, 1 g — ne jamais melanger avec l'ISS."),
    "PSI-99": ("partiel", "9 echantillons seulement. Trop peu pour entrainer, mais seule source comparant microgravite et 1 g sur la meme ligne."),
    "PSI-115": ("partiel", "5 cas simules, 42 grilles de champ. Pas de lignes a modeliser ; sert aux vues 3D."),
    "PSI-10": ("impasse", "327 lignes de conditions, aucune colonne de resultat."),
    "PSI-20": ("impasse", "Journal par jour d'essai (18 jours), pas de resultat par essai."),
    "PSI-21": ("impasse", "Journal par jour d'essai (32 jours)."),
    "PSI-22": ("impasse", "Journal par jour d'essai (22 jours). NASA precise que ce n'est pas destine aux applications spatiales."),
    "PSI-23": ("impasse", "Journal par jour d'essai (18 jours)."),
    "PSI-26": ("impasse", "Liste de 122 essais, materiau en texte libre seulement."),
    "PSI-68": ("impasse", "Matrice d'essais prevus (8 carburants), pas des resultats."),
    "PSI-98": ("impasse", "2 echantillons. Le fichier de 12 Mo est un index de noms d'images."),
    "PSI-100": ("impasse", "2 echantillons, index d'images."),
    "PSI-102": ("impasse", "Matrice d'essais. Ses resultats de vol sont dans le classeur de PSI-101."),
    "PSI-106": ("impasse", "12 lignes : jour d'essai vers melange de carburant. C'est un planning."),
    "PSI-47": ("impasse", "Qualification d'instrument (compteur de particules), pas de combustion."),
    "PSI-60": ("impasse", "Archive de metadonnees seulement."),
    "PSI-62": ("impasse", "Archive de metadonnees seulement. Valide contre BASS, pas contre FLEX."),
}

VERDICT_ORDER = {"exploite": 0, "partiel": 1, "impasse": 2}
VERDICT_LABELS = {
    "exploite": "Exploite",
    "partiel": "Partiellement utilisable",
    "impasse": "Sans resultat par essai",
}

# Tables produites, pour rattacher un compte de lignes a son investigation.
OUTPUTS: dict[str, list[str]] = {
    "PSI-69": ["psi69_suppression.csv"],
    "PSI-39": ["psi39_cool_flames.csv"],
    "PSI-159": ["psi159_sustainment.csv"],
    "PSI-101": ["psi101_detection.csv"],
    "PSI-107": ["psi107_smoke_point.csv"],
    "PSI-25": ["psi25_materials_catalogue.csv"],
    "PSI-117": ["psi117_measurements.csv"],
    "PSI-142": ["psi142_extinction_limits.csv"],
    "PSI-99": ["psi99_microgravity_vs_earth.csv"],
    "PSI-115": ["psi115_field_summary.csv"],
}

FIELDS = {
    "title": r"\*\*Proposal Title:\*\*\s*(.+)",
    "platform": r"\*\*Flight Platform:\*\*\s*(.+)",
    "start": r"\*\*Investigation Start Date:\*\*\s*(.+)",
    "end": r"\*\*Investigation End Date:\*\*\s*(.+)",
    "sponsor": r"\*\*Sponsoring Agency:\*\*\s*(.+)",
    "centre": r"\*\*NASA Center:\*\*\s*(.+)",
}


@dataclass
class Investigation:
    """Une investigation : les faits NASA, puis notre évaluation."""

    psi: str
    category: str
    folder: Path
    nasa: dict
    objectives: str
    files: dict
    verdict: str
    assessment: str
    rows: int | None


def _read_info(folder: Path) -> tuple[dict, str]:
    """Extrait les métadonnées NASA et le premier paragraphe d'objectifs.

    NASA ne renseigne pas de titre partout. Dans ce cas le premier paragraphe
    des objectifs porte la substance, et c'est lui qu'on affiche — toujours
    son texte, jamais un resume de notre fait.
    """
    info = folder / "info.md"
    if not info.exists():
        return {}, ""
    text = info.read_text(encoding="utf-8")

    nasa = {}
    for key, pattern in FIELDS.items():
        found = re.search(pattern, text)
        if found:
            nasa[key] = found.group(1).strip()

    objectives = ""
    section = re.search(r"##\s*Objectives\s*\n(.+?)(?=\n##|\Z)", text, re.S)
    if section:
        for line in section.group(1).strip().splitlines():
            stripped = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            # Une ligne qui se termine par deux-points annonce une liste, elle
            # ne dit rien elle-meme : on passe a la suivante.
            if len(stripped) > 30 and not stripped.rstrip().endswith(":"):
                objectives = stripped
                break
    return nasa, objectives


def _count_files(folder: Path) -> dict:
    return {
        "csv": len(list((folder / "csv").glob("*"))) if (folder / "csv").is_dir() else 0,
        "reports": len(list((folder / "reports").glob("*.pdf")))
        if (folder / "reports").is_dir()
        else 0,
        "fields": len(list((folder / "fields").glob("*.csv")))
        if (folder / "fields").is_dir()
        else 0,
        "archives": len(list(folder.glob("*.zip"))),
    }


def _row_count(psi: str) -> int | None:
    for name in OUTPUTS.get(psi, []):
        path = PROCESSED / name
        if path.exists():
            return int(len(pd.read_csv(path)))
    return None


def scan() -> list[Investigation]:
    """Parcourt le disque et construit l'inventaire."""
    found: list[Investigation] = []
    for parent, category in [(MICROGRAVITY, "microgravite"), (GROUND, "sol")]:
        if not parent.is_dir():
            continue
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir():
                continue
            nasa, objectives = _read_info(folder)
            verdict, assessment = VERDICTS.get(
                folder.name, ("impasse", "Non evaluee.")
            )
            found.append(
                Investigation(
                    psi=folder.name,
                    category=category,
                    folder=folder,
                    nasa=nasa,
                    objectives=objectives,
                    files=_count_files(folder),
                    verdict=verdict,
                    assessment=assessment,
                    rows=_row_count(folder.name),
                )
            )
    return sorted(found, key=lambda i: (VERDICT_ORDER[i.verdict], -(i.rows or 0)))


def table() -> pd.DataFrame:
    """L'inventaire sous forme de tableau."""
    return pd.DataFrame(
        [
            {
                "PSI": item.psi,
                "verdict": VERDICT_LABELS[item.verdict],
                "lignes exploitables": item.rows,
                "categorie": item.category,
                "plateforme": item.nasa.get("platform", ""),
                "periode": " a ".join(
                    x for x in [item.nasa.get("start", ""), item.nasa.get("end", "")] if x
                ),
                "titre NASA": item.nasa.get("title", ""),
                "notre evaluation": item.assessment,
                "csv": item.files["csv"],
                "PDF": item.files["reports"],
            }
            for item in scan()
        ]
    )


def summary() -> dict:
    items = scan()
    usable = [i for i in items if i.verdict != "impasse"]
    return {
        "total": len(items),
        "exploitable": sum(1 for i in items if i.verdict == "exploite"),
        "partiel": sum(1 for i in items if i.verdict == "partiel"),
        "impasse": sum(1 for i in items if i.verdict == "impasse"),
        "rows": sum(i.rows or 0 for i in usable),
        "reports": sum(i.files["reports"] for i in items),
        "csv": sum(i.files["csv"] for i in items),
    }


def read_info_text(psi: str) -> str:
    """Le texte NASA intégral, tel qu'il a été copié — jamais résumé."""
    for parent in (MICROGRAVITY, GROUND):
        info = parent / psi / "info.md"
        if info.exists():
            return info.read_text(encoding="utf-8")
    return ""
