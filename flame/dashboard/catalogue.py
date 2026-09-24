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
#
# LE TEXTE EST EN ANGLAIS PARCE QU'IL EST AFFICHE. Tout ce qui apparait dans le
# tableau de bord est en anglais ; les commentaires et la documentation restent
# en francais. La regle vaut pour ce fichier comme pour les autres.
VERDICTS: dict[str, tuple[str, str]] = {
    "PSI-69": ("exploite", "213 labelled tests. The only dataset here about putting a fire out."),
    "PSI-39": ("exploite", "146 tests written up out of 227. The other 81 are blank, and farnesane has no usable row at all."),
    "PSI-159": ("exploite", "272 labelled tests. The best balanced labels in the catalogue, 67 % floor."),
    "PSI-101": ("exploite", "134 tests on real spacecraft materials. Also carries PSI-102's flight results."),
    "PSI-107": ("exploite", "70 SPICE tests. The sheet mixes three different microgravity platforms."),
    "PSI-25": ("partiel", "129 ISS burns, but NO trainable label: only 20 of them state an outcome. A materials catalogue."),
    "PSI-117": ("exploite", "141 measurements, the data behind the figures of four papers. Mixes measurement and simulation."),
    "PSI-142": ("exploite", "95 extinction limits. GROUND data, 1 g. Never to be pooled with the ISS."),
    "PSI-99": ("partiel", "9 samples only. Far too few to train on, and the only source comparing microgravity with 1 g on one row."),
    "PSI-115": ("partiel", "5 simulated cases, 42 field grids. No rows to model; it feeds the 3D views."),
    "PSI-10": ("impasse", "327 rows of conditions, not one outcome column."),
    "PSI-20": ("impasse", "A log by test day (18 days), with no per-test result."),
    "PSI-21": ("impasse", "A log by test day (32 days)."),
    "PSI-22": ("impasse", "A log by test day (22 days). NASA states it is not aimed at space applications."),
    "PSI-23": ("impasse", "A log by test day (18 days)."),
    "PSI-26": ("impasse", "A list of 122 tests, with the material as free text and nothing else."),
    "PSI-68": ("impasse", "A matrix of planned tests (8 fuels), not of results."),
    "PSI-98": ("impasse", "2 samples. The 12 MB file is an index of image names."),
    "PSI-100": ("impasse", "2 samples, an image index."),
    "PSI-102": ("impasse", "A test matrix. Its flight results live in PSI-101's workbook."),
    "PSI-106": ("impasse", "12 rows: test day against fuel mixture. That is a schedule."),
    "PSI-47": ("impasse", "Instrument qualification (a particle counter), no combustion."),
    "PSI-60": ("impasse", "Metadata archive only."),
    "PSI-62": ("impasse", "Metadata archive only. Validated against BASS, not against FLEX."),
}

VERDICT_ORDER = {"exploite": 0, "partiel": 1, "impasse": 2}
VERDICT_LABELS = {
    "exploite": "Fully used",
    "partiel": "Partly usable",
    "impasse": "No per-test outcome",
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
    for parent, category in [(MICROGRAVITY, "microgravity"), (GROUND, "ground")]:
        if not parent.is_dir():
            continue
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir():
                continue
            nasa, objectives = _read_info(folder)
            verdict, assessment = VERDICTS.get(
                folder.name, ("impasse", "Not assessed.")
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
                "status": VERDICT_LABELS[item.verdict],
                "usable rows": item.rows,
                "category": item.category,
                "platform": item.nasa.get("platform", ""),
                "period": " to ".join(
                    x for x in [item.nasa.get("start", ""), item.nasa.get("end", "")] if x
                ),
                "NASA's title": item.nasa.get("title", ""),
                "what we found": item.assessment,
                "csv files": item.files["csv"],
                "reports": item.files["reports"],
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
